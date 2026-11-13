# -*- coding: utf-8 -*-
import os
import re
import time
from urllib.parse import quote_plus
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
import torch
from flask import Flask, render_template, request

# Import the trained LightGCN module
import comparison_lightgcn as lgcn

app = Flask(__name__)

# 1. Load LightGCN model weights
model_loaded = lgcn.load_model()
if not model_loaded:
    print("[Warning] lightgcn_model.pth not found. Please train or place the checkpoint in root.")

# 2. Initialize SVD baseline model for real-time comparison
rows = lgcn.train_df['user_id_idx'].values
cols = lgcn.train_df['item_id_idx'].values
sparse_mat = sp.csr_matrix(
    (np.ones(len(lgcn.train_df)), (rows, cols)),
    shape=(lgcn.n_users, lgcn.n_items)
)
svd_model = TruncatedSVD(n_components=64, random_state=42)
svd_user_emb = svd_model.fit_transform(sparse_mat)
svd_item_emb = svd_model.components_

# English-to-Chinese movie genre mapping
GENRE_BILINGUAL = {
    "Action": "動作",
    "Adventure": "冒險",
    "Animation": "動畫",
    "Children": "兒童",
    "Children's": "兒童",
    "Comedy": "喜劇",
    "Crime": "犯罪",
    "Documentary": "紀錄片",
    "Drama": "劇情",
    "Fantasy": "奇幻",
    "Film-Noir": "黑色電影",
    "Horror": "恐怖",
    "IMAX": "IMAX",
    "Musical": "音樂劇",
    "Mystery": "懸疑",
    "Romance": "愛情",
    "Sci-Fi": "科幻",
    "Thriller": "驚悚",
    "War": "戰爭",
    "Western": "西部"
}

ALL_GENRES = sorted([
    g for g in {
        genre for genres_set in lgcn.MOVIE_GENRES.values()
        for genre in genres_set
    }
    if g and g not in ('(no genres listed)', 'unknown')
])

AVAILABLE_USERS = sorted([int(u) for u in lgcn.le_user.classes_[:100]])


def format_movie_title(raw_title):
    """Parse raw MovieLens title (e.g., 'Matrix, The (1999)') into clean title and release year."""
    raw_title = str(raw_title).strip()
    year_match = re.search(r'\((\d{4})\)\s*$', raw_title)
    year = year_match.group(1) if year_match else ""
    clean_title = re.sub(r'\s*\(\d{4}\)\s*$', '', raw_title).strip()

    # Handle inverted English articles (e.g., 'Godfather, The' -> 'The Godfather')
    for suffix in [', The', ', A', ', An']:
        if clean_title.endswith(suffix):
            article = suffix.replace(', ', '')
            clean_title = f"{article} {clean_title[:-len(suffix)]}"
            break

    return clean_title, year


def build_imdb_url(clean_title, year):
    """Generate an IMDb title-specific search URL (s=tt)."""
    query = f"{clean_title} {year}".strip()
    return f"https://www.imdb.com/find/?q={quote_plus(query)}&s=tt"


def format_genres(genre_str):
    """Return a list of genre dictionaries with both English and Chinese names."""
    raw_list = [g for g in str(genre_str).split('|') if g and g not in ('(no genres listed)', 'unknown')]
    return [{'en': g, 'zh': GENRE_BILINGUAL.get(g, g)} for g in raw_list]


def compute_list_ild(movie_ids):
    """Compute real-time Intra-List Diversity (ILD) for a recommendation list."""
    genre_sets = [lgcn.MOVIE_GENRES.get(m_id, set()) for m_id in movie_ids if m_id in lgcn.MOVIE_GENRES]
    if len(genre_sets) < 2:
        return 0.0
    total_sim, pairs = 0.0, 0
    for i in range(len(genre_sets)):
        for j in range(i + 1, len(genre_sets)):
            union_len = len(genre_sets[i].union(genre_sets[j]))
            if union_len > 0:
                inter_len = len(genre_sets[i].intersection(genre_sets[j]))
                total_sim += (inter_len / union_len)
                pairs += 1
    return round(1.0 - (total_sim / pairs), 4) if pairs > 0 else 0.0


def get_model_recs(user_id, selected_genres, mode='lightgcn', top_n=10):
    """Retrieve Top-N recommendations, real-time ILD, and per-user inference latency for a given model."""
    t0 = time.time()

    if user_id in lgcn.le_user.classes_:
        user_idx = int(lgcn.le_user.transform([user_id])[0])
    else:
        user_idx = 0

    if mode == 'lightgcn':
        lgcn.lightgcn.eval()
        with torch.no_grad():
            _, out = lgcn.lightgcn(lgcn.train_edge_index)
            u_emb, i_emb = torch.split(out, (lgcn.n_users, lgcn.n_items))
            scores = torch.matmul(i_emb, u_emb[user_idx]).cpu().numpy()
    else:
        scores = np.dot(svd_user_emb[user_idx], svd_item_emb)

    # Mask out movies already seen in the training set
    seen_idxs = lgcn.USER_POS_SET.get(user_idx, set())
    for s_idx in seen_idxs:
        scores[s_idx] = -np.inf

    top_indices = np.argsort(scores)[::-1]
    infer_ms = round((time.time() - t0) * 1000, 3)

    rec_movie_ids = lgcn.le_item.inverse_transform(top_indices)
    rec_movie_ids = [m_id for m_id in rec_movie_ids if m_id in lgcn.movies_df.index]

    df_recs = lgcn.movies_df.loc[rec_movie_ids].copy()
    if selected_genres:
        df_recs['matching_genres'] = df_recs['genres'].apply(
            lambda x: sum(g in str(x).split('|') for g in selected_genres)
        )
        df_recs = df_recs[df_recs['matching_genres'] > 0]
        df_recs = df_recs.sort_values(by='matching_genres', ascending=False)

    top_df = df_recs.head(top_n).copy()
    top_df['movieId'] = top_df.index
    ild_score = compute_list_ild(top_df['movieId'].tolist())

    # Retrieve the user's top-rated historical movies (top 6)
    seen_movie_ids = lgcn.le_item.inverse_transform(list(seen_idxs)[:6])
    history_movies = []
    for m_id in seen_movie_ids:
        if m_id in lgcn.movies_df.index:
            row = lgcn.movies_df.loc[m_id]
            c_title, year = format_movie_title(row['title'])
            history_movies.append({
                'movieId': int(m_id),
                'title': c_title,
                'year': year,
                'imdb_url': build_imdb_url(c_title, year),
                'genres': format_genres(row['genres'])
            })

    recs_list = []
    for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
        c_title, year = format_movie_title(row['title'])
        recs_list.append({
            'rank': f"{rank:02d}",
            'movieId': int(row['movieId']),
            'title': c_title,
            'year': year,
            'imdb_url': build_imdb_url(c_title, year),
            'genres': format_genres(row['genres'])
        })

    return recs_list, f"{ild_score:.4f}", f"{infer_ms:.3f}", history_movies


@app.route('/', methods=['GET', 'POST'])
def index():
    selected_user = AVAILABLE_USERS[0]
    selected_genres = []
    top_n = 10

    if request.method == 'POST':
        try:
            selected_user = int(request.form.get('user_id', AVAILABLE_USERS[0]))
        except ValueError:
            selected_user = AVAILABLE_USERS[0]
        try:
            top_n = int(request.form.get('top_n', 10))
            if top_n not in (10, 20):
                top_n = 10
        except ValueError:
            top_n = 10
        selected_genres = request.form.getlist('genres')

    lgcn_recs, lgcn_ild, lgcn_ms, history = get_model_recs(
        selected_user, selected_genres, mode='lightgcn', top_n=top_n
    )
    svd_recs, svd_ild, svd_ms, _ = get_model_recs(
        selected_user, selected_genres, mode='svd', top_n=top_n
    )

    genre_options = [{'en': g, 'zh': GENRE_BILINGUAL.get(g, g)} for g in ALL_GENRES]

    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    charts = []
    for fname, title_en, title_zh in [
        ('baseline_comparison.png', 'Benchmark Metrics Comparison', '基準模型指標綜合對比'),
        ('scatter_plot.png', 'Accuracy vs. Diversity Trade-off', '準確度與多樣性權衡分析'),
        ('training_time_plot.png', 'Computational Efficiency (Log Scale)', '模型訓練耗時對比 (對數尺度)')
    ]:
        if os.path.exists(os.path.join(static_dir, fname)):
            charts.append({'file': fname, 'en': title_en, 'zh': title_zh})

    return render_template(
        'index.html',
        users=AVAILABLE_USERS,
        selected_user=selected_user,
        top_n=top_n,
        genre_options=genre_options,
        selected_genres=selected_genres,
        history=history,
        lgcn_recs=lgcn_recs,
        lgcn_ild=lgcn_ild,
        lgcn_ms=lgcn_ms,
        svd_recs=svd_recs,
        svd_ild=svd_ild,
        svd_ms=svd_ms,
        charts=charts,
        total_users=lgcn.n_users,
        total_items=lgcn.n_items
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)