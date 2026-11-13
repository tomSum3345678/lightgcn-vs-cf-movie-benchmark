# -*- coding: utf-8 -*-
"""
Baselines: Popularity, SVD, Cosine Similarity & MF-BPR

1. Train / Val / Test Leave-One-Out Split (Ratings >= 4)
2. Masking Seen Items (-np.inf) in the Score Matrix Before Sorting
3. Upgrade MF-BPR to Standard Mini-batch Training (50 Epochs, Batch Size 1024, L2 Regularization)
4. Support @10 & @20 Metrics, Single-User Inference Time, and Perform Paired t-test Against MF-BPR
"""
import os
import time
import random
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sp
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ============ Path Configuration ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'Datasets')
MOVIES_PATH = os.path.join(DATA_DIR, 'movies.csv')
RATINGS_PATH = os.path.join(DATA_DIR, 'ratings.csv')

# ============ Data Loading (Train / Val / Test Leave-One-Out Split) ============
movies_df = pd.read_csv(MOVIES_PATH, index_col='movieId')
ratings_df = pd.read_csv(RATINGS_PATH)
ratings_df = ratings_df[ratings_df['rating'] >= 4].copy()
ratings_df = ratings_df.sort_values(by=['userId', 'timestamp'])

train_list, val_list, test_list = [], [], []
for user_id, group in ratings_df.groupby('userId'):
    if len(group) < 3:
        continue
    train_list.append(group.iloc[:-2])
    val_list.append(group.iloc[-2:-1])
    test_list.append(group.iloc[-1:])

train_df = pd.concat(train_list).reset_index(drop=True)
val_df = pd.concat(val_list).reset_index(drop=True)
test_df = pd.concat(test_list).reset_index(drop=True)

users = train_df['userId'].unique()
items = train_df['movieId'].unique()

user_to_idx = {u: i for i, u in enumerate(users)}
item_to_idx = {movie_id: idx for idx, movie_id in enumerate(items)}
idx_to_item = {idx: movie_id for idx, movie_id in enumerate(items)}

val_df = val_df[val_df['userId'].isin(user_to_idx) & val_df['movieId'].isin(item_to_idx)].copy()
test_df = test_df[test_df['userId'].isin(user_to_idx) & test_df['movieId'].isin(item_to_idx)].copy()

n_users = len(users)
n_items = len(items)
K = 10

MOVIE_GENRES = {m_id: set(str(row['genres']).split('|')) for m_id, row in movies_df.iterrows()}

# Record watched movies and internal indices for each user in Train and Train+Val
train_val_df = pd.concat([train_df, val_df], ignore_index=True)
USER_SEEN_TRAIN_IDX = train_df.groupby('userId')['movieId'].apply(lambda s: [item_to_idx[m] for m in s]).to_dict()
USER_SEEN_TRAIN_SET = {u: set(idxs) for u, idxs in USER_SEEN_TRAIN_IDX.items()}
USER_SEEN_MASK_IDX = train_val_df.groupby('userId')['movieId'].apply(lambda s: [item_to_idx[m] for m in s]).to_dict()
USER_SEEN_MASK_MOVIES = train_val_df.groupby('userId')['movieId'].apply(set).to_dict()

# ============ Metric Calculation Functions ============
def calculate_ild(recommended_item_ids):
    if len(recommended_item_ids) < 2:
        return 0.0
    genre_sets = [MOVIE_GENRES[i] for i in recommended_item_ids if i in MOVIE_GENRES]
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
    return 1.0 - (total_sim / pairs) if pairs > 0 else 0.0

def calculate_ndcg(actual_items, recommended_items, k):
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(recommended_items[:k]) if item in actual_items)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual_items), k)))
    return dcg / idcg if idcg > 0 else 0.0

def get_top20_from_score_matrix(score_matrix):
    """Mask seen items as -inf in the score matrix before extracting Top-20, and precisely measure inference time"""
    t0 = time.time()
    masked_scores = score_matrix.copy()
    for u_idx, user in enumerate(users):
        seen_idxs = USER_SEEN_MASK_IDX.get(user, [])
        masked_scores[u_idx, seen_idxs] = -np.inf

    top20_idx_matrix = np.argpartition(-masked_scores, 20, axis=1)[:, :20]
    recs = {}
    for u_idx, user in enumerate(users):
        cand_idxs = top20_idx_matrix[u_idx]
        sorted_cand = cand_idxs[np.argsort(-masked_scores[u_idx, cand_idxs])]
        recs[user] = [idx_to_item[idx] for idx in sorted_cand]
    infer_per_user = (time.time() - t0) / n_users
    return recs, infer_per_user

def evaluate_recommendations(recommendations, eval_df, infer_per_user):
    p10s, r10s, n10s, ild10s = [], [], [], []
    p20s, r20s, n20s = [], [], []
    recommended_set = set()
    test_grouped = eval_df.groupby('userId')['movieId'].apply(set).to_dict()

    for user in users:
        if user not in test_grouped:
            continue
        actual_items = test_grouped[user]
        rec_20 = recommendations[user][:20]
        rec_10 = rec_20[:10]

        hits_10 = len(set(rec_10) & actual_items)
        hits_20 = len(set(rec_20) & actual_items)

        p10s.append(hits_10 / 10.0)
        r10s.append(hits_10 / len(actual_items) if len(actual_items) > 0 else 0)
        n10s.append(calculate_ndcg(actual_items, rec_10, 10))
        ild10s.append(calculate_ild(rec_10))

        p20s.append(hits_20 / 20.0)
        r20s.append(hits_20 / len(actual_items) if len(actual_items) > 0 else 0)
        n20s.append(calculate_ndcg(actual_items, rec_20, 20))

        for item in rec_10:
            recommended_set.add(item)

    coverage = len(recommended_set) / len(movies_df)
    return {
        'precision': np.mean(p10s),
        'recall': np.mean(r10s),
        'ndcg': np.mean(n10s),
        'precision_20': np.mean(p20s),
        'recall_20': np.mean(r20s),
        'ndcg_20': np.mean(n20s),
        'ild': np.mean(ild10s),
        'coverage': coverage,
        'infer': infer_per_user,
        'user_ndcg': np.array(n10s),
        'user_precision': np.array(p10s)
    }

# ============ Run Baselines ============
results_list = []
times_list = []

# 1. Popularity Baseline
print("Running Popularity Baseline...")
start_time = time.time()
popular_items = train_df['movieId'].value_counts().index.tolist()
pop_time = time.time() - start_time

t_infer = time.time()
pop_recs = {}
for user in users:
    seen = USER_SEEN_MASK_MOVIES.get(user, set())
    rec = []
    for m_id in popular_items:
        if m_id not in seen:
            rec.append(m_id)
            if len(rec) == 20:
                break
    pop_recs[user] = rec
pop_infer_per_user = (time.time() - t_infer) / n_users

res_pop = evaluate_recommendations(pop_recs, test_df, pop_infer_per_user)
res_pop['name'] = "Popularity"
results_list.append(res_pop)
times_list.append(pop_time)

# 2. SVD Baseline (3 Seeds)
print("\nRunning SVD Baseline (3 Seeds)...")
rows = train_df['userId'].map(user_to_idx)
cols = train_df['movieId'].map(item_to_idx)
data = np.ones(len(train_df))
sparse_matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

svd_seeds = [16, 42, 2026]
svd_runs, svd_times = [], []
for seed in svd_seeds:
    start_time = time.time()
    svd = TruncatedSVD(n_components=64, random_state=seed)
    user_embeddings = svd.fit_transform(sparse_matrix)
    item_embeddings = svd.components_
    svd_time = time.time() - start_time

    predicted_scores = np.dot(user_embeddings, item_embeddings)
    svd_recs, svd_infer = get_top20_from_score_matrix(predicted_scores)
    svd_runs.append(evaluate_recommendations(svd_recs, test_df, svd_infer))
    svd_times.append(svd_time)

res_svd = {
    'name': 'SVD',
    'precision': np.mean([r['precision'] for r in svd_runs]),
    'precision_std': np.std([r['precision'] for r in svd_runs]),
    'recall': np.mean([r['recall'] for r in svd_runs]),
    'ndcg': np.mean([r['ndcg'] for r in svd_runs]),
    'precision_20': np.mean([r['precision_20'] for r in svd_runs]),
    'recall_20': np.mean([r['recall_20'] for r in svd_runs]),
    'ndcg_20': np.mean([r['ndcg_20'] for r in svd_runs]),
    'ild': np.mean([r['ild'] for r in svd_runs]),
    'coverage': np.mean([r['coverage'] for r in svd_runs]),
    'infer': np.mean([r['infer'] for r in svd_runs]),
    'user_ndcg': np.mean([r['user_ndcg'] for r in svd_runs], axis=0)
}
results_list.append(res_svd)
times_list.append(np.mean(svd_times))

# 3. Cosine Similarity Baseline (Normalized User-User Cosine Similarity)
print("\nRunning Cosine Similarity Baseline...")
start_time = time.time()
row_norms = np.sqrt(sparse_matrix.multiply(sparse_matrix).sum(axis=1)).A1
row_norms[row_norms == 0] = 1.0
norm_sparse = sp.diags(1.0 / row_norms).dot(sparse_matrix)
user_similarity = norm_sparse.dot(norm_sparse.T)
cosine_scores = user_similarity.dot(sparse_matrix).toarray()
cosine_time = time.time() - start_time

cosine_recs, cos_infer = get_top20_from_score_matrix(cosine_scores)
res_cos = evaluate_recommendations(cosine_recs, test_df, cos_infer)
res_cos['name'] = "Cosine"
results_list.append(res_cos)
times_list.append(cosine_time)

# 4. MF-BPR Baseline (Aligned with LightGCN Mini-batch Training Standard, 3 Seeds)
print("\nRunning MF-BPR Baseline (3 Seeds, Mini-batch)...")
class MF_BPR(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def forward(self, u, i, j):
        u_e, i_e, j_e = self.user_emb(u), self.item_emb(i), self.item_emb(j)
        pos_scores = (u_e * i_e).sum(dim=1)
        neg_scores = (u_e * j_e).sum(dim=1)
        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))
        reg_loss = 0.5 * (u_e.norm().pow(2) + i_e.norm().pow(2) + j_e.norm().pow(2)) / float(len(u))
        return bpr_loss, reg_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS_MF = 50
BATCH_SIZE_MF = 1024
DECAY_MF = 0.0001
LR_MF = 0.005

def sample_mf_batch(batch_size):
    u_indices = [random.choice(range(n_users)) for _ in range(batch_size)] if n_users < batch_size else random.sample(range(n_users), batch_size)
    pos_i, neg_j = [], []
    for u_idx in u_indices:
        u_id = users[u_idx]
        pos_list = USER_SEEN_TRAIN_IDX[u_id]
        pos_set = USER_SEEN_TRAIN_SET[u_id]
        pos_i.append(random.choice(pos_list))
        while True:
            neg_idx = random.randint(0, n_items - 1)
            if neg_idx not in pos_set:
                neg_j.append(neg_idx)
                break
    return (
        torch.LongTensor(u_indices).to(device),
        torch.LongTensor(pos_i).to(device),
        torch.LongTensor(neg_j).to(device)
    )

mf_seeds = [16, 42, 2026]
mf_results, mf_times = [], []
n_batch_mf = max(1, int(len(train_df) / BATCH_SIZE_MF))

for seed in mf_seeds:
    print(f"  -> Training MF-BPR with Seed {seed}...")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    start_time = time.time()

    mf_bpr = MF_BPR(n_users, n_items, dim=64).to(device)
    optimizer_mf = optim.Adam(mf_bpr.parameters(), lr=LR_MF)

    for _ in range(EPOCHS_MF):
        mf_bpr.train()
        for _ in range(n_batch_mf):
            optimizer_mf.zero_grad()
            u_b, i_b, j_b = sample_mf_batch(BATCH_SIZE_MF)
            bpr_loss, reg_loss = mf_bpr(u_b, i_b, j_b)
            loss = bpr_loss + DECAY_MF * reg_loss
            loss.backward()
            optimizer_mf.step()

    mf_time = time.time() - start_time
    mf_bpr.eval()
    with torch.no_grad():
        mf_scores = torch.matmul(mf_bpr.user_emb.weight, mf_bpr.item_emb.weight.T).cpu().numpy()

    mf_recs, mf_infer = get_top20_from_score_matrix(mf_scores)
    mf_results.append(evaluate_recommendations(mf_recs, test_df, mf_infer))
    mf_times.append(mf_time)

res_mf = {
    'name': 'MF-BPR',
    'precision': np.mean([r['precision'] for r in mf_results]),
    'precision_std': np.std([r['precision'] for r in mf_results]),
    'recall': np.mean([r['recall'] for r in mf_results]),
    'recall_std': np.std([r['recall'] for r in mf_results]),
    'ndcg': np.mean([r['ndcg'] for r in mf_results]),
    'ndcg_std': np.std([r['ndcg'] for r in mf_results]),
    'precision_20': np.mean([r['precision_20'] for r in mf_results]),
    'recall_20': np.mean([r['recall_20'] for r in mf_results]),
    'ndcg_20': np.mean([r['ndcg_20'] for r in mf_results]),
    'ild': np.mean([r['ild'] for r in mf_results]),
    'ild_std': np.std([r['ild'] for r in mf_results]),
    'coverage': np.mean([r['coverage'] for r in mf_results]),
    'coverage_std': np.std([r['coverage'] for r in mf_results]),
    'infer': np.mean([r['infer'] for r in mf_results]),
    'user_ndcg': np.mean([r['user_ndcg'] for r in mf_results], axis=0)
}
results_list.append(res_mf)
times_list.append(np.mean(mf_times))

# ============ Terminal Output Table ============
print("\n\n" + "=" * 95)
print("FINAL BASELINES RESULTS (Mean ± Std)")
print("=" * 95)
for r in results_list:
    p_std_str = f"± {r['precision_std']:.4f}" if 'precision_std' in r else "        "
    print(
        f"{r['name']:12s} | P@10: {r['precision']:.4f} {p_std_str} | R@10: {r['recall']:.4f} | "
        f"NDCG@10: {r['ndcg']:.4f} | NDCG@20: {r['ndcg_20']:.4f} | ILD@10: {r['ild']:.4f} | "
        f"Cov: {r['coverage']*100:.2f}% | Infer/User: {r['infer']*1000:.4f}ms"
    )
print("=" * 95)

# Paired t-test: MF-BPR vs Other Baselines (on User-level NDCG@10)
print("\nPAIRED T-TEST (MF-BPR vs Other Baselines on NDCG@10):")
for r in results_list:
    if r['name'] == 'MF-BPR':
        continue
    t_stat, p_val = stats.ttest_rel(res_mf['user_ndcg'], r['user_ndcg'])
    print(f"  MF-BPR vs {r['name']:10s}: t-stat = {t_stat:+.4f}, p-value = {p_val:.4e}")

# ============ Plotting (If comparison_lightgcn.py has finished running, update the dictionary below with the latest values) ============
lightgcn_results = {
    'name': 'LightGCN',
    'precision': 0.0138,
    'recall': 0.1381,
    'ndcg': 0.0679,
    'ild': 0.7737,
    'coverage': 0.2903,   # 29.03% to decimal
    'infer': 0.0000017    # to seconds
}
lightgcn_time = 22.34

results_list.insert(0, lightgcn_results)
times_list.insert(0, lightgcn_time)

models = [r['name'] for r in results_list]
metrics = ['precision', 'recall', 'ndcg', 'ild', 'coverage']
titles = ['Precision@10', 'Recall@10', 'NDCG@10', 'ILD@10 (Diversity)', 'Coverage@10']
colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, metric in enumerate(metrics):
    values = [r[metric] for r in results_list]
    bars = axes[i].bar(models, values, color=colors)
    axes[i].set_title(titles[i], fontsize=14, fontweight='bold')
    axes[i].set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
    for bar in bars:
        yval = bar.get_height()
        label = f'{yval*100:.2f}%' if metric == 'coverage' else f'{yval:.4f}'
        axes[i].text(bar.get_x() + bar.get_width() / 2.0, yval, label, ha='center', va='bottom', fontsize=10, fontweight='bold')

fig.delaxes(axes[5])
plt.tight_layout()
plt.savefig('baseline_comparison.png')
plt.close()

plt.figure(figsize=(8, 6))
precision_vals = [r['precision'] for r in results_list]
ild_vals = [r['ild'] for r in results_list]
plt.scatter(precision_vals, ild_vals, color=colors, s=200, alpha=0.8, edgecolors='black')
for i, model in enumerate(models):
    plt.annotate(model, (precision_vals[i], ild_vals[i]), textcoords="offset points", xytext=(0, 12), ha='center', fontsize=11, fontweight='bold')
plt.xlabel('Precision@10 (Accuracy)', fontsize=12)
plt.ylabel('ILD@10 (Diversity)', fontsize=12)
plt.title('Accuracy vs Diversity Trade-off', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('scatter_plot.png')
plt.close()

plt.figure(figsize=(10, 6))
log_time = [max(t, 0.001) for t in times_list]
bars = plt.bar(models, log_time, color=colors)
plt.yscale('log')
plt.ylabel('Training Time in Seconds (Log Scale)', fontsize=12)
plt.title('Training Time Comparison', fontsize=14, fontweight='bold')
for bar, real_time in zip(bars, times_list):
    yval = bar.get_height()
    label = f'{real_time:.4f}s' if real_time > 0 else '<0.001s'
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval, label, ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.savefig('training_time_plot.png')
plt.close()
print("\nAll models evaluated and ALL charts generated!")