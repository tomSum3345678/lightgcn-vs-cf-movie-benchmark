# -*- coding: utf-8 -*-
"""
Comprehensive Comparative Experiment: LightGCN (Baseline) vs. LightGCN (with Diversity Regularization) + Layer Analysis

Experiment Setup & Key Features:
1. Train / Val / Test Leave-One-Out split (Ratings >= 4)
2. Evaluation metrics covering @10 and @20 (Precision, Recall, NDCG), ILD@10, and Coverage@10
3. Precise measurement of Training Time and Inference Time per User
4. 3-Seed evaluation (Mean ± Std) and Paired t-test for statistical significance
"""
import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from tqdm import tqdm
from sklearn import preprocessing as pp
from scipy import stats


# ============ Path Configuration ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'Datasets')  # Corresponds to ml-latest-small (compatible with 100K/1M)
MOVIES_PATH = os.path.join(DATA_DIR, 'movies.csv')
RATINGS_PATH = os.path.join(DATA_DIR, 'ratings.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'lightgcn_model.pth')


# ============ Device ============
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[LightGCN Comparison] Using device: {device}")


# ============ Data Loading (Train / Val / Test Leave-One-Out Split) ============
movies_df = pd.read_csv(MOVIES_PATH, index_col='movieId')
ratings_df = pd.read_csv(RATINGS_PATH)

# Filter for ratings >= 4 to represent positive implicit feedback.
ratings_df = ratings_df[ratings_df['rating'] >= 4].copy()
ratings_df = ratings_df.sort_values(by=['userId', 'timestamp'])

train_list, val_list, test_list = [], [], []
for user_id, group in ratings_df.groupby('userId'):
    if len(group) < 3:
        continue  # Require at least 3 interactions to split into Train / Val / Test
    train_list.append(group.iloc[:-2])    # All interactions before the second-to-last for Train
    val_list.append(group.iloc[-2:-1])    # Second-to-last interaction for Validation
    test_list.append(group.iloc[-1:])     # Last interaction for Test

train_df = pd.concat(train_list).reset_index(drop=True)
val_df = pd.concat(val_list).reset_index(drop=True)
test_df = pd.concat(test_list).reset_index(drop=True)

le_user = pp.LabelEncoder()
le_item = pp.LabelEncoder()
train_df['user_id_idx'] = le_user.fit_transform(train_df['userId'].values)
train_df['item_id_idx'] = le_item.fit_transform(train_df['movieId'].values)

train_user_ids = set(train_df['userId'].unique())
train_item_ids = set(train_df['movieId'].unique())

# Filter Val and Test sets to ensure all userIds and movieIds appear in the training set
val_df = val_df[val_df['userId'].isin(train_user_ids) & val_df['movieId'].isin(train_item_ids)].copy()
val_df['user_id_idx'] = le_user.transform(val_df['userId'].values)
val_df['item_id_idx'] = le_item.transform(val_df['movieId'].values)

test_df = test_df[test_df['userId'].isin(train_user_ids) & test_df['movieId'].isin(train_item_ids)].copy()
test_df['user_id_idx'] = le_user.transform(test_df['userId'].values)
test_df['item_id_idx'] = le_item.transform(test_df['movieId'].values)

n_users = train_df['user_id_idx'].nunique()
n_items = train_df['item_id_idx'].nunique()
print(f"[LightGCN Comparison] n_users={n_users}, n_items={n_items}, train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

# Pre-build genre and user positive interaction dictionaries (significantly speeds up ILD and negative sampling)
MOVIE_GENRES = {
    m_id: set(str(row['genres']).split('|'))
    for m_id, row in movies_df.iterrows()
}
IDX_TO_GENRES = {
    idx: MOVIE_GENRES.get(m_id, set())
    for idx, m_id in enumerate(le_item.classes_)
}
USER_POS_LIST = train_df.groupby('user_id_idx')['item_id_idx'].apply(list).to_dict()
USER_POS_SET = {u: set(items) for u, items in USER_POS_LIST.items()}


# ============ Hyperparameters ============
latent_dim = 64
n_layers = 3
EPOCHS = 50
BATCH_SIZE = 1024
DECAY = 0.0001
LR = 0.005
K = 10  # Primary metric @10; @20 is also computed automatically


# ============ Fast Data Loader ============
def data_loader(data, batch_size, n_usr, n_itm):
    indices = list(range(n_usr))
    if n_usr < batch_size:
        users = [random.choice(indices) for _ in range(batch_size)]
    else:
        users = random.sample(indices, batch_size)
    users.sort()

    pos_items, neg_items = [], []
    for u in users:
        pos_list = USER_POS_LIST[u]
        pos_set = USER_POS_SET[u]
        pos_items.append(random.choice(pos_list))
        while True:
            neg_id = random.randint(0, n_itm - 1)
            if neg_id not in pos_set:
                neg_items.append(neg_id)
                break

    return (
        torch.LongTensor(users).to(device),
        torch.LongTensor(pos_items).to(device) + n_usr,
        torch.LongTensor(neg_items).to(device) + n_usr
    )


# ============ Edge index ============
u_t = torch.LongTensor(train_df.user_id_idx.values)
i_t = torch.LongTensor(train_df.item_id_idx.values) + n_users
train_edge_index = torch.stack((
    torch.cat([u_t, i_t]),
    torch.cat([i_t, u_t])
)).to(device)


# ============ LightGCN Layer ============
class LightGCNConv(MessagePassing):
    def __init__(self, **kwargs):
        super().__init__(aggr='add')

    def forward(self, x, edge_index):
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


# ============ Model ============
class RecSysGNN(nn.Module):
    def __init__(self, latent_dim, num_layers, num_users, num_items, model='LightGCN'):
        super(RecSysGNN, self).__init__()
        assert model == 'LightGCN', 'Model must be LightGCN'
        self.model = model
        self.embedding = nn.Embedding(num_users + num_items, latent_dim)
        self.convs = nn.ModuleList(LightGCNConv() for _ in range(num_layers))
        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, edge_index):
        emb0 = self.embedding.weight
        embs = [emb0]
        emb = emb0
        for conv in self.convs:
            emb = conv(x=emb, edge_index=edge_index)
            embs.append(emb)
        out = torch.mean(torch.stack(embs, dim=0), dim=0)
        return emb0, out

    def encode_minibatch(self, users, pos_items, neg_items, edge_index):
        emb0, out = self(edge_index)
        return (
            out[users],
            out[pos_items],
            out[neg_items],
            emb0[users],
            emb0[pos_items],
            emb0[neg_items]
        )


# ============ Loss (Support Diversity Regularization) ============
def compute_bpr_loss(users, users_emb, pos_emb, neg_emb, user_emb0, pos_emb0, neg_emb0, diversity_lambda):
    reg_loss = (1 / 2) * (
        user_emb0.norm().pow(2) +
        pos_emb0.norm().pow(2) +
        neg_emb0.norm().pow(2)
    ) / float(len(users))

    pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
    neg_scores = torch.mul(users_emb, neg_emb).sum(dim=1)
    bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))

    if diversity_lambda > 0:
        pos_norm = F.normalize(pos_emb, p=2, dim=1)
        sim_matrix = torch.matmul(pos_norm, pos_norm.T)
        diag_mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device).bool()
        sim_matrix = sim_matrix.masked_fill(diag_mask, 0)
        diversity_penalty = torch.mean(torch.abs(sim_matrix))
        bpr_loss = bpr_loss + (diversity_lambda * diversity_penalty)

    return bpr_loss, reg_loss


# ============ Functions for calculating ILD and NDCG ============
def calculate_ild_from_indices(recommended_item_indices):
    """Fast ILD calculation using the pre-built IDX_TO_GENRES dictionary."""
    if len(recommended_item_indices) < 2:
        return 0.0
    genre_sets = [IDX_TO_GENRES[idx] for idx in recommended_item_indices if idx in IDX_TO_GENRES]
    if len(genre_sets) < 2:
        return 0.0

    total_similarity = 0.0
    pairs = 0
    for i in range(len(genre_sets)):
        for j in range(i + 1, len(genre_sets)):
            union_len = len(genre_sets[i].union(genre_sets[j]))
            if union_len > 0:
                inter_len = len(genre_sets[i].intersection(genre_sets[j]))
                total_similarity += (inter_len / union_len)
                pairs += 1
    if pairs == 0:
        return 0.0
    return 1.0 - (total_similarity / pairs)


def calculate_ndcg(actual_items_set, recommended_items, k):
    dcg = 0.0
    for i, item in enumerate(recommended_items[:k]):
        if item in actual_items_set:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual_items_set), k)))
    return dcg / idcg if idcg > 0 else 0.0


# ============ Metrics (Supports both @10 and @20, -inf Masking and inference time) ============
def get_metrics(user_Embed_wts, item_Embed_wts, n_users, n_items, mask_df, eval_df, K=10, return_user_metrics=False):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_infer = time.time()

    relevance_score = torch.matmul(user_Embed_wts, torch.transpose(item_Embed_wts, 0, 1))

    # Create a mask for the watched movie and set it to -inf
    i = torch.stack((
        torch.LongTensor(mask_df['user_id_idx'].values),
        torch.LongTensor(mask_df['item_id_idx'].values)
    )).to(device)
    interactions_t = torch.sparse_coo_tensor(
        i, torch.ones(len(mask_df), dtype=torch.bool, device=device),
        (n_users, n_items), device=device
    ).to_dense()
    relevance_score = relevance_score.masked_fill(interactions_t, float('-inf'))

    # Take the Top-20 at once to calculate both @10 and @20 simultaneously
    max_k = max(K, 20)
    topk_indices = torch.topk(relevance_score, max_k, dim=1).indices.cpu().numpy()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_time_total = time.time() - start_infer
    infer_time_per_user = infer_time_total / n_users

    eval_grouped = eval_df.groupby('user_id_idx')['item_id_idx'].apply(set).to_dict()

    p10_list, r10_list, ndcg10_list, ild10_list = [], [], [], []
    p20_list, r20_list, ndcg20_list = [], [], []
    recommended_items_set = set()

    for user_idx, actual_items in eval_grouped.items():
        if not actual_items:
            continue
        rec_20 = topk_indices[user_idx]
        rec_10 = rec_20[:10]

        for item_idx in rec_10:
            recommended_items_set.add(item_idx)

        hits_10 = len(set(rec_10).intersection(actual_items))
        hits_20 = len(set(rec_20).intersection(actual_items))

        p10_list.append(hits_10 / 10.0)
        r10_list.append(hits_10 / len(actual_items))
        ndcg10_list.append(calculate_ndcg(actual_items, rec_10, 10))
        ild10_list.append(calculate_ild_from_indices(rec_10))

        p20_list.append(hits_20 / 20.0)
        r20_list.append(hits_20 / len(actual_items))
        ndcg20_list.append(calculate_ndcg(actual_items, rec_20, 20))

    coverage = len(recommended_items_set) / len(movies_df)

    summary = {
        'recall': np.mean(r10_list),
        'precision': np.mean(p10_list),
        'ild': np.mean(ild10_list),
        'ndcg': np.mean(ndcg10_list),
        'recall_20': np.mean(r20_list),
        'precision_20': np.mean(p20_list),
        'ndcg_20': np.mean(ndcg20_list),
        'coverage': coverage,
        'infer_total': infer_time_total,
        'infer_per_user': infer_time_per_user
    }

    if return_user_metrics:
        user_level = {
            'precision': np.array(p10_list),
            'recall': np.array(r10_list),
            'ndcg': np.array(ndcg10_list),
            'ild': np.array(ild10_list)
        }
        return summary, user_level
    return summary


# ============ Training and Validation ============
def train_and_eval(model, optimizer, train_df, diversity_lambda):
    loss_list_epoch, bpr_loss_list_epoch, reg_loss_list_epoch = [], [], []
    recall_list, precision_list, ild_list_epoch, ndcg_list_epoch = [], [], [], []
    coverage_list_epoch, infer_time_list_epoch = [], []

    pure_training_time = 0.0
    train_val_mask_df = pd.concat([train_df, val_df], ignore_index=True)

    for epoch in tqdm(range(EPOCHS), desc=f"Lambda={diversity_lambda}"):
        n_batch = max(1, int(len(train_df) / BATCH_SIZE))
        final_loss_list, bpr_loss_list, reg_loss_list = [], [], []

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()

        model.train()
        for _ in range(n_batch):
            optimizer.zero_grad()
            users, pos_items, neg_items = data_loader(train_df, BATCH_SIZE, n_users, n_items)
            users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0 = model.encode_minibatch(
                users, pos_items, neg_items, train_edge_index
            )
            bpr_loss, reg_loss = compute_bpr_loss(
                users, users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0, diversity_lambda
            )
            reg_loss = DECAY * reg_loss
            final_loss = bpr_loss + reg_loss

            final_loss.backward()
            optimizer.step()

            final_loss_list.append(final_loss.item())
            bpr_loss_list.append(bpr_loss.item())
            reg_loss_list.append(reg_loss.item())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pure_training_time += (time.time() - t0)

        # Log test set convergence curves at the end of each epoch (masking train and validation items).
        model.eval()
        with torch.no_grad():
            _, out = model(train_edge_index)
            final_user_Embed, final_item_Embed = torch.split(out, (n_users, n_items))
            metrics = get_metrics(
                final_user_Embed, final_item_Embed, n_users, n_items,
                train_val_mask_df, test_df, K
            )

        loss_list_epoch.append(round(np.mean(final_loss_list), 4))
        bpr_loss_list_epoch.append(round(np.mean(bpr_loss_list), 4))
        reg_loss_list_epoch.append(round(np.mean(reg_loss_list), 4))
        recall_list.append(round(metrics['recall'], 4))
        precision_list.append(round(metrics['precision'], 4))
        ild_list_epoch.append(round(metrics['ild'], 4))
        ndcg_list_epoch.append(round(metrics['ndcg'], 4))
        coverage_list_epoch.append(round(metrics['coverage'], 4))
        infer_time_list_epoch.append(round(metrics['infer_per_user'], 6))

    # Retrieve full @10, @20, and user-level metrics at the final epoch for t-test
    model.eval()
    with torch.no_grad():
        _, out = model(train_edge_index)
        final_user_Embed, final_item_Embed = torch.split(out, (n_users, n_items))
        val_metrics = get_metrics(final_user_Embed, final_item_Embed, n_users, n_items, train_df, val_df, K)
        final_test_metrics, user_level_metrics = get_metrics(
            final_user_Embed, final_item_Embed, n_users, n_items,
            train_val_mask_df, test_df, K, return_user_metrics=True
        )

    return (
        loss_list_epoch, bpr_loss_list_epoch, reg_loss_list_epoch,
        recall_list, precision_list, ild_list_epoch, ndcg_list_epoch,
        coverage_list_epoch, infer_time_list_epoch,
        pure_training_time, val_metrics, final_test_metrics, user_level_metrics
    )


# ============ Initialize global model for Flask deployment ============
lightgcn = RecSysGNN(
    latent_dim=latent_dim,
    num_layers=n_layers,
    num_users=n_users,
    num_items=n_items,
    model='LightGCN'
).to(device)


def train_model(seed=16, diversity_lambda=0.0, num_layers=n_layers):
    """Run the full training pipeline, including pure training time tracking and model checkpointing."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    global lightgcn
    lightgcn = RecSysGNN(
        latent_dim=latent_dim,
        num_layers=num_layers,
        num_users=n_users,
        num_items=n_items,
        model='LightGCN'
    ).to(device)

    optimizer = torch.optim.Adam(lightgcn.parameters(), lr=LR)
    results = train_and_eval(lightgcn, optimizer, train_df, diversity_lambda)
    training_time = results[9]

    model_info = {
        'model_architecture': 'LightGCN',
        'latent_dim': latent_dim,
        'num_layers': num_layers,
        'num_users': n_users,
        'num_items': n_items,
        'training_params': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'decay': DECAY,
            'learning_rate': LR,
            'top_k': K,
            'diversity_lambda': diversity_lambda
        },
        'training_time_seconds': training_time
    }
    torch.save({
        'model_state_dict': lightgcn.state_dict(),
        'model_info': model_info
    }, MODEL_PATH)
    print(f"[LightGCN] Model saved to {MODEL_PATH}")
    return results


def load_model():
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        lightgcn.load_state_dict(checkpoint['model_state_dict'])
        lightgcn.eval()
        print(f"[LightGCN] Model loaded from {MODEL_PATH}")
        return True
    else:
        print(f"[LightGCN] No saved model found at {MODEL_PATH}")
        return False


# ============ Recommendation (for Flask demo calls) ============
def get_top_recommendations(user_id, selected_genres, model, movies_df, le_item, n_items, top_n=10):
    model.eval()
    with torch.no_grad():
        _, out = model(train_edge_index)
        final_user_Embed, final_item_Embed = torch.split(out, (n_users, n_items))

    # Convert raw userId to internal index if applicable
    if user_id in le_user.classes_:
        user_idx = int(le_user.transform([user_id])[0])
    else:
        user_idx = min(max(int(user_id), 0), n_users - 1)

    user_embedding = final_user_Embed[user_idx]
    scores = torch.matmul(final_item_Embed, user_embedding)

    # Mask out movies the user has already seen in the training set
    seen_items = USER_POS_SET.get(user_idx, set())
    for seen_idx in seen_items:
        scores[seen_idx] = float('-inf')

    top_indices = torch.argsort(scores, descending=True).cpu().numpy()
    recommended_movie_ids = le_item.inverse_transform(top_indices)
    recommended_movie_ids = [m_id for m_id in recommended_movie_ids if m_id in movies_df.index]

    recommendations = movies_df.loc[recommended_movie_ids].copy()
    if selected_genres:
        recommendations['matching_genres'] = recommendations['genres'].apply(
            lambda x: sum(genre in str(x) for genre in selected_genres)
        )
        recommendations = recommendations[recommendations['matching_genres'] > 0]
        recommendations = recommendations.sort_values(by='matching_genres', ascending=False)

    top_recommendations = recommendations.head(top_n).copy()
    top_recommendations['movieId'] = top_recommendations.index
    top_recommendations.reset_index(drop=True, inplace=True)
    return top_recommendations[['movieId', 'title']]


# ============ Execution ============
if __name__ == '__main__':
    seeds = [16, 42, 2026]
    configs = [
        {'name': 'Original', 'lambda': 0.0},
        {'name': 'Improved', 'lambda': 0.1}
    ]

    final_report = {}
    user_metrics_store = {}

    for config in configs:
        print(f"\n\n" + "=" * 65)
        print(f"Starting Experiment: {config['name']} (Diversity Lambda = {config['lambda']})")
        print("=" * 65)

        all_results = []
        for seed in seeds:
            print(f"\n--- Seed: {seed} ---")
            result = train_model(seed=seed, diversity_lambda=config['lambda'])
            all_results.append(result)
            test_m = result[11]
            print(
                f"[Seed {seed}] Val NDCG@10: {result[10]['ndcg']:.4f} | "
                f"Test P@10: {test_m['precision']:.4f}, R@10: {test_m['recall']:.4f}, "
                f"NDCG@10: {test_m['ndcg']:.4f}, NDCG@20: {test_m['ndcg_20']:.4f}, "
                f"ILD@10: {test_m['ild']:.4f}, Cov: {test_m['coverage']*100:.2f}%"
            )

        # Aggregate Test Set metrics across the 3 seeds
        p10s = [r[11]['precision'] for r in all_results]
        r10s = [r[11]['recall'] for r in all_results]
        n10s = [r[11]['ndcg'] for r in all_results]
        p20s = [r[11]['precision_20'] for r in all_results]
        r20s = [r[11]['recall_20'] for r in all_results]
        n20s = [r[11]['ndcg_20'] for r in all_results]
        ilds = [r[11]['ild'] for r in all_results]
        covs = [r[11]['coverage'] for r in all_results]
        infers_user = [r[11]['infer_per_user'] for r in all_results]
        times = [r[9] for r in all_results]

        final_report[config['name']] = {
            'Precision@10': (np.mean(p10s), np.std(p10s), p10s),
            'Recall@10': (np.mean(r10s), np.std(r10s), r10s),
            'NDCG@10': (np.mean(n10s), np.std(n10s), n10s),
            'Precision@20': (np.mean(p20s), np.std(p20s), p20s),
            'Recall@20': (np.mean(r20s), np.std(r20s), r20s),
            'NDCG@20': (np.mean(n20s), np.std(n20s), n20s),
            'ILD@10': (np.mean(ilds), np.std(ilds), ilds),
            'Coverage': (np.mean(covs), np.std(covs), covs),
            'Inference/User': (np.mean(infers_user), np.std(infers_user), infers_user),
            'Train Time': (np.mean(times), np.std(times), times)
        }

        # Store per-user metrics averaged across the 3 seeds for paired t-tests
        user_metrics_store[config['name']] = {
            metric_key: np.mean([r[12][metric_key] for r in all_results], axis=0)
            for metric_key in ['precision', 'recall', 'ndcg', 'ild']
        }

        # Plot learning curves (with Mean ± Std shaded bands)
        epoch_list = list(range(1, EPOCHS + 1))
        all_loss_arr = np.array([r[0] for r in all_results])
        all_bpr_arr = np.array([r[1] for r in all_results])
        all_reg_arr = np.array([r[2] for r in all_results])
        all_recall_arr = np.array([r[3] for r in all_results])
        all_precision_arr = np.array([r[4] for r in all_results])
        all_ild_arr = np.array([r[5] for r in all_results])
        all_ndcg_arr = np.array([r[6] for r in all_results])

        plt.figure(figsize=(10, 6))
        for arr, label, color in [
            (all_loss_arr, 'Total Training Loss', 'tab:blue'),
            (all_bpr_arr, 'BPR Training Loss', 'tab:orange'),
            (all_reg_arr, 'Reg Training Loss', 'tab:green')
        ]:
            m, s = np.mean(arr, axis=0), np.std(arr, axis=0)
            plt.plot(epoch_list, m, label=label, color=color)
            plt.fill_between(epoch_list, m - s, m + s, color=color, alpha=0.2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss over Epochs ({config["name"]})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'training_loss_{config["name"]}.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        for arr, label, color in [
            (all_recall_arr, f'Recall@{K}', 'tab:blue'),
            (all_precision_arr, f'Precision@{K}', 'tab:orange'),
            (all_ndcg_arr, f'NDCG@{K}', 'tab:green'),
            (all_ild_arr, f'ILD@{K}', 'tab:red')
        ]:
            m, s = np.mean(arr, axis=0), np.std(arr, axis=0)
            plt.plot(epoch_list, m, label=label, color=color)
            plt.fill_between(epoch_list, m - s, m + s, color=color, alpha=0.2)
        plt.xlabel('Epoch')
        plt.ylabel('Metrics')
        plt.title(f'Evaluation Metrics@{K} over Epochs ({config["name"]})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'evaluation_metrics_{config["name"]}.png')
        plt.close()

    # ============ Final Comparison and Paired t-test ============
    print("\n\n" + "=" * 80)
    print("FINAL COMPARISON: ORIGINAL vs IMPROVED (Mean ± Std across 3 Seeds)")
    print("=" * 80)
    for name, data in final_report.items():
        print(f"\n[{name}]")
        print(f"  Precision@10:   {data['Precision@10'][0]:.4f} ± {data['Precision@10'][1]:.4f} | Precision@20: {data['Precision@20'][0]:.4f} ± {data['Precision@20'][1]:.4f}")
        print(f"  Recall@10:      {data['Recall@10'][0]:.4f} ± {data['Recall@10'][1]:.4f} | Recall@20:    {data['Recall@20'][0]:.4f} ± {data['Recall@20'][1]:.4f}")
        print(f"  NDCG@10:        {data['NDCG@10'][0]:.4f} ± {data['NDCG@10'][1]:.4f} | NDCG@20:      {data['NDCG@20'][0]:.4f} ± {data['NDCG@20'][1]:.4f}")
        print(f"  ILD@10:         {data['ILD@10'][0]:.4f} ± {data['ILD@10'][1]:.4f}")
        print(f"  Coverage@10:    {data['Coverage'][0]*100:.2f}% ± {data['Coverage'][1]*100:.2f}%")
        print(f"  Infer/User:     {data['Inference/User'][0]*1000:.4f} ms ± {data['Inference/User'][1]*1000:.4f} ms")
        print(f"  Train Time:     {data['Train Time'][0]:.2f}s ± {data['Train Time'][1]:.2f}s")

    print("\n" + "-" * 80)
    print("PAIRED T-TEST (Original vs Improved across Test Users)")
    print("-" * 80)
    for metric_key, disp_name in [('precision', 'Precision@10'), ('recall', 'Recall@10'), ('ndcg', 'NDCG@10'), ('ild', 'ILD@10')]:
        t_stat, p_val = stats.ttest_rel(
            user_metrics_store['Improved'][metric_key],
            user_metrics_store['Original'][metric_key]
        )
        sig = "Significant (p < 0.05)" if p_val < 0.05 else "Not Significant (p >= 0.05)"
        print(f"  {disp_name:14s}: t-stat = {t_stat:+.4f}, p-value = {p_val:.4e} -> {sig}")

    # ============ Layer Analysis (1, 2, 3, 4 Layers) ============
    print("\n" + "=" * 80)
    print("LIGHTGCN LAYER ANALYSIS (Layers = 1, 2, 3, 4 | Seed = 16)")
    print("=" * 80)
    for l_num in [1, 2, 3, 4]:
        res_layer = train_model(seed=16, diversity_lambda=0.0, num_layers=l_num)
        m_layer = res_layer[11]
        print(
            f"  Layers={l_num} | P@10: {m_layer['precision']:.4f} | R@10: {m_layer['recall']:.4f} | "
            f"NDCG@10: {m_layer['ndcg']:.4f} | ILD@10: {m_layer['ild']:.4f} | Train Time: {res_layer[9]:.2f}s"
        )
    print("=" * 80 + "\n")