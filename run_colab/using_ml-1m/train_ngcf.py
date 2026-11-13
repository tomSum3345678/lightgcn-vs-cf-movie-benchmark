# -*- coding: utf-8 -*-
"""
NGCF Model Training and Evaluation
1. Train / Val / Test Leave-One-Out Split (Ratings >= 4)
2. Supporting @10 and @20 Metrics
3. 50 Epochs & 3-Seed Mean ± Std, Precisely Measuring Pure Training Time and Single-User Inference Time
"""
import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from tqdm import tqdm
from sklearn import preprocessing as pp

# ============ Path Configuration ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'Datasets')
MOVIES_PATH = os.path.join(DATA_DIR, 'movies.csv')
RATINGS_PATH = os.path.join(DATA_DIR, 'ratings.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'ngcf_model.pth')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[NGCF] Using device: {device}")

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

le_user = pp.LabelEncoder()
le_item = pp.LabelEncoder()
train_df['user_id_idx'] = le_user.fit_transform(train_df['userId'].values)
train_df['item_id_idx'] = le_item.fit_transform(train_df['movieId'].values)

train_user_ids = set(train_df['userId'].unique())
train_item_ids = set(train_df['movieId'].unique())

val_df = val_df[val_df['userId'].isin(train_user_ids) & val_df['movieId'].isin(train_item_ids)].copy()
val_df['user_id_idx'] = le_user.transform(val_df['userId'].values)
val_df['item_id_idx'] = le_item.transform(val_df['movieId'].values)

test_df = test_df[test_df['userId'].isin(train_user_ids) & test_df['movieId'].isin(train_item_ids)].copy()
test_df['user_id_idx'] = le_user.transform(test_df['userId'].values)
test_df['item_id_idx'] = le_item.transform(test_df['movieId'].values)

n_users = train_df['user_id_idx'].nunique()
n_items = train_df['item_id_idx'].nunique()

MOVIE_GENRES = {m_id: set(str(row['genres']).split('|')) for m_id, row in movies_df.iterrows()}
IDX_TO_GENRES = {idx: MOVIE_GENRES.get(m_id, set()) for idx, m_id in enumerate(le_item.classes_)}
USER_POS_LIST = train_df.groupby('user_id_idx')['item_id_idx'].apply(list).to_dict()
USER_POS_SET = {u: set(items) for u, items in USER_POS_LIST.items()}

# ============ Hyperparameters ============
latent_dim = 64
n_layers = 3
EPOCHS = 50
BATCH_SIZE = 512
DECAY = 0.0001
LR = 0.001
K = 10

# ============ Fast Data Loader ============
def data_loader(data, batch_size, n_usr, n_itm):
    indices = list(range(n_usr))
    users = random.sample(indices, min(batch_size, n_usr))
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

u_t = torch.LongTensor(train_df.user_id_idx.values)
i_t = torch.LongTensor(train_df.item_id_idx.values) + n_users
train_edge_index = torch.stack((torch.cat([u_t, i_t]), torch.cat([i_t, u_t]))).to(device)

# ============ NGCF Layer ============
class NGCFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(aggr='add')
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, edge_index):
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
        agg = self.propagate(edge_index, x=x, norm=norm)
        out = self.leaky_relu(torch.matmul(agg, self.weight))
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

# ============ NGCF Model ============
class NGCF(nn.Module):
    def __init__(self, latent_dim, num_layers, num_users, num_items):
        super(NGCF, self).__init__()
        self.embedding = nn.Embedding(num_users + num_items, latent_dim)
        self.convs = nn.ModuleList([NGCFConv(latent_dim, latent_dim) for _ in range(num_layers)])
        nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, edge_index):
        emb0 = self.embedding.weight
        embs = [emb0]
        emb = emb0
        for conv in self.convs:
            emb = conv(x=emb, edge_index=edge_index)
            embs.append(emb)
        out = torch.cat(embs, dim=-1)
        return emb0, out

    def encode_minibatch(self, users, pos_items, neg_items, edge_index):
        emb0, out = self(edge_index)
        return (out[users], out[pos_items], out[neg_items], emb0[users], emb0[pos_items], emb0[neg_items])

# ============ Loss & Metrics ============
def compute_bpr_loss(users, users_emb, pos_emb, neg_emb, user_emb0, pos_emb0, neg_emb0):
    reg_loss = (1 / 2) * (user_emb0.norm().pow(2) + pos_emb0.norm().pow(2) + neg_emb0.norm().pow(2)) / float(len(users))
    pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
    neg_scores = torch.mul(users_emb, neg_emb).sum(dim=1)
    bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))
    return bpr_loss, reg_loss

def calculate_ild_from_indices(recommended_item_indices):
    if len(recommended_item_indices) < 2:
        return 0.0
    genre_sets = [IDX_TO_GENRES[idx] for idx in recommended_item_indices if idx in IDX_TO_GENRES]
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

def calculate_ndcg(actual_items_set, recommended_items, k):
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(recommended_items[:k]) if item in actual_items_set)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual_items_set), k)))
    return dcg / idcg if idcg > 0 else 0.0

def get_metrics(user_Embed_wts, item_Embed_wts, n_users, n_items, mask_df, eval_df, K=10):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_infer = time.time()

    relevance_score = torch.matmul(user_Embed_wts, torch.transpose(item_Embed_wts, 0, 1))
    i = torch.stack((
        torch.LongTensor(mask_df['user_id_idx'].values),
        torch.LongTensor(mask_df['item_id_idx'].values)
    )).to(device)
    interactions_t = torch.sparse_coo_tensor(
        i, torch.ones(len(mask_df), dtype=torch.bool, device=device),
        (n_users, n_items), device=device
    ).to_dense()
    relevance_score = relevance_score.masked_fill(interactions_t, float('-inf'))

    topk_indices = torch.topk(relevance_score, 20, dim=1).indices.cpu().numpy()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_per_user = (time.time() - start_infer) / n_users

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
    return {
        'precision': np.mean(p10_list),
        'recall': np.mean(r10_list),
        'ndcg': np.mean(ndcg10_list),
        'precision_20': np.mean(p20_list),
        'recall_20': np.mean(r20_list),
        'ndcg_20': np.mean(ndcg20_list),
        'ild': np.mean(ild10_list),
        'coverage': coverage,
        'infer_per_user': infer_per_user
    }

# ============ Execution (3-Seed Loop) ============
if __name__ == '__main__':
    seeds = [16, 42, 2026]
    all_metrics = []
    all_train_time = []
    train_val_mask_df = pd.concat([train_df, val_df], ignore_index=True)

    for seed in seeds:
        print(f"\n" + "=" * 50)
        print(f"Starting NGCF training with Seed: {seed}")
        print("=" * 50)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = NGCF(latent_dim, n_layers, n_users, n_items).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

        n_batch = max(1, int(len(train_df) / BATCH_SIZE))
        for epoch in tqdm(range(EPOCHS)):
            model.train()
            for _ in range(n_batch):
                optimizer.zero_grad()
                users, pos_items, neg_items = data_loader(train_df, BATCH_SIZE, n_users, n_items)
                users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0 = model.encode_minibatch(
                    users, pos_items, neg_items, train_edge_index
                )
                bpr_loss, reg_loss = compute_bpr_loss(users, users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0)
                final_loss = bpr_loss + (DECAY * reg_loss)
                final_loss.backward()
                optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        training_time = time.time() - start_time

        model.eval()
        with torch.no_grad():
            _, out = model(train_edge_index)
            final_user_Embed, final_item_Embed = torch.split(out, (n_users, n_items))
            m = get_metrics(final_user_Embed, final_item_Embed, n_users, n_items, train_val_mask_df, test_df, K)

        all_metrics.append(m)
        all_train_time.append(training_time)
        print(
            f"Seed {seed} Done! P@10: {m['precision']:.4f}, NDCG@10: {m['ndcg']:.4f}, "
            f"NDCG@20: {m['ndcg_20']:.4f}, ILD@10: {m['ild']:.4f}, Train Time: {training_time:.2f}s"
        )

        torch.save({
            'model_state_dict': model.state_dict(),
            'model_info': {
                'model_architecture': 'NGCF',
                'latent_dim': latent_dim,
                'num_layers': n_layers,
                'num_users': n_users,
                'num_items': n_items,
                'training_time_seconds': training_time
                }
        }, MODEL_PATH)
        print(f"[NGCF] Model saved to {MODEL_PATH}")

    print("\n\n" + "=" * 65)
    print("NGCF FINAL RESULTS (Mean ± Standard Deviation across 3 Seeds)")
    print("=" * 65)
    print(f"Precision@10:   {np.mean([m['precision'] for m in all_metrics]):.4f} ± {np.std([m['precision'] for m in all_metrics]):.4f}")
    print(f"Recall@10:      {np.mean([m['recall'] for m in all_metrics]):.4f} ± {np.std([m['recall'] for m in all_metrics]):.4f}")
    print(f"NDCG@10:        {np.mean([m['ndcg'] for m in all_metrics]):.4f} ± {np.std([m['ndcg'] for m in all_metrics]):.4f}")
    print(f"Precision@20:   {np.mean([m['precision_20'] for m in all_metrics]):.4f} ± {np.std([m['precision_20'] for m in all_metrics]):.4f}")
    print(f"Recall@20:      {np.mean([m['recall_20'] for m in all_metrics]):.4f} ± {np.std([m['recall_20'] for m in all_metrics]):.4f}")
    print(f"NDCG@20:        {np.mean([m['ndcg_20'] for m in all_metrics]):.4f} ± {np.std([m['ndcg_20'] for m in all_metrics]):.4f}")
    print(f"ILD@10:         {np.mean([m['ild'] for m in all_metrics]):.4f} ± {np.std([m['ild'] for m in all_metrics]):.4f}")
    print(f"Coverage@10:    {np.mean([m['coverage'] for m in all_metrics])*100:.2f}% ± {np.std([m['coverage'] for m in all_metrics])*100:.2f}%")
    print(f"Training Time:  {np.mean(all_train_time):.2f}s ± {np.std(all_train_time):.2f}s")
    print(f"Infer/User:     {np.mean([m['infer_per_user'] for m in all_metrics])*1000:.4f} ms ± {np.std([m['infer_per_user'] for m in all_metrics])*1000:.4f} ms")
    print("=" * 65 + "\n")