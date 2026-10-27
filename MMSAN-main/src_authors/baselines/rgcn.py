"""
R-GCN per dataset recommendation su ScholarlyDataset.

Task: dato un nodo 'publication' in test, ranking di tutti i nodi 'dataset'
per rilevanza (link 'cites' tra publication e dataset). Valutazione con
MRR, NDCG@5, NDCG@10, Recall@5, Recall@10.

Uso:
    python rgcn_dataset_recommendation.py --dataset <nome_dataset> [--epochs 100] [--hidden 128]

Assume che tu abbia già `loader.py` (il file che hai incollato) importabile
come modulo, con ScholarlyDataset, e che args_list.get_args() esponga almeno
--dataset. Se il tuo entrypoint è diverso, adatta la sezione "DATA LOADING".
"""

import argparse
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv, HeteroConv, GCNConv

# ----------------------------------------------------------------------------
# Riproducibilità
# ----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# MODELLO: R-GCN eterogeneo con score bilineare tra publication e dataset
# ----------------------------------------------------------------------------
class RGCNEncoder(nn.Module):
    """
    Incapsula la data HeteroData in un grafo omogeneo con relazioni tipizzate
    (approccio classico R-GCN: un solo RGCNConv per layer, che vede tutte le
    relazioni codificate come edge_type interi).
    """

    def __init__(self, node_types, num_nodes_dict, in_channels_dict, hidden_channels,
                 num_relations, num_bases=30, num_layers=2, dropout=0.2):
        super().__init__()
        self.node_types = node_types
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        # proiezione iniziale per portare tutte le feature (dimensioni diverse
        # per tipo di nodo) nello stesso spazio hidden_channels
        self.input_proj = nn.ModuleDict()
        for ntype in node_types:
            in_dim = in_channels_dict[ntype]
            self.input_proj[ntype] = nn.Linear(in_dim, hidden_channels)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, num_relations,
                         num_bases=num_bases)
            )

    def forward(self, x_dict, edge_index, edge_type, node_offsets):
        # concatena tutte le feature di nodo (già proiettate) in un unico tensore
        # nell'ordine definito da node_offsets
        xs = []
        for ntype in self.node_types:
            xs.append(self.input_proj[ntype](x_dict[ntype]))
        x = torch.cat(xs, dim=0)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class BilinearScorer(nn.Module):
    """Score(pub, dataset) = pub^T W dataset (DistMult-style, relazione unica)."""

    def __init__(self, hidden_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_channels, hidden_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, src_emb, dst_emb):
        # src_emb: [N, H], dst_emb: [N, H] -> score per riga (coppie allineate)
        return (src_emb @ self.weight * dst_emb).sum(dim=-1)

    def score_all(self, src_emb, dst_all_emb):
        # src_emb: [H], dst_all_emb: [M, H] -> [M] score contro tutti i dataset
        return dst_all_emb @ (self.weight.t() @ src_emb)


# ----------------------------------------------------------------------------
# Costruzione grafo omogeneo per R-GCN a partire da HeteroData
# ----------------------------------------------------------------------------
def build_homogeneous_graph(data: HeteroData, feature_key='x'):
    """
    Converte HeteroData in:
      - x_dict: {ntype: tensor features}
      - edge_index (2, E) globale, edge_type (E,)
      - offsets: {ntype: start_index} per mappare id locali -> id globali
      - node_types ordinati

    Usa solo gli edge_index "di message passing" (edge_index_train per
    pub-dataset, e i restanti edge_index normali per le altre relazioni),
    per evitare data leakage dai link di validation/test.
    """
    node_types = list(data.node_types)
    num_nodes_dict = {nt: data[nt].num_nodes for nt in node_types}

    offsets = {}
    cum = 0
    for nt in node_types:
        offsets[nt] = cum
        cum += num_nodes_dict[nt]
    total_nodes = cum

    # feature dict: usa 'x' se presente e non-None, altrimenti 'net_x' (node2vec)
    x_dict = {}
    in_channels_dict = {}
    for nt in node_types:
        feat = getattr(data[nt], feature_key, None)
        if feat is None:
            feat = data[nt].net_x
        feat = feat.float()
        x_dict[nt] = feat
        in_channels_dict[nt] = feat.size(-1)

    # raccogli le relazioni edge, usando SOLO edge_index di training per pub-dataset
    edge_types = list(data.edge_types)
    edge_index_list = []
    edge_type_list = []
    rel2id = {}

    def add_relation(src_type, rel_name, dst_type, edge_index, bidirectional=True):
        if edge_index is None or edge_index.numel() == 0:
            return
        key_fwd = (src_type, rel_name, dst_type)
        if key_fwd not in rel2id:
            rel2id[key_fwd] = len(rel2id)
        rid_fwd = rel2id[key_fwd]

        src_global = edge_index[0] + offsets[src_type]
        dst_global = edge_index[1] + offsets[dst_type]

        edge_index_list.append(torch.stack([src_global, dst_global], dim=0))
        edge_type_list.append(torch.full((edge_index.size(1),), rid_fwd, dtype=torch.long))

        if bidirectional:
            key_rev = (dst_type, rel_name + '_rev', src_type)
            if key_rev not in rel2id:
                rel2id[key_rev] = len(rel2id)
            rid_rev = rel2id[key_rev]
            edge_index_list.append(torch.stack([dst_global, src_global], dim=0))
            edge_type_list.append(torch.full((edge_index.size(1),), rid_rev, dtype=torch.long))

    for et in edge_types:
        src_type, rel_name, dst_type = et
        if src_type == 'publication' and dst_type == 'dataset' and rel_name == 'cites':
            # relazione target: usa SOLO edge_index_train per il message passing
            ei = data[et].get('edge_index_train', None)
            add_relation(src_type, rel_name, dst_type, ei)
            continue

        ei = data[et].get('edge_index', None)
        if ei is None:
            continue
        add_relation(src_type, rel_name, dst_type, ei)

    edge_index = torch.cat(edge_index_list, dim=1)
    edge_type = torch.cat(edge_type_list, dim=0)

    return {
        'x_dict': x_dict,
        'in_channels_dict': in_channels_dict,
        'node_types': node_types,
        'edge_index': edge_index,
        'edge_type': edge_type,
        'num_relations': len(rel2id),
        'offsets': offsets,
        'num_nodes_dict': num_nodes_dict,
        'total_nodes': total_nodes,
    }


# ----------------------------------------------------------------------------
# Sampling negativi per training (link prediction)
# ----------------------------------------------------------------------------
def sample_negative_datasets(pos_edge_index, num_pub, num_dataset, num_neg_per_pos=1):
    """Per ogni edge positivo (pub, dataset), campiona num_neg_per_pos dataset negativi
    (non filtra rigorosamente i falsi negativi per efficienza; su grafi grandi va bene,
    su grafi piccoli valuta un filtro esplicito)."""
    E = pos_edge_index.size(1)
    neg_dst = torch.randint(0, num_dataset, (E * num_neg_per_pos,))
    neg_src = pos_edge_index[0].repeat_interleave(num_neg_per_pos)
    return torch.stack([neg_src, neg_dst], dim=0)


# ----------------------------------------------------------------------------
# METRICHE DI RANKING: MRR, NDCG@k, Recall@k
# ----------------------------------------------------------------------------
def dcg_at_k(relevances, k):
    relevances = relevances[:k]
    if len(relevances) == 0:
        return 0.0
    gains = (2 ** relevances - 1)
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(ranked_relevances, num_relevant_total, k):
    dcg = dcg_at_k(np.array(ranked_relevances), k)
    ideal_relevances = np.ones(min(k, num_relevant_total))
    idcg = dcg_at_k(ideal_relevances, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_ranking(pub_emb, dataset_emb, scorer, eval_pairs, k_list=(5, 10), device='cpu'):
    max_k = max(k_list)

    mrr_at = {k: [] for k in k_list}
    ndcg_at = {k: [] for k in k_list}
    recall_at = {k: [] for k in k_list}

    dataset_emb = dataset_emb.to(device)

    with torch.no_grad():
        for pub_id, relevant_set in eval_pairs.items():
            if len(relevant_set) == 0:
                continue

            src_emb = pub_emb[pub_id].to(device)
            scores = scorer.score_all(src_emb, dataset_emb)
            ranked_idx = torch.argsort(scores, descending=True).cpu().numpy()

            # posizione del primo rilevante
            rank_of_first_relevant = None

            for pos, idx in enumerate(ranked_idx):
                if idx in relevant_set:
                    rank_of_first_relevant = pos + 1
                    break

            # MRR@K
            for k in k_list:
                if rank_of_first_relevant is not None and rank_of_first_relevant <= k:
                    mrr_at[k].append(1.0 / rank_of_first_relevant)
                else:
                    mrr_at[k].append(0.0)

                # Recall@K
                top_k_idx = ranked_idx[:k]
                hits = sum(1 for idx in top_k_idx if idx in relevant_set)
                recall_at[k].append(
                    hits / min(len(relevant_set), k)
                )

                # NDCG@K
                rel_at_k = [
                    1 if idx in relevant_set else 0
                    for idx in top_k_idx
                ]
                ndcg_at[k].append(
                    ndcg_at_k(rel_at_k, len(relevant_set), k)
                )

    results = {}

    for k in k_list:
        results[f'MRR@{k}'] = (
            float(np.mean(mrr_at[k]))
            if mrr_at[k] else 0.0
        )
        results[f'NDCG@{k}'] = (
            float(np.mean(ndcg_at[k]))
            if ndcg_at[k] else 0.0
        )
        results[f'Recall@{k}'] = (
            float(np.mean(recall_at[k]))
            if recall_at[k] else 0.0
        )

    return results


def build_eval_pairs(edge_label_index):
    """Da edge_index [2, E] (pub, dataset) costruisce dict pub -> set(dataset)."""
    pairs = defaultdict(set)
    src = edge_label_index[0].tolist()
    dst = edge_label_index[1].tolist()
    for s, d in zip(src, dst):
        pairs[s].add(d)
    return pairs


# ----------------------------------------------------------------------------
# TRAINING LOOP
# ----------------------------------------------------------------------------
def train_and_evaluate(data: HeteroData, args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f'Device: {device}')

    graph = build_homogeneous_graph(data, feature_key='x')
    node_types = graph['node_types']
    offsets = graph['offsets']

    model = RGCNEncoder(
        node_types=node_types,
        num_nodes_dict=graph['num_nodes_dict'],
        in_channels_dict=graph['in_channels_dict'],
        hidden_channels=args.hidden,
        num_relations=graph['num_relations'],
        num_bases=min(30, graph['num_relations']),
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    scorer = BilinearScorer(args.hidden).to(device)

    x_dict = {nt: graph['x_dict'][nt].to(device) for nt in node_types}
    edge_index = graph['edge_index'].to(device)
    edge_type = graph['edge_type'].to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(scorer.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    pub_off = offsets['publication']
    ds_off = offsets['dataset']
    num_pub = data['publication'].num_nodes
    num_dataset = data['dataset'].num_nodes

    train_edge_index = data['publication', 'cites', 'dataset'].edge_index_train  # [2, E_train]
    train_edge_index = train_edge_index.to(device)

    # split labels: dict {pub_local: set(dataset_local)} per validation/test
    eval_splits = {}
    for split in ['validation_trans', 'test_trans', 'test_semi', 'test_ind']:
        key = f'edge_label_index_{split}'
        if key in data['publication', 'cites', 'dataset']:
            eli = data['publication', 'cites', 'dataset'][key]
            eval_splits[split] = build_eval_pairs(eli)

    best_val_mrr = -1
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        scorer.train()
        optimizer.zero_grad()

        emb = model(x_dict, edge_index, edge_type, offsets)
        pub_emb_all = emb[pub_off: pub_off + num_pub]
        ds_emb_all = emb[ds_off: ds_off + num_dataset]

        pos_src = train_edge_index[0]
        pos_dst = train_edge_index[1]
        pos_scores = scorer(pub_emb_all[pos_src], ds_emb_all[pos_dst])

        neg_edge_index = sample_negative_datasets(
            train_edge_index.cpu(), num_pub, num_dataset, num_neg_per_pos=args.num_neg
        ).to(device)
        neg_scores = scorer(pub_emb_all[neg_edge_index[0]], ds_emb_all[neg_edge_index[1]])

        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        scores = torch.cat([pos_scores, neg_scores])
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        loss.backward()
        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            scorer.eval()
            with torch.no_grad():
                emb = model(x_dict, edge_index, edge_type, offsets)
                pub_emb_all = emb[pub_off: pub_off + num_pub]
                ds_emb_all = emb[ds_off: ds_off + num_dataset]

            log_line = f'Epoch {epoch:03d} | loss {loss.item():.4f}'
            if 'validation_trans' in eval_splits:
                val_results = evaluate_ranking(
                    pub_emb_all, ds_emb_all, scorer, eval_splits['validation_trans'],
                    k_list=(5, 10), device=device
                )
                log_line += f" | val MRR@10 {val_results['MRR@10']:.4f} NDCG@10 {val_results['NDCG@10']:.4f}"

                if val_results['NDCG@10'] > best_val_mrr:
                    best_val_mrr = val_results['NDCG@10']
                    best_state = {
                        'model': {k: v.clone() for k, v in model.state_dict().items()},
                        'scorer': {k: v.clone() for k, v in scorer.state_dict().items()},
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1
            print(log_line)

            if args.patience > 0 and patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch}')
                break

    # ricarica il miglior modello su validation (se disponibile) e valuta su tutti i test split
    if best_state is not None:
        model.load_state_dict(best_state['model'])
        scorer.load_state_dict(best_state['scorer'])

    model.eval()
    scorer.eval()
    with torch.no_grad():
        emb = model(x_dict, edge_index, edge_type, offsets)
        pub_emb_all = emb[pub_off: pub_off + num_pub]
        ds_emb_all = emb[ds_off: ds_off + num_dataset]

    print('\n=== Risultati finali ===')
    final_results = {}
    for split in ['test_trans', 'test_semi', 'test_ind']:
        if split in eval_splits:
            res = evaluate_ranking(
                pub_emb_all, ds_emb_all, scorer, eval_splits[split],
                k_list=(5, 10), device=device
            )
            final_results[split] = res
            print(f'[{split}]', {k: round(v, 4) for k, v in res.items()})

    return final_results, model, scorer


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='mes',
                    help='nome cartella dataset dentro model/data/<dataset>')
    p.add_argument('--root', type=str, default=None,
                    help='path root del dataset (default: model/data/<dataset>)')
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--eval_every', type=int, default=5)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--num_neg', type=int, default=1)
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()


import loader
def load_data(dataset):
    dataset = loader.ScholarlyDataset(root=f'../datasets/{dataset}/')
    data = dataset[0]
    return data

if __name__ == '__main__':
    args = parse_args()
    set_seed(SEED)

    root = args.root if args.root else f'../dataset/{args.dataset}'

    # ------------------------------------------------------------------
    # DATA LOADING: usa il tuo ScholarlyDataset esistente (dal loader che
    # hai incollato). Import relativo al tuo progetto: adatta il path se
    # il modulo si chiama diversamente (es. `from loader import ScholarlyDataset`).
    # ------------------------------------------------------------------
    data = load_data('mes')

    results, model, scorer = train_and_evaluate(data, args)

    print('\nFatto.')