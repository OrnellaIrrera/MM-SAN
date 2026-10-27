# import argparse
# import copy
# import random
# from collections import defaultdict
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from torch_geometric.data import HeteroData
# from torch_geometric.nn import RGCNConv
# from torch_geometric.utils import softmax as scatter_softmax
#
# import loader
#
#
# # ============================================================================
# # RIPRODUCIBILITÀ
# # ============================================================================
#
# SEED = 42
#
#
# def set_seed(seed=SEED):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
#     # Per risultati più riproducibili.
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
#
# # ============================================================================
# # GRAFO OMOGENEO
# # ============================================================================
#
# def build_homogeneous_graph(data: HeteroData, feature_key="x"):
#     """
#     Converte HeteroData in un grafo omogeneo con:
#         x_dict
#         edge_index globale
#         edge_type
#         offsets
#         numero di relazioni
#
#     Per publication -> dataset usa SOLO edge_index_train.
#     """
#
#     node_types = list(data.node_types)
#
#     num_nodes_dict = {
#         nt: data[nt].num_nodes
#         for nt in node_types
#     }
#
#     offsets = {}
#     total_nodes = 0
#
#     for nt in node_types:
#         offsets[nt] = total_nodes
#         total_nodes += num_nodes_dict[nt]
#
#     x_dict = {}
#     in_channels_dict = {}
#
#     for nt in node_types:
#         feat = getattr(data[nt], feature_key, None)
#
#         if feat is None:
#             feat = data[nt].net_x
#
#         feat = feat.float()
#
#         x_dict[nt] = feat
#         in_channels_dict[nt] = feat.size(-1)
#
#     edge_index_list = []
#     edge_type_list = []
#
#     rel2id = {}
#
#     def add_relation(
#         src_type,
#         rel_name,
#         dst_type,
#         edge_index,
#         bidirectional=True,
#     ):
#         if edge_index is None:
#             return
#
#         if edge_index.numel() == 0:
#             return
#
#         key_fwd = (src_type, rel_name, dst_type)
#
#         if key_fwd not in rel2id:
#             rel2id[key_fwd] = len(rel2id)
#
#         rid_fwd = rel2id[key_fwd]
#
#         src_global = edge_index[0] + offsets[src_type]
#         dst_global = edge_index[1] + offsets[dst_type]
#
#         edge_index_list.append(
#             torch.stack(
#                 [src_global, dst_global],
#                 dim=0,
#             )
#         )
#
#         edge_type_list.append(
#             torch.full(
#                 (edge_index.size(1),),
#                 rid_fwd,
#                 dtype=torch.long,
#             )
#         )
#
#         if bidirectional:
#             key_rev = (
#                 dst_type,
#                 rel_name + "_rev",
#                 src_type,
#             )
#
#             if key_rev not in rel2id:
#                 rel2id[key_rev] = len(rel2id)
#
#             rid_rev = rel2id[key_rev]
#
#             edge_index_list.append(
#                 torch.stack(
#                     [dst_global, src_global],
#                     dim=0,
#                 )
#             )
#
#             edge_type_list.append(
#                 torch.full(
#                     (edge_index.size(1),),
#                     rid_rev,
#                     dtype=torch.long,
#                 )
#             )
#
#     for et in data.edge_types:
#
#         src_type, rel_name, dst_type = et
#
#         # IMPORTANTISSIMO:
#         # per publication -> dataset utilizziamo esclusivamente
#         # gli edge di training.
#         if (
#             src_type == "publication"
#             and dst_type == "dataset"
#             and rel_name == "cites"
#         ):
#             edge_index = data[et].get(
#                 "edge_index_train",
#                 None,
#             )
#
#             add_relation(
#                 src_type,
#                 rel_name,
#                 dst_type,
#                 edge_index,
#             )
#
#             continue
#
#         edge_index = data[et].get(
#             "edge_index",
#             None,
#         )
#
#         add_relation(
#             src_type,
#             rel_name,
#             dst_type,
#             edge_index,
#         )
#
#     if not edge_index_list:
#         raise RuntimeError(
#             "Il grafo non contiene edge utilizzabili."
#         )
#
#     edge_index = torch.cat(
#         edge_index_list,
#         dim=1,
#     )
#
#     edge_type = torch.cat(
#         edge_type_list,
#         dim=0,
#     )
#
#     print(
#         f"[graph] nodes={total_nodes} "
#         f"edges={edge_index.size(1)} "
#         f"relations={len(rel2id)}"
#     )
#
#     return {
#         "x_dict": x_dict,
#         "in_channels_dict": in_channels_dict,
#         "node_types": node_types,
#         "edge_index": edge_index,
#         "edge_type": edge_type,
#         "num_relations": len(rel2id),
#         "offsets": offsets,
#         "num_nodes_dict": num_nodes_dict,
#         "total_nodes": total_nodes,
#     }
#
#
# # ============================================================================
# # SCORER
# # ============================================================================
# class InnerProductScorer(nn.Module):
#     """
#     Decoder inner-product puro:
#
#         score(i,j) = z_i^T z_j
#     """
#
#     def forward(self, src_emb, dst_emb):
#         return (src_emb * dst_emb).sum(dim=-1)
#
#     def score_all(self, src_emb, dst_all_emb):
#         return dst_all_emb @ src_emb
#
# class BilinearScorer(nn.Module):
#     """
#     Score(pub, dataset) = pub^T W dataset.
#
#     A differenza della versione precedente NON normalizziamo gli embedding.
#     """
#
#     def __init__(self, dim):
#         super().__init__()
#
#         self.weight = nn.Parameter(
#             torch.empty(dim, dim)
#         )
#
#         nn.init.xavier_uniform_(
#             self.weight
#         )
#
#     def forward(self, src_emb, dst_emb):
#
#         return (
#             src_emb @ self.weight
#             * dst_emb
#         ).sum(dim=-1)
#
#     def score_all(self, src_emb, dst_all_emb):
#
#         transformed_src = self.weight.t() @ src_emb
#
#         return dst_all_emb @ transformed_src
#
#
# # ============================================================================
# # RGCN
# # ============================================================================
#
# class RGCNEncoder(nn.Module):
#     """
#     RGCN migliorato:
#
#       projection per tipo di nodo
#       -> RGCN
#       -> ReLU
#       -> Dropout
#       -> residual
#       -> LayerNorm
#
#     La residual connection riduce il rischio che aumentando la profondità
#     il segnale originario venga perso.
#     """
#
#     def __init__(
#         self,
#         node_types,
#         in_channels_dict,
#         hidden_channels,
#         num_relations,
#         num_bases=16,
#         num_layers=2,
#         dropout=0.1,
#     ):
#         super().__init__()
#
#         self.node_types = node_types
#         self.hidden_channels = hidden_channels
#         self.dropout = dropout
#
#         self.input_proj = nn.ModuleDict()
#
#         for nt in node_types:
#             self.input_proj[nt] = nn.Linear(
#                 in_channels_dict[nt],
#                 hidden_channels,
#             )
#
#         self.convs = nn.ModuleList()
#         self.norms = nn.ModuleList()
#
#         for _ in range(num_layers):
#
#             self.convs.append(
#                 RGCNConv(
#                     hidden_channels,
#                     hidden_channels,
#                     num_relations,
#                     num_bases=num_bases,
#                 )
#             )
#
#             self.norms.append(
#                 nn.LayerNorm(hidden_channels)
#             )
#
#     def forward(
#         self,
#         x_dict,
#         edge_index,
#         edge_type,
#     ):
#
#         xs = []
#
#         for nt in self.node_types:
#
#             x_nt = self.input_proj[nt](
#                 x_dict[nt]
#             )
#
#             xs.append(x_nt)
#
#         x = torch.cat(
#             xs,
#             dim=0,
#         )
#
#         for conv, norm in zip(
#             self.convs,
#             self.norms,
#         ):
#
#             residual = x
#
#             h = conv(
#                 x,
#                 edge_index,
#                 edge_type,
#             )
#
#             h = F.relu(h)
#
#             h = F.dropout(
#                 h,
#                 p=self.dropout,
#                 training=self.training,
#             )
#
#             x = norm(
#                 h + residual
#             )
#
#         return x
#
#
# # ============================================================================
# # KGAT
# # ============================================================================
#
# class KGATEncoder(nn.Module):
#     """
#     KGAT adattato al grafo Scholarly.
#
#     Migliorie rispetto alla versione precedente:
#
#       1. attention esplicita su:
#             [head, relation, tail]
#
#       2. trasformazione relation-specifica diagonale
#
#       3. bi-interaction aggregator
#
#       4. residual
#
#       5. LayerNorm
#
#       6. NIENTE L2-normalization ad ogni layer
#
#     L'embedding finale concatena layer 0 ... layer L.
#     """
#
#     def __init__(
#         self,
#         node_types,
#         in_channels_dict,
#         hidden_channels,
#         num_relations,
#         num_layers=3,
#         dropout=0.1,
#     ):
#         super().__init__()
#
#         self.node_types = node_types
#         self.hidden_channels = hidden_channels
#         self.num_layers = num_layers
#         self.dropout = dropout
#
#         self.out_dim = (
#             hidden_channels
#             * (num_layers + 1)
#         )
#
#         self.input_proj = nn.ModuleDict()
#
#         for nt in node_types:
#
#             self.input_proj[nt] = nn.Linear(
#                 in_channels_dict[nt],
#                 hidden_channels,
#             )
#
#         # embedding relazione
#         self.rel_emb = nn.Embedding(
#             num_relations,
#             hidden_channels,
#         )
#
#         # trasformazione diagonale
#         self.rel_transform = nn.Embedding(
#             num_relations,
#             hidden_channels,
#         )
#
#         nn.init.ones_(
#             self.rel_transform.weight
#         )
#
#         self.attn = nn.ModuleList()
#         self.w1 = nn.ModuleList()
#         self.w2 = nn.ModuleList()
#         self.norms = nn.ModuleList()
#
#         for _ in range(num_layers):
#
#             # head + relation + tail
#             self.attn.append(
#                 nn.Sequential(
#                     nn.Linear(
#                         hidden_channels * 3,
#                         hidden_channels,
#                     ),
#                     nn.LeakyReLU(0.2),
#                     nn.Linear(
#                         hidden_channels,
#                         1,
#                     ),
#                 )
#             )
#
#             self.w1.append(
#                 nn.Linear(
#                     hidden_channels,
#                     hidden_channels,
#                 )
#             )
#
#             self.w2.append(
#                 nn.Linear(
#                     hidden_channels,
#                     hidden_channels,
#                 )
#             )
#
#             self.norms.append(
#                 nn.LayerNorm(
#                     hidden_channels
#                 )
#             )
#
#     def forward(
#         self,
#         x_dict,
#         edge_index,
#         edge_type,
#         total_nodes,
#     ):
#
#         xs = []
#
#         for nt in self.node_types:
#
#             xs.append(
#                 self.input_proj[nt](
#                     x_dict[nt]
#                 )
#             )
#
#         x = torch.cat(
#             xs,
#             dim=0,
#         )
#
#         src, dst = edge_index
#
#         all_layers = [x]
#
#         for layer in range(
#             self.num_layers
#         ):
#
#             rel_scale = (
#                 self.rel_transform(
#                     edge_type
#                 )
#             )
#
#             rel_vec = (
#                 self.rel_emb(
#                     edge_type
#                 )
#             )
#
#             # head = nodo che aggrega
#             # tail = vicino
#             head = (
#                 x[src]
#                 * rel_scale
#             )
#
#             tail = (
#                 x[dst]
#                 * rel_scale
#             )
#
#             attn_input = torch.cat(
#                 [
#                     head,
#                     rel_vec,
#                     tail,
#                 ],
#                 dim=-1,
#             )
#
#             score = (
#                 self.attn[layer](
#                     attn_input
#                 )
#                 .squeeze(-1)
#             )
#
#             alpha = scatter_softmax(
#                 score,
#                 src,
#                 num_nodes=total_nodes,
#             )
#
#             alpha = F.dropout(
#                 alpha,
#                 p=self.dropout,
#                 training=self.training,
#             )
#
#             messages = (
#                 alpha.unsqueeze(-1)
#                 * (
#                     tail + rel_vec
#                 )
#             )
#
#             neigh = x.new_zeros(
#                 total_nodes,
#                 self.hidden_channels,
#             )
#
#             neigh.index_add_(
#                 0,
#                 src,
#                 messages,
#             )
#
#             # bi-interaction
#             first = self.w1[layer](
#                 x + neigh
#             )
#
#             second = self.w2[layer](
#                 x * neigh
#             )
#
#             h = (
#                 F.leaky_relu(
#                     first,
#                     0.2,
#                 )
#                 +
#                 F.leaky_relu(
#                     second,
#                     0.2,
#                 )
#             )
#
#             # residual + norm
#             x = self.norms[layer](
#                 h + x
#             )
#
#             x = F.dropout(
#                 x,
#                 p=self.dropout,
#                 training=self.training,
#             )
#
#             all_layers.append(x)
#
#         return torch.cat(
#             all_layers,
#             dim=-1,
#         )
#
#
# # ============================================================================
# # KGCN
# # ============================================================================
#
# class KGCNEncoder(nn.Module):
#     """
#     KGCN query-dependent.
#
#     La differenza fondamentale rispetto al codice precedente è che
#     l'attention NON dipende solamente da:
#
#         relation @ query
#
#     ma da:
#
#         query + relation + neighbor
#
#     quindi lo stesso tipo di relazione può avere peso diverso per vicini
#     diversi e per query diverse.
#     """
#
#     def __init__(
#         self,
#         node_types,
#         in_channels_dict,
#         hidden_channels,
#         num_relations,
#         num_layers=2,
#         dropout=0.1,
#     ):
#         super().__init__()
#
#         self.node_types = node_types
#         self.hidden_channels = hidden_channels
#         self.num_relations = num_relations
#         self.num_layers = num_layers
#         self.dropout = dropout
#
#         self.input_proj = nn.ModuleDict()
#
#         for nt in node_types:
#
#             self.input_proj[nt] = nn.Linear(
#                 in_channels_dict[nt],
#                 hidden_channels,
#             )
#
#         self.rel_emb = nn.Embedding(
#             num_relations,
#             hidden_channels,
#         )
#
#         self.rel_transform = nn.Embedding(
#             num_relations,
#             hidden_channels,
#         )
#
#         nn.init.ones_(
#             self.rel_transform.weight
#         )
#
#         self.query_proj = nn.Linear(
#             hidden_channels,
#             hidden_channels,
#         )
#
#         self.self_transform = nn.ModuleList()
#         self.neigh_transform = nn.ModuleList()
#         self.attention = nn.ModuleList()
#         self.norms = nn.ModuleList()
#
#         for _ in range(num_layers):
#
#             self.self_transform.append(
#                 nn.Linear(
#                     hidden_channels,
#                     hidden_channels,
#                 )
#             )
#
#             self.neigh_transform.append(
#                 nn.Linear(
#                     hidden_channels,
#                     hidden_channels,
#                 )
#             )
#
#             # query + relation + neighbor
#             self.attention.append(
#                 nn.Sequential(
#                     nn.Linear(
#                         hidden_channels * 3,
#                         hidden_channels,
#                     ),
#                     nn.ReLU(),
#                     nn.Linear(
#                         hidden_channels,
#                         1,
#                     ),
#                 )
#             )
#
#             self.norms.append(
#                 nn.LayerNorm(
#                     hidden_channels
#                 )
#             )
#
#     def base_embeddings(
#         self,
#         x_dict,
#     ):
#
#         xs = []
#
#         for nt in self.node_types:
#
#             xs.append(
#                 self.input_proj[nt](
#                     x_dict[nt]
#                 )
#             )
#
#         return torch.cat(
#             xs,
#             dim=0,
#         )
#
#     def conditioned_embeddings(
#         self,
#         x0,
#         edge_index,
#         edge_type,
#         total_nodes,
#         query_emb,
#     ):
#
#         src, dst = edge_index
#
#         query = self.query_proj(
#             query_emb
#         )
#
#         x = x0
#
#         for layer in range(
#             self.num_layers
#         ):
#
#             rel_vec = self.rel_emb(
#                 edge_type
#             )
#
#             rel_scale = self.rel_transform(
#                 edge_type
#             )
#
#             # il vicino trasformato dalla relazione
#             neighbor = (
#                 x[dst]
#                 * rel_scale
#             )
#
#             query_edge = query.expand(
#                 neighbor.size(0),
#                 -1,
#             )
#
#             attn_input = torch.cat(
#                 [
#                     query_edge,
#                     rel_vec,
#                     neighbor,
#                 ],
#                 dim=-1,
#             )
#
#             attention_score = (
#                 self.attention[layer](
#                     attn_input
#                 )
#                 .squeeze(-1)
#             )
#
#             # attention tra i vicini dello stesso nodo
#             alpha = scatter_softmax(
#                 attention_score,
#                 src,
#                 num_nodes=total_nodes,
#             )
#
#             alpha = F.dropout(
#                 alpha,
#                 p=self.dropout,
#                 training=self.training,
#             )
#
#             message = (
#                 alpha.unsqueeze(-1)
#                 *
#                 (
#                     neighbor
#                     +
#                     rel_vec
#                 )
#             )
#
#             neigh = x.new_zeros(
#                 total_nodes,
#                 self.hidden_channels,
#             )
#
#             neigh.index_add_(
#                 0,
#                 src,
#                 message,
#             )
#
#             self_part = self.self_transform[layer](
#                 x
#             )
#
#             neigh_part = self.neigh_transform[layer](
#                 neigh
#             )
#
#             h = F.relu(
#                 self_part
#                 +
#                 neigh_part
#             )
#
#             h = F.dropout(
#                 h,
#                 p=self.dropout,
#                 training=self.training,
#             )
#
#             x = self.norms[layer](
#                 x + h
#             )
#
#         return x
#
#
# # ============================================================================
# # BPR
# # ============================================================================
#
# def bpr_loss(
#     positive_scores,
#     negative_scores,
# ):
#     """
#     Massimizza:
#
#         log sigma(pos - neg)
#
#     coerente direttamente con un obiettivo di ranking.
#     """
#
#     return -F.logsigmoid(
#         positive_scores
#         - negative_scores
#     ).mean()
#
#
# # ============================================================================
# # NEGATIVE SAMPLING
# # ============================================================================
#
# def build_positive_sets(
#     edge_index,
# ):
#     """
#     pub -> set(dataset positivi)
#     """
#
#     result = defaultdict(set)
#
#     src = edge_index[0].tolist()
#     dst = edge_index[1].tolist()
#
#     for s, d in zip(src, dst):
#         result[s].add(d)
#
#     return result
#
#
# def build_negative_pool(
#     edge_index,
# ):
#     """
#     Dataset presenti nel training.
#     Usare questo pool produce negativi più informativi rispetto
#     a scegliere sempre da tutti i dataset.
#     """
#
#     return torch.unique(
#         edge_index[1]
#     )
#
#
# def sample_filtered_negatives(
#     pos_edge_index,
#     num_dataset,
#     num_neg_per_pos,
#     positive_sets,
#     dataset_pool=None,
# ):
#     """
#     Negative sampling filtrato.
#
#     Ogni negativo:
#       - non deve essere positivo per quella publication
#       - preferibilmente viene scelto dal pool dei dataset osservati
#     """
#
#     if dataset_pool is None:
#         dataset_pool = torch.arange(
#             num_dataset,
#             dtype=torch.long,
#         )
#
#     negatives = []
#
#     src_list = pos_edge_index[0].tolist()
#     dst_list = pos_edge_index[1].tolist()
#
#     for pub_id, _ in zip(
#         src_list,
#         dst_list,
#     ):
#
#         positives = positive_sets[pub_id]
#
#         # tentativi multipli per trovare negativi veri
#         sampled = []
#
#         max_attempts = (
#             num_neg_per_pos * 10
#             + 20
#         )
#
#         attempts = 0
#
#         while (
#             len(sampled)
#             < num_neg_per_pos
#             and attempts < max_attempts
#         ):
#
#             idx = random.randrange(
#                 dataset_pool.numel()
#             )
#
#             candidate = int(
#                 dataset_pool[idx]
#             )
#
#             attempts += 1
#
#             if candidate in positives:
#                 continue
#
#             sampled.append(
#                 candidate
#             )
#
#         # fallback completo
#         if len(sampled) < num_neg_per_pos:
#
#             candidates = [
#                 d
#                 for d in range(num_dataset)
#                 if d not in positives
#             ]
#
#             if not candidates:
#                 candidates = list(
#                     range(num_dataset)
#                 )
#
#             while len(sampled) < num_neg_per_pos:
#
#                 sampled.append(
#                     random.choice(
#                         candidates
#                     )
#                 )
#
#         negatives.extend(
#             sampled
#         )
#
#     neg_src = []
#
#     for pub_id in src_list:
#         neg_src.extend(
#             [pub_id]
#             * num_neg_per_pos
#         )
#
#     return torch.tensor(
#         [neg_src, negatives],
#         dtype=torch.long,
#     )
#
#
# # ============================================================================
# # EVALUATION
# # ============================================================================
#
# def dcg_at_k(
#     relevances,
#     k,
# ):
#     relevances = relevances[:k]
#
#     if len(relevances) == 0:
#         return 0.0
#
#     gains = (
#         2 ** relevances
#         - 1
#     )
#
#     discounts = np.log2(
#         np.arange(
#             2,
#             len(relevances) + 2,
#         )
#     )
#
#     return float(
#         np.sum(
#             gains / discounts
#         )
#     )
#
#
# def ndcg_at_k(
#     ranked_relevances,
#     num_relevant_total,
#     k,
# ):
#
#     dcg = dcg_at_k(
#         np.array(ranked_relevances),
#         k,
#     )
#
#     ideal_relevances = np.ones(
#         min(k, num_relevant_total)
#     )
#
#     idcg = dcg_at_k(
#         ideal_relevances,
#         k,
#     )
#
#     if idcg == 0:
#         return 0.0
#
#     return dcg / idcg
#
#
# def build_eval_pairs(
#     edge_label_index,
# ):
#
#     pairs = defaultdict(set)
#
#     src = edge_label_index[0].tolist()
#     dst = edge_label_index[1].tolist()
#
#     for s, d in zip(src, dst):
#
#         pairs[s].add(d)
#
#     return pairs
#
#
# def evaluate_ranking(
#     score_fn,
#     eval_pairs,
#     k_list=(5, 10),
#     exclude_pairs=None,
# ):
#
#     mrr_at = {
#         k: []
#         for k in k_list
#     }
#
#     ndcg_at = {
#         k: []
#         for k in k_list
#     }
#
#     recall_at = {
#         k: []
#         for k in k_list
#     }
#
#     with torch.no_grad():
#
#         for pub_id, relevant_set in eval_pairs.items():
#
#             if len(relevant_set) == 0:
#                 continue
#
#             scores = score_fn(
#                 pub_id
#             ).clone()
#
#             # NON rankare i dataset già osservati in training
#             # per quella publication.
#             if exclude_pairs is not None:
#
#                 known = exclude_pairs.get(
#                     pub_id,
#                     set(),
#                 )
#
#                 if known:
#
#                     known_idx = torch.tensor(
#                         list(known),
#                         dtype=torch.long,
#                         device=scores.device,
#                     )
#
#                     scores[known_idx] = -torch.inf
#
#             ranked_idx = torch.argsort(
#                 scores,
#                 descending=True,
#             ).cpu().numpy()
#
#             rank_first = None
#
#             for pos, idx in enumerate(
#                 ranked_idx
#             ):
#
#                 if idx in relevant_set:
#
#                     rank_first = pos + 1
#                     break
#
#             for k in k_list:
#
#                 if (
#                     rank_first is not None
#                     and rank_first <= k
#                 ):
#                     mrr_at[k].append(
#                         1.0 / rank_first
#                     )
#                 else:
#                     mrr_at[k].append(
#                         0.0
#                     )
#
#                 top_k = ranked_idx[:k]
#
#                 hits = sum(
#                     1
#                     for idx in top_k
#                     if idx in relevant_set
#                 )
#
#                 recall_at[k].append(
#                     hits
#                     /
#                     min(
#                         len(relevant_set),
#                         k,
#                     )
#                 )
#
#                 rel_at_k = [
#                     1
#                     if idx in relevant_set
#                     else 0
#                     for idx in top_k
#                 ]
#
#                 ndcg_at[k].append(
#                     ndcg_at_k(
#                         rel_at_k,
#                         len(relevant_set),
#                         k,
#                     )
#                 )
#
#     results = {}
#
#     for k in k_list:
#
#         results[
#             f"MRR@{k}"
#         ] = (
#             float(
#                 np.mean(
#                     mrr_at[k]
#                 )
#             )
#             if mrr_at[k]
#             else 0.0
#         )
#
#         results[
#             f"NDCG@{k}"
#         ] = (
#             float(
#                 np.mean(
#                     ndcg_at[k]
#                 )
#             )
#             if ndcg_at[k]
#             else 0.0
#         )
#
#         results[
#             f"Recall@{k}"
#         ] = (
#             float(
#                 np.mean(
#                     recall_at[k]
#                 )
#             )
#             if recall_at[k]
#             else 0.0
#         )
#
#     return results
#
#
# # ============================================================================
# # UTILITY
# # ============================================================================
#
# def move_graph_to_device(
#     graph,
#     device,
# ):
#
#     x_dict = {
#         nt: x.to(device)
#         for nt, x in graph["x_dict"].items()
#     }
#
#     edge_index = graph[
#         "edge_index"
#     ].to(device)
#
#     edge_type = graph[
#         "edge_type"
#     ].to(device)
#
#     return (
#         x_dict,
#         edge_index,
#         edge_type,
#     )
#
#
# def get_eval_splits(
#     data,
# ):
#
#     relation = data[
#         "publication",
#         "cites",
#         "dataset",
#     ]
#
#     result = {}
#
#     for split in [
#         "validation_trans",
#         "test_trans",
#         "test_semi",
#         "test_ind",
#     ]:
#
#         key = (
#             f"edge_label_index_{split}"
#         )
#
#         if key in relation:
#
#             result[split] = build_eval_pairs(
#                 relation[key]
#             )
#
#     return result
#
#
# # ============================================================================
# # TRAIN STATIC MODELS
# # RGCN / KGAT
# # ============================================================================
#
# def train_static_model(
#     data,
#     args,
#     graph,
#     device,
# ):
#
#     node_types = graph["node_types"]
#     offsets = graph["offsets"]
#
#     total_nodes = graph[
#         "total_nodes"
#     ]
#
#     if args.model == "rgcn":
#
#         model = RGCNEncoder(
#             node_types=node_types,
#             in_channels_dict=graph[
#                 "in_channels_dict"
#             ],
#             hidden_channels=args.hidden,
#             num_relations=graph[
#                 "num_relations"
#             ],
#             num_bases=args.num_bases,
#             num_layers=args.num_layers,
#             dropout=args.dropout,
#         ).to(device)
#
#         out_dim = args.hidden
#
#     elif args.model == "kgat":
#
#         model = KGATEncoder(
#             node_types=node_types,
#             in_channels_dict=graph[
#                 "in_channels_dict"
#             ],
#             hidden_channels=args.hidden,
#             num_relations=graph[
#                 "num_relations"
#             ],
#             num_layers=args.num_layers,
#             dropout=args.dropout,
#         ).to(device)
#
#         out_dim = model.out_dim
#
#     else:
#
#         raise ValueError(
#             args.model
#         )
#
#     scorer = BilinearScorer(
#         out_dim
#     ).to(device)
#
#     x_dict, edge_index, edge_type = (
#         move_graph_to_device(
#             graph,
#             device,
#         )
#     )
#
#     optimizer = torch.optim.AdamW(
#         list(model.parameters())
#         +
#         list(scorer.parameters()),
#         lr=args.lr,
#         weight_decay=args.weight_decay,
#     )
#
#     pub_off = offsets[
#         "publication"
#     ]
#
#     ds_off = offsets[
#         "dataset"
#     ]
#
#     num_pub = data[
#         "publication"
#     ].num_nodes
#
#     num_dataset = data[
#         "dataset"
#     ].num_nodes
#
#     train_edge_index_cpu = data[
#         "publication",
#         "cites",
#         "dataset",
#     ].edge_index_train.cpu()
#
#     train_edge_index = (
#         train_edge_index_cpu.to(device)
#     )
#
#     positive_sets = build_positive_sets(
#         train_edge_index_cpu
#     )
#
#     negative_pool = build_negative_pool(
#         train_edge_index_cpu
#     )
#
#     eval_splits = get_eval_splits(
#         data
#     )
#
#     best_val_ndcg = -1.0
#     best_state = None
#     patience_counter = 0
#
#     params = (
#         list(model.parameters())
#         +
#         list(scorer.parameters())
#     )
#
#     for epoch in range(
#         1,
#         args.epochs + 1,
#     ):
#
#         model.train()
#         scorer.train()
#
#         optimizer.zero_grad()
#
#         if args.model == "kgat":
#             embeddings = model(
#                 x_dict,
#                 edge_index,
#                 edge_type,
#                 total_nodes,
#             )
#         else:
#             embeddings = model(
#                 x_dict,
#                 edge_index,
#                 edge_type,
#             )
#
#         pub_emb = embeddings[
#             pub_off:
#             pub_off + num_pub
#         ]
#
#         ds_emb = embeddings[
#             ds_off:
#             ds_off + num_dataset
#         ]
#
#         pos_src = train_edge_index[
#             0
#         ]
#
#         pos_dst = train_edge_index[
#             1
#         ]
#
#         pos_scores = scorer(
#             pub_emb[pos_src],
#             ds_emb[pos_dst],
#         )
#
#         neg_edge_index = (
#             sample_filtered_negatives(
#                 train_edge_index_cpu,
#                 num_dataset,
#                 args.num_neg,
#                 positive_sets,
#                 negative_pool,
#             ).to(device)
#         )
#
#         neg_scores = scorer(
#             pub_emb[
#                 neg_edge_index[0]
#             ],
#             ds_emb[
#                 neg_edge_index[1]
#             ],
#         )
#
#         # Ripetiamo i positivi per allinearli ai negativi
#         pos_scores_expanded = (
#             pos_scores.repeat_interleave(
#                 args.num_neg
#             )
#         )
#
#         loss = bpr_loss(
#             pos_scores_expanded,
#             neg_scores,
#         )
#
#         loss.backward()
#
#         torch.nn.utils.clip_grad_norm_(
#             params,
#             max_norm=5.0,
#         )
#
#         optimizer.step()
#
#         if (
#             epoch % args.eval_every == 0
#             or epoch == args.epochs
#         ):
#
#             model.eval()
#             scorer.eval()
#
#             with torch.no_grad():
#
#                 if args.model == "kgat":
#                     embeddings = model(
#                         x_dict,
#                         edge_index,
#                         edge_type,
#                         total_nodes,
#                     )
#                 else:
#                     embeddings = model(
#                         x_dict,
#                         edge_index,
#                         edge_type,
#                     )
#
#                 pub_emb = embeddings[
#                     pub_off:
#                     pub_off + num_pub
#                 ]
#
#                 ds_emb = embeddings[
#                     ds_off:
#                     ds_off + num_dataset
#                 ]
#
#                 def score_fn(pub_id):
#                     return scorer.score_all(
#                         pub_emb[pub_id],
#                         ds_emb,
#                     )
#
#                 line = (
#                     f"[{args.model}] "
#                     f"epoch={epoch:03d} "
#                     f"loss={loss.item():.5f}"
#                 )
#
#                 if "validation_trans" in eval_splits:
#
#                     val = evaluate_ranking(
#                         score_fn,
#                         eval_splits[
#                             "validation_trans"
#                         ],
#                         k_list=(5, 10),
#                         exclude_pairs=positive_sets,
#                     )
#
#                     line += (
#                         f" | "
#                         f"MRR@10={val['MRR@10']:.4f}"
#                         f" "
#                         f"NDCG@10={val['NDCG@10']:.4f}"
#                         f" "
#                         f"Recall@10={val['Recall@10']:.4f}"
#                     )
#
#                     current_ndcg = val[
#                         "NDCG@10"
#                     ]
#
#                     if (
#                         current_ndcg
#                         > best_val_ndcg
#                     ):
#
#                         best_val_ndcg = (
#                             current_ndcg
#                         )
#
#                         best_state = {
#                             "model": {
#                                 k: v.detach()
#                                 .cpu()
#                                 .clone()
#                                 for k, v
#                                 in model.state_dict().items()
#                             },
#                             "scorer": {
#                                 k: v.detach()
#                                 .cpu()
#                                 .clone()
#                                 for k, v
#                                 in scorer.state_dict().items()
#                             },
#                         }
#
#                         patience_counter = 0
#
#                     else:
#
#                         patience_counter += 1
#
#                 print(line)
#
#                 if (
#                     args.patience > 0
#                     and patience_counter
#                     >= args.patience
#                 ):
#
#                     print(
#                         f"[{args.model}] "
#                         f"early stopping "
#                         f"at epoch {epoch}"
#                     )
#
#                     break
#
#     # Ripristina miglior modello
#     if best_state is not None:
#
#         model.load_state_dict(
#             best_state["model"]
#         )
#
#         scorer.load_state_dict(
#             best_state["scorer"]
#         )
#
#     model.eval()
#     scorer.eval()
#
#     with torch.no_grad():
#
#         if args.model == "kgat":
#             embeddings = model(
#                 x_dict,
#                 edge_index,
#                 edge_type,
#                 total_nodes,
#             )
#         else:
#             embeddings = model(
#                 x_dict,
#                 edge_index,
#                 edge_type,
#             )
#         pub_emb = embeddings[
#             pub_off:
#             pub_off + num_pub
#         ]
#
#         ds_emb = embeddings[
#             ds_off:
#             ds_off + num_dataset
#         ]
#
#     print()
#     print(
#         f"=== FINAL {args.model} ==="
#     )
#
#     final_results = {}
#
#     def score_fn(pub_id):
#         return scorer.score_all(
#             pub_emb[pub_id],
#             ds_emb,
#         )
#
#     for split in [
#         "validation_trans",
#         "test_trans",
#         "test_semi",
#         "test_ind",
#     ]:
#
#         if split not in eval_splits:
#             continue
#
#         result = evaluate_ranking(
#             score_fn,
#             eval_splits[split],
#             k_list=(5, 10),
#             exclude_pairs=positive_sets,
#         )
#
#         final_results[split] = result
#
#         print(
#             f"[{split}] "
#             + str(
#                 {
#                     k: round(v, 4)
#                     for k, v
#                     in result.items()
#                 }
#             )
#         )
#
#     return (
#         final_results,
#         model,
#         scorer,
#     )
#
#
# # ============================================================================
# # TRAIN KGCN
# # ============================================================================
#
# def train_kgcn(
#     data,
#     args,
#     graph,
#     device,
# ):
#
#     node_types = graph["node_types"]
#     offsets = graph["offsets"]
#
#     total_nodes = graph[
#         "total_nodes"
#     ]
#
#     model = KGCNEncoder(
#         node_types=node_types,
#         in_channels_dict=graph[
#             "in_channels_dict"
#         ],
#         hidden_channels=args.hidden,
#         num_relations=graph[
#             "num_relations"
#         ],
#         num_layers=args.num_layers,
#         dropout=args.dropout,
#     ).to(device)
#
#     x_dict, edge_index, edge_type = (
#         move_graph_to_device(
#             graph,
#             device,
#         )
#     )
#
#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=args.lr,
#         weight_decay=args.weight_decay,
#     )
#
#     pub_off = offsets[
#         "publication"
#     ]
#
#     ds_off = offsets[
#         "dataset"
#     ]
#
#     num_pub = data[
#         "publication"
#     ].num_nodes
#
#     num_dataset = data[
#         "dataset"
#     ].num_nodes
#
#     train_edge_index_cpu = data[
#         "publication",
#         "cites",
#         "dataset",
#     ].edge_index_train.cpu()
#
#     train_pairs = build_positive_sets(
#         train_edge_index_cpu
#     )
#
#     train_pubs = list(
#         train_pairs.keys()
#     )
#
#     negative_pool = build_negative_pool(
#         train_edge_index_cpu
#     )
#
#     eval_splits = get_eval_splits(
#         data
#     )
#
#     # ----------------------------------------------------
#     # score helper
#     # ----------------------------------------------------
#
#     def score_for_pub(
#         pub_id,
#         base_x,
#     ):
#
#         query_emb = base_x[
#             pub_off + pub_id
#         ]
#
#         conditioned = (
#             model.conditioned_embeddings(
#                 base_x,
#                 edge_index,
#                 edge_type,
#                 total_nodes,
#                 query_emb,
#             )
#         )
#
#         ds_emb = conditioned[
#             ds_off:
#             ds_off + num_dataset
#         ]
#
#         # dot product query/item
#         return (
#             ds_emb @ query_emb
#         )
#
#     # ----------------------------------------------------
#     # early stopping
#     # ----------------------------------------------------
#
#     best_val_ndcg = -1.0
#     best_state = None
#     patience_counter = 0
#
#     for epoch in range(
#         1,
#         args.epochs + 1,
#     ):
#
#         model.train()
#         optimizer.zero_grad()
#
#         base_x = model.base_embeddings(
#             x_dict
#         )
#
#         # mini-batch di publication
#         batch_size = min(
#             args.kgcn_batch_pubs,
#             len(train_pubs),
#         )
#
#         batch_pubs = random.sample(
#             train_pubs,
#             batch_size,
#         )
#
#         losses = []
#
#         for pub_id in batch_pubs:
#
#             pos_datasets = list(
#                 train_pairs[
#                     pub_id
#                 ]
#             )
#
#             if not pos_datasets:
#                 continue
#
#             query_emb = base_x[
#                 pub_off + pub_id
#             ]
#
#             conditioned = (
#                 model.conditioned_embeddings(
#                     base_x,
#                     edge_index,
#                     edge_type,
#                     total_nodes,
#                     query_emb,
#                 )
#             )
#
#             ds_emb = conditioned[
#                 ds_off:
#                 ds_off + num_dataset
#             ]
#
#             pos_idx = torch.tensor(
#                 pos_datasets,
#                 dtype=torch.long,
#                 device=device,
#             )
#
#             # negative sampling filtrato
#             local_pos_edge = torch.stack(
#                 [
#                     torch.full(
#                         (len(pos_datasets),),
#                         pub_id,
#                         dtype=torch.long,
#                     ),
#                     torch.tensor(
#                         pos_datasets,
#                         dtype=torch.long,
#                     ),
#                 ],
#                 dim=0,
#             )
#
#             neg_edge = sample_filtered_negatives(
#                 local_pos_edge,
#                 num_dataset,
#                 args.num_neg,
#                 train_pairs,
#                 negative_pool,
#             ).to(device)
#
#             pos_scores = (
#                 ds_emb[pos_idx]
#                 @ query_emb
#             )
#
#             neg_scores = (
#                 ds_emb[
#                     neg_edge[1]
#                 ]
#                 @ query_emb
#             )
#
#             pos_scores_expanded = (
#                 pos_scores.repeat_interleave(
#                     args.num_neg
#                 )
#             )
#
#             losses.append(
#                 bpr_loss(
#                     pos_scores_expanded,
#                     neg_scores,
#                 )
#             )
#
#         if not losses:
#             raise RuntimeError(
#                 "KGCN: nessuna loss generata."
#             )
#
#         loss = torch.stack(
#             losses
#         ).mean()
#
#         loss.backward()
#
#         torch.nn.utils.clip_grad_norm_(
#             model.parameters(),
#             max_norm=5.0,
#         )
#
#         optimizer.step()
#
#         # ------------------------------------------------
#         # evaluation
#         # ------------------------------------------------
#
#         if (
#             epoch % args.eval_every == 0
#             or epoch == args.epochs
#         ):
#
#             model.eval()
#
#             with torch.no_grad():
#
#                 base_x_eval = (
#                     model.base_embeddings(
#                         x_dict
#                     )
#                 )
#
#                 line = (
#                     f"[kgcn] "
#                     f"epoch={epoch:03d} "
#                     f"loss={loss.item():.5f}"
#                 )
#
#                 if "validation_trans" in eval_splits:
#
#                     validation = (
#                         eval_splits[
#                             "validation_trans"
#                         ]
#                     )
#
#                     # Per contenere il costo
#                     # usiamo il subsample solo
#                     # per early stopping.
#                     if (
#                         len(validation)
#                         >
#                         args.kgcn_val_subsample
#                     ):
#
#                         selected = random.sample(
#                             list(
#                                 validation.keys()
#                             ),
#                             args.kgcn_val_subsample,
#                         )
#
#                         validation_periodic = {
#                             p: validation[p]
#                             for p in selected
#                         }
#
#                     else:
#
#                         validation_periodic = (
#                             validation
#                         )
#
#                     def periodic_score(
#                         pub_id
#                     ):
#
#                         return score_for_pub(
#                             pub_id,
#                             base_x_eval,
#                         )
#
#                     val = evaluate_ranking(
#                         periodic_score,
#                         validation_periodic,
#                         k_list=(5, 10),
#                         exclude_pairs=train_pairs,
#                     )
#
#                     line += (
#                         f" | "
#                         f"MRR@10={val['MRR@10']:.4f}"
#                         f" "
#                         f"NDCG@10={val['NDCG@10']:.4f}"
#                         f" "
#                         f"Recall@10={val['Recall@10']:.4f}"
#                     )
#
#                     current_ndcg = val[
#                         "NDCG@10"
#                     ]
#
#                     if (
#                         current_ndcg
#                         > best_val_ndcg
#                     ):
#
#                         best_val_ndcg = (
#                             current_ndcg
#                         )
#
#                         best_state = {
#                             k: v.detach()
#                             .cpu()
#                             .clone()
#                             for k, v
#                             in model.state_dict().items()
#                         }
#
#                         patience_counter = 0
#
#                     else:
#
#                         patience_counter += 1
#
#                 print(line)
#
#                 if (
#                     args.patience > 0
#                     and patience_counter
#                     >= args.patience
#                 ):
#
#                     print(
#                         "[kgcn] "
#                         f"early stopping "
#                         f"at epoch {epoch}"
#                     )
#
#                     break
#
#     # ripristino
#     if best_state is not None:
#
#         model.load_state_dict(
#             best_state
#         )
#
#     model.eval()
#
#     final_results = {}
#
#     with torch.no_grad():
#
#         base_x_eval = model.base_embeddings(
#             x_dict
#         )
#
#         def final_score(
#             pub_id
#         ):
#
#             return score_for_pub(
#                 pub_id,
#                 base_x_eval,
#             )
#
#         print()
#         print(
#             "=== FINAL kgcn ==="
#         )
#
#         for split in [
#             "validation_trans",
#             "test_trans",
#             "test_semi",
#             "test_ind",
#         ]:
#
#             if split not in eval_splits:
#                 continue
#
#             result = evaluate_ranking(
#                 final_score,
#                 eval_splits[split],
#                 k_list=(5, 10),
#                 exclude_pairs=train_pairs,
#             )
#
#             final_results[
#                 split
#             ] = result
#
#             print(
#                 f"[{split}] "
#                 + str(
#                     {
#                         k: round(v, 4)
#                         for k, v
#                         in result.items()
#                     }
#                 )
#             )
#
#     return (
#         final_results,
#         model,
#         None,
#     )
#
#
# # ============================================================================
# # DATA LOADING
# # ============================================================================
#
# def load_data(
#     dataset,
#     split=0,
# ):
#
#     ds = loader.ScholarlyDataset(
#         root=f"datasets/{dataset}/"
#     )
#
#     data = ds[0]
#
#     if split > 0:
#
#         num_dataset = data[
#             "dataset"
#         ].num_nodes
#
#         feat_dim = data[
#             "dataset"
#         ].x.size(-1)
#
#         n_sampled = int(
#             split / 100.0
#             * num_dataset
#         )
#
#         sampled_indices = torch.randperm(
#             num_dataset
#         )[:n_sampled]
#
#         # manteniamo la stessa logica del tuo script:
#         # sostituzione con vettore costante.
#         data[
#             "dataset"
#         ].x[
#             sampled_indices
#         ] = torch.ones(
#             n_sampled,
#             feat_dim,
#             dtype=data[
#                 "dataset"
#             ].x.dtype,
#         )
#
#         print(
#             f"[load_data] split={split}% "
#             f"-> {n_sampled}/{num_dataset} "
#             f"dataset alterati"
#         )
#
#     return data
#
#
# # ============================================================================
# # CONFIGURAZIONE
# # ============================================================================
#
# BASE_DEFAULTS = dict(
#
#     hidden=128,
#
#     num_layers=2,
#
#     dropout=0.1,
#
#     lr=3e-4,
#
#     weight_decay=1e-5,
#
#     num_neg=5,
#
#     epochs=150,
#
#     eval_every=5,
#
#     patience=20,
#
#     num_bases=16,
#
#     kgcn_batch_pubs=64,
#
#     kgcn_val_subsample=200,
#
#     split=0,
#
#     cpu=False,
#
#     dataset="mes",
#
# )
#
#
# DATASET_DEFAULTS = {
#
#     "mes": {
#         "epochs": 150,
#         "eval_every": 5,
#         "patience": 20,
#     },
#
#     "pubmed": {
#         "epochs": 100,
#         "eval_every": 5,
#         "patience": 10,
#     },
#
# }
#
#
# MODEL_DEFAULTS = {
#
#     "rgcn": {
#
#         "hidden": 128,
#
#         "num_layers": 2,
#
#         "dropout": 0.10,
#
#         "lr": 3e-4,
#
#         "num_bases": 16,
#
#     },
#
#     "kgat": {
#
#         "hidden": 64,
#
#         "num_layers": 3,
#
#         "dropout": 0.10,
#
#         "lr": 3e-4,
#
#     },
#
#     "kgcn": {
#
#         "hidden": 64,
#
#         "num_layers": 2,
#
#         "dropout": 0.10,
#
#         "lr": 3e-4,
#
#         "kgcn_batch_pubs": 64,
#
#         "kgcn_val_subsample": 200,
#
#     },
#
# }
#
#
# DATASET_MODEL_DEFAULTS = {
#
#     (
#         "pubmed",
#         "kgcn",
#     ): {
#
#         "hidden": 32,
#
#         "kgcn_batch_pubs": 32,
#
#         "kgcn_val_subsample": 300,
#
#     },
#
#     (
#         "pubmed",
#         "kgat",
#     ): {
#
#         "hidden": 32,
#
#     },
#
# }
#
#
# def load_params(
#     model,
#     dataset,
#     cli_args,
# ):
#
#     params = dict(
#         BASE_DEFAULTS
#     )
#
#     params.update(
#         DATASET_DEFAULTS.get(
#             dataset,
#             {},
#         )
#     )
#
#     params.update(
#         MODEL_DEFAULTS.get(
#             model,
#             {},
#         )
#     )
#
#     params.update(
#         DATASET_MODEL_DEFAULTS.get(
#             (
#                 dataset,
#                 model,
#             ),
#             {},
#         )
#     )
#
#     # CLI solo se specificato.
#     for key, value in vars(
#         cli_args
#     ).items():
#
#         if value is not None:
#             params[key] = value
#
#     params["model"] = model
#     params["dataset"] = dataset
#
#     if params.get("root") is None:
#         params["root"] = (
#             f"datasets/{dataset}"
#         )
#
#     args = argparse.Namespace(
#         **params
#     )
#
#     print()
#     print(
#         f"[config] "
#         f"dataset={dataset} "
#         f"model={model}"
#     )
#
#     print(
#         f"[config] "
#         f"hidden={args.hidden} "
#         f"layers={args.num_layers} "
#         f"dropout={args.dropout}"
#     )
#
#     print(
#         f"[config] "
#         f"lr={args.lr} "
#         f"wd={args.weight_decay}"
#     )
#
#     print(
#         f"[config] "
#         f"negatives={args.num_neg} "
#         f"epochs={args.epochs}"
#     )
#
#     if model == "rgcn":
#
#         print(
#             f"[config] "
#             f"num_bases={args.num_bases}"
#         )
#
#     if model == "kgcn":
#
#         print(
#             f"[config] "
#             f"batch_pubs={args.kgcn_batch_pubs} "
#             f"val_subsample={args.kgcn_val_subsample}"
#         )
#
#     return args
#
#
# # ============================================================================
# # DISPATCH
# # ============================================================================
#
# def train_and_evaluate(
#     data,
#     args,
# ):
#
#     device = torch.device(
#         "cuda"
#         if (
#             torch.cuda.is_available()
#             and not args.cpu
#         )
#         else "cpu"
#     )
#
#     print(
#         f"\nDevice: {device}"
#     )
#
#     graph = build_homogeneous_graph(
#         data
#     )
#
#     if args.model in (
#         "rgcn",
#         "kgat",
#     ):
#
#         return train_static_model(
#             data,
#             args,
#             graph,
#             device,
#         )
#
#     if args.model == "kgcn":
#
#         return train_kgcn(
#             data,
#             args,
#             graph,
#             device,
#         )
#
#     raise ValueError(
#         f"Modello non supportato: "
#         f"{args.model}"
#     )
#
#
# # ============================================================================
# # ARGUMENTS
# # ============================================================================
#
# def parse_args():
#
#     p = argparse.ArgumentParser()
#
#     p.add_argument(
#         "--dataset",
#         "-dataset",
#         type=str,
#         default="mes",
#     )
#     p.add_argument(
#         "--split",
#         "-split",
#         type=float,
#         default=0,
#     )
#     p.add_argument(
#         "--model",
#         "-model",
#         type=str,
#         default=None,
#         choices=[
#             "rgcn",
#             "kgat",
#             "kgcn",
#         ],
#     )
#
#     # IMPORTANTISSIMO:
#     # default=None perché questi valori vengono definiti
#     # da MODEL_DEFAULTS / DATASET_DEFAULTS.
#     p.add_argument(
#         "--hidden",
#         type=int,
#         default=None,
#     )
#
#     p.add_argument(
#         "--num_layers",
#         type=int,
#         default=None,
#     )
#
#     p.add_argument(
#         "--dropout",
#         type=float,
#         default=None,
#     )
#
#     p.add_argument(
#         "--lr",
#         type=float,
#         default=None,
#     )
#
#     p.add_argument(
#         "--weight_decay",
#         type=float,
#         default=None,
#     )
#
#     p.add_argument(
#         "--num_neg",
#         type=int,
#         default=None,
#     )
#
#     p.add_argument(
#         "--epochs",
#         type=int,
#         default=None,
#     )
#
#     p.add_argument(
#         "--eval_every",
#         type=int,
#         default=None,
#     )
#
#     p.add_argument(
#         "--patience",
#         type=int,
#         default=None,
#     )
#
#     p.add_argument(
#         "--num_bases",
#         type=int,
#         default=None,
#     )
#
#     p.add_argument(
#         "--kgcn_batch_pubs",
#         type=int,
#         default=None,
#     )
#
#     p.add_argument(
#         "--kgcn_val_subsample",
#         type=int,
#         default=None,
#     )
#
#
#     p.add_argument(
#         "--root",
#         type=str,
#         default=None,
#     )
#
#     p.add_argument(
#         "--cpu",
#         action="store_true",
#     )
#
#     return p.parse_args()
#
#
# # ============================================================================
# # MAIN
# # ============================================================================
#
# if __name__ == "__main__":
#
#     base_args = parse_args()
#
#     set_seed(SEED)
#
#     methods = (
#         [base_args.model]
#         if base_args.model is not None
#         else [
#             "vgae",
#             "hvgae",
#             "kgat",
#             "kgcn",
#             "rgcn",
#
#
#         ]
#     )
#
#     for method in methods:
#
#         print()
#         print(
#             "=================================================="
#         )
#         print(
#             f" DATASET: {base_args.dataset}"
#         )
#         print(
#             f" MODEL:   {method}"
#         )
#         print(
#             "=================================================="
#         )
#
#         args = load_params(
#             method,
#             base_args.dataset,
#             base_args,
#         )
#
#         data = load_data(
#             args.dataset,
#             split=args.split,
#         )
#
#         results, model, scorer = (
#             train_and_evaluate(
#                 data,
#                 args,
#             )
#         )
#
#         print()
#         print(
#             f"[{method}] training terminato."
#         )



import argparse
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv, SAGEConv
from torch_geometric.utils import softmax as scatter_softmax

import loader


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# HOMOGENEOUS GRAPH
# ============================================================================

def build_homogeneous_graph(data: HeteroData, feature_key="x"):

    node_types = list(data.node_types)

    num_nodes_dict = {
        nt: data[nt].num_nodes
        for nt in node_types
    }

    offsets = {}
    total_nodes = 0

    for nt in node_types:
        offsets[nt] = total_nodes
        total_nodes += num_nodes_dict[nt]

    x_dict = {}
    in_channels_dict = {}

    for nt in node_types:

        feat = getattr(
            data[nt],
            feature_key,
            None,
        )

        if feat is None:
            feat = data[nt].net_x

        feat = feat.float()

        x_dict[nt] = feat
        in_channels_dict[nt] = feat.size(-1)

    edge_index_list = []
    edge_type_list = []
    rel2id = {}

    def add_relation(
        src_type,
        rel_name,
        dst_type,
        edge_index,
        bidirectional=True,
    ):

        if edge_index is None:
            return

        if edge_index.numel() == 0:
            return

        key_fwd = (
            src_type,
            rel_name,
            dst_type,
        )

        if key_fwd not in rel2id:
            rel2id[key_fwd] = len(rel2id)

        rid_fwd = rel2id[key_fwd]

        src_global = (
            edge_index[0]
            + offsets[src_type]
        )

        dst_global = (
            edge_index[1]
            + offsets[dst_type]
        )

        edge_index_list.append(
            torch.stack(
                [
                    src_global,
                    dst_global,
                ],
                dim=0,
            )
        )

        edge_type_list.append(
            torch.full(
                (edge_index.size(1),),
                rid_fwd,
                dtype=torch.long,
            )
        )

        if bidirectional:

            key_rev = (
                dst_type,
                rel_name + "_rev",
                src_type,
            )

            if key_rev not in rel2id:
                rel2id[key_rev] = len(rel2id)

            rid_rev = rel2id[key_rev]

            edge_index_list.append(
                torch.stack(
                    [
                        dst_global,
                        src_global,
                    ],
                    dim=0,
                )
            )

            edge_type_list.append(
                torch.full(
                    (edge_index.size(1),),
                    rid_rev,
                    dtype=torch.long,
                )
            )

    for et in data.edge_types:

        src_type, rel_name, dst_type = et

        # publication -> dataset:
        # ONLY training edges enter message passing.
        if (
            src_type == "publication"
            and dst_type == "dataset"
            and rel_name == "cites"
        ):

            edge_index = data[et].get(
                "edge_index_train",
                None,
            )

            add_relation(
                src_type,
                rel_name,
                dst_type,
                edge_index,
            )

            continue

        edge_index = data[et].get(
            "edge_index",
            None,
        )

        add_relation(
            src_type,
            rel_name,
            dst_type,
            edge_index,
        )

    if not edge_index_list:
        raise RuntimeError(
            "Nessun edge disponibile nel grafo."
        )

    edge_index = torch.cat(
        edge_index_list,
        dim=1,
    )

    edge_type = torch.cat(
        edge_type_list,
        dim=0,
    )

    print(
        f"[graph] nodes={total_nodes} "
        f"edges={edge_index.size(1)} "
        f"relations={len(rel2id)}"
    )

    return {
        "x_dict": x_dict,
        "in_channels_dict": in_channels_dict,
        "node_types": node_types,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "num_relations": len(rel2id),
        "offsets": offsets,
        "num_nodes_dict": num_nodes_dict,
        "total_nodes": total_nodes,
    }


# ============================================================================
# COMMON SCORERS
# ============================================================================

class InnerProductScorer(nn.Module):

    def forward(
        self,
        src_emb,
        dst_emb,
    ):
        return (
            src_emb * dst_emb
        ).sum(dim=-1)

    def score_all(
        self,
        src_emb,
        dst_all_emb,
    ):
        return dst_all_emb @ src_emb


class BilinearScorer(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(dim, dim)
        )

        nn.init.xavier_uniform_(
            self.weight
        )

    def forward(
        self,
        src_emb,
        dst_emb,
    ):
        return (
            src_emb @ self.weight
            * dst_emb
        ).sum(dim=-1)

    def score_all(
        self,
        src_emb,
        dst_all_emb,
    ):
        transformed = (
            self.weight.t()
            @ src_emb
        )

        return dst_all_emb @ transformed


# ============================================================================
# RGCN
# ============================================================================

class RGCNEncoder(nn.Module):

    def __init__(
        self,
        node_types,
        in_channels_dict,
        hidden_channels,
        num_relations,
        num_bases=16,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.node_types = node_types
        self.dropout = dropout

        self.input_proj = nn.ModuleDict()

        for nt in node_types:

            self.input_proj[nt] = nn.Linear(
                in_channels_dict[nt],
                hidden_channels,
            )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):

            self.convs.append(
                RGCNConv(
                    hidden_channels,
                    hidden_channels,
                    num_relations,
                    num_bases=num_bases,
                )
            )

            self.norms.append(
                nn.LayerNorm(
                    hidden_channels
                )
            )

    def forward(
        self,
        x_dict,
        edge_index,
        edge_type,
    ):

        x = torch.cat(
            [
                self.input_proj[nt](
                    x_dict[nt]
                )
                for nt in self.node_types
            ],
            dim=0,
        )

        for conv, norm in zip(
            self.convs,
            self.norms,
        ):

            residual = x

            h = conv(
                x,
                edge_index,
                edge_type,
            )

            h = F.relu(h)

            h = F.dropout(
                h,
                p=self.dropout,
                training=self.training,
            )

            x = norm(
                h + residual
            )

        return x


# ============================================================================
# KGAT
# ============================================================================

class KGATEncoder(nn.Module):

    def __init__(
        self,
        node_types,
        in_channels_dict,
        hidden_channels,
        num_relations,
        num_layers=3,
        dropout=0.1,
    ):
        super().__init__()

        self.node_types = node_types
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.out_dim = (
            hidden_channels
            * (num_layers + 1)
        )

        self.input_proj = nn.ModuleDict()

        for nt in node_types:

            self.input_proj[nt] = nn.Linear(
                in_channels_dict[nt],
                hidden_channels,
            )

        self.rel_emb = nn.Embedding(
            num_relations,
            hidden_channels,
        )

        self.rel_transform = nn.Embedding(
            num_relations,
            hidden_channels,
        )

        nn.init.ones_(
            self.rel_transform.weight
        )

        self.attn = nn.ModuleList()
        self.w1 = nn.ModuleList()
        self.w2 = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):

            self.attn.append(
                nn.Sequential(
                    nn.Linear(
                        hidden_channels * 3,
                        hidden_channels,
                    ),
                    nn.LeakyReLU(0.2),
                    nn.Linear(
                        hidden_channels,
                        1,
                    ),
                )
            )

            self.w1.append(
                nn.Linear(
                    hidden_channels,
                    hidden_channels,
                )
            )

            self.w2.append(
                nn.Linear(
                    hidden_channels,
                    hidden_channels,
                )
            )

            self.norms.append(
                nn.LayerNorm(
                    hidden_channels
                )
            )

    def forward(
        self,
        x_dict,
        edge_index,
        edge_type,
        total_nodes,
    ):

        x = torch.cat(
            [
                self.input_proj[nt](
                    x_dict[nt]
                )
                for nt in self.node_types
            ],
            dim=0,
        )

        src, dst = edge_index

        all_layers = [x]

        for layer in range(
            self.num_layers
        ):

            rel_scale = self.rel_transform(
                edge_type
            )

            rel_vec = self.rel_emb(
                edge_type
            )

            head = (
                x[src]
                * rel_scale
            )

            tail = (
                x[dst]
                * rel_scale
            )

            attn_input = torch.cat(
                [
                    head,
                    rel_vec,
                    tail,
                ],
                dim=-1,
            )

            score = (
                self.attn[layer](
                    attn_input
                )
                .squeeze(-1)
            )

            alpha = scatter_softmax(
                score,
                src,
                num_nodes=total_nodes,
            )

            alpha = F.dropout(
                alpha,
                p=self.dropout,
                training=self.training,
            )

            messages = (
                alpha.unsqueeze(-1)
                *
                (
                    tail + rel_vec
                )
            )

            neigh = x.new_zeros(
                total_nodes,
                self.hidden_channels,
            )

            neigh.index_add_(
                0,
                src,
                messages,
            )

            first = self.w1[layer](
                x + neigh
            )

            second = self.w2[layer](
                x * neigh
            )

            h = (
                F.leaky_relu(
                    first,
                    0.2,
                )
                +
                F.leaky_relu(
                    second,
                    0.2,
                )
            )

            x = self.norms[layer](
                x + h
            )

            x = F.dropout(
                x,
                p=self.dropout,
                training=self.training,
            )

            all_layers.append(x)

        return torch.cat(
            all_layers,
            dim=-1,
        )


# ============================================================================
# KGCN
# ============================================================================

class KGCNEncoder(nn.Module):

    def __init__(
        self,
        node_types,
        in_channels_dict,
        hidden_channels,
        num_relations,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.node_types = node_types
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.input_proj = nn.ModuleDict()

        for nt in node_types:

            self.input_proj[nt] = nn.Linear(
                in_channels_dict[nt],
                hidden_channels,
            )

        self.rel_emb = nn.Embedding(
            num_relations,
            hidden_channels,
        )

        self.rel_transform = nn.Embedding(
            num_relations,
            hidden_channels,
        )

        nn.init.ones_(
            self.rel_transform.weight
        )

        self.query_proj = nn.Linear(
            hidden_channels,
            hidden_channels,
        )

        self.self_transform = nn.ModuleList()
        self.neigh_transform = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):

            self.self_transform.append(
                nn.Linear(
                    hidden_channels,
                    hidden_channels,
                )
            )

            self.neigh_transform.append(
                nn.Linear(
                    hidden_channels,
                    hidden_channels,
                )
            )

            self.attention.append(
                nn.Sequential(
                    nn.Linear(
                        hidden_channels * 3,
                        hidden_channels,
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        hidden_channels,
                        1,
                    ),
                )
            )

            self.norms.append(
                nn.LayerNorm(
                    hidden_channels
                )
            )

    def base_embeddings(
        self,
        x_dict,
    ):

        return torch.cat(
            [
                self.input_proj[nt](
                    x_dict[nt]
                )
                for nt in self.node_types
            ],
            dim=0,
        )

    def conditioned_embeddings(
        self,
        x0,
        edge_index,
        edge_type,
        total_nodes,
        query_emb,
    ):

        src, dst = edge_index

        query = self.query_proj(
            query_emb
        )

        x = x0

        for layer in range(
            self.num_layers
        ):

            rel_vec = self.rel_emb(
                edge_type
            )

            rel_scale = self.rel_transform(
                edge_type
            )

            neighbor = (
                x[dst]
                * rel_scale
            )

            query_edge = query.expand(
                neighbor.size(0),
                -1,
            )

            attn_input = torch.cat(
                [
                    query_edge,
                    rel_vec,
                    neighbor,
                ],
                dim=-1,
            )

            attention_score = (
                self.attention[layer](
                    attn_input
                )
                .squeeze(-1)
            )

            alpha = scatter_softmax(
                attention_score,
                src,
                num_nodes=total_nodes,
            )

            alpha = F.dropout(
                alpha,
                p=self.dropout,
                training=self.training,
            )

            message = (
                alpha.unsqueeze(-1)
                *
                (
                    neighbor
                    + rel_vec
                )
            )

            neigh = x.new_zeros(
                total_nodes,
                self.hidden_channels,
            )

            neigh.index_add_(
                0,
                src,
                message,
            )

            self_part = self.self_transform[layer](
                x
            )

            neigh_part = self.neigh_transform[layer](
                neigh
            )

            h = F.relu(
                self_part + neigh_part
            )

            h = F.dropout(
                h,
                p=self.dropout,
                training=self.training,
            )

            x = self.norms[layer](
                x + h
            )

        return x


# ============================================================================
# VGAE
# ============================================================================

class VGAEEncoder(nn.Module):

    def __init__(
        self,
        node_types,
        in_channels_dict,
        hidden_channels,
        num_relations,
        num_bases=16,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.node_types = node_types
        self.dropout = dropout

        self.input_proj = nn.ModuleDict()

        for nt in node_types:

            self.input_proj[nt] = nn.Linear(
                in_channels_dict[nt],
                hidden_channels,
            )

        self.convs = nn.ModuleList()

        for _ in range(
            max(num_layers - 1, 0)
        ):

            self.convs.append(
                RGCNConv(
                    hidden_channels,
                    hidden_channels,
                    num_relations,
                    num_bases=num_bases,
                )
            )

        self.conv_mu = RGCNConv(
            hidden_channels,
            hidden_channels,
            num_relations,
            num_bases=num_bases,
        )

        self.conv_logvar = RGCNConv(
            hidden_channels,
            hidden_channels,
            num_relations,
            num_bases=num_bases,
        )

    def forward(
        self,
        x_dict,
        edge_index,
        edge_type,
    ):

        x = torch.cat(
            [
                self.input_proj[nt](
                    x_dict[nt]
                )
                for nt in self.node_types
            ],
            dim=0,
        )

        for conv in self.convs:

            residual = x

            h = conv(
                x,
                edge_index,
                edge_type,
            )

            h = F.relu(h)

            h = F.dropout(
                h,
                p=self.dropout,
                training=self.training,
            )

            if h.shape == residual.shape:
                h = h + residual

            x = h

        mu = self.conv_mu(
            x,
            edge_index,
            edge_type,
        )

        logvar = self.conv_logvar(
            x,
            edge_index,
            edge_type,
        )

        logvar = torch.clamp(
            logvar,
            min=-8.0,
            max=8.0,
        )

        return mu, logvar

    def reparameterize(
        self,
        mu,
        logvar,
    ):

        if self.training:

            std = torch.exp(
                0.5 * logvar
            )

            eps = torch.randn_like(
                std
            )

            return mu + eps * std

        return mu

    @staticmethod
    def kl_loss(
        mu,
        logvar,
    ):

        return -0.5 * torch.mean(
            torch.sum(
                1
                + logvar
                - mu.pow(2)
                - logvar.exp(),
                dim=1,
            )
        )


# ============================================================================
# HVGAE
# ============================================================================

class HVGAEPaperEncoder(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        latent_dim,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        c_in = in_channels

        for _ in range(num_layers):

            self.convs.append(
                SAGEConv(
                    c_in,
                    hidden_channels,
                    aggr="mean",
                )
            )

            self.norms.append(
                nn.LayerNorm(
                    hidden_channels
                )
            )

            c_in = hidden_channels

        self.mu_head = nn.Linear(
            hidden_channels,
            latent_dim,
        )

        self.var_head = nn.Linear(
            hidden_channels,
            latent_dim,
        )

    def forward(
        self,
        x,
        edge_index,
    ):

        h = x

        for conv, norm in zip(
            self.convs,
            self.norms,
        ):

            residual = h

            h = conv(
                h,
                edge_index,
            )

            h = F.gelu(h)

            h = F.dropout(
                h,
                p=self.dropout,
                training=self.training,
            )

            if residual.shape[-1] == h.shape[-1]:
                h = h + residual

            h = norm(h)

        mu = self.mu_head(h)

        logvar = self.var_head(h)

        logvar = torch.clamp(
            logvar,
            min=-6.0,
            max=4.0,
        )

        var = torch.exp(logvar)

        return mu, var


class HVGAEDatasetEncoder(nn.Module):

    def __init__(
        self,
        num_pub,
        hidden_channels,
        latent_dim,
        dropout=0.1,
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            num_pub,
            hidden_channels,
        )

        self.fc2 = nn.Linear(
            hidden_channels,
            hidden_channels,
        )

        self.norm1 = nn.LayerNorm(
            hidden_channels
        )

        self.norm2 = nn.LayerNorm(
            hidden_channels
        )

        self.dropout = dropout

        self.mu_head = nn.Linear(
            hidden_channels,
            latent_dim,
        )

        self.var_head = nn.Linear(
            hidden_channels,
            latent_dim,
        )

    def forward(
        self,
        H_T,
    ):

        h = self.fc1(H_T)
        h = self.norm1(h)
        h = F.gelu(h)

        h = F.dropout(
            h,
            p=self.dropout,
            training=self.training,
        )

        h = self.fc2(h)
        h = self.norm2(h)
        h = F.gelu(h)

        mu = self.mu_head(h)

        logvar = self.var_head(h)

        logvar = torch.clamp(
            logvar,
            min=-6.0,
            max=4.0,
        )

        var = torch.exp(logvar)

        return mu, var


# ============================================================================
# GENERAL UTILS
# ============================================================================

def bpr_loss(
    positive_scores,
    negative_scores,
):
    return -F.logsigmoid(
        positive_scores
        - negative_scores
    ).mean()


def build_positive_sets(
    edge_index,
):

    result = defaultdict(set)

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    for s, d in zip(src, dst):
        result[s].add(d)

    return result


def build_negative_pool(
    edge_index,
):

    return torch.unique(
        edge_index[1]
    )


def sample_filtered_negatives(
    pos_edge_index,
    num_dataset,
    num_neg_per_pos,
    positive_sets,
    dataset_pool=None,
):

    if dataset_pool is None:
        dataset_pool = torch.arange(
            num_dataset,
            dtype=torch.long,
        )

    negatives = []

    src_list = pos_edge_index[
        0
    ].tolist()

    for pub_id in src_list:

        positives = positive_sets[
            pub_id
        ]

        sampled = []

        attempts = 0

        max_attempts = (
            10 * num_neg_per_pos
            + 20
        )

        while (
            len(sampled)
            < num_neg_per_pos
            and attempts
            < max_attempts
        ):

            idx = random.randrange(
                dataset_pool.numel()
            )

            candidate = int(
                dataset_pool[idx]
            )

            attempts += 1

            if candidate in positives:
                continue

            sampled.append(
                candidate
            )

        if len(sampled) < num_neg_per_pos:

            candidates = [
                d
                for d in range(num_dataset)
                if d not in positives
            ]

            if not candidates:
                candidates = list(
                    range(num_dataset)
                )

            while (
                len(sampled)
                < num_neg_per_pos
            ):

                sampled.append(
                    random.choice(
                        candidates
                    )
                )

        negatives.extend(
            sampled
        )

    neg_src = []

    for pub_id in src_list:
        neg_src.extend(
            [pub_id] * num_neg_per_pos
        )

    return torch.tensor(
        [
            neg_src,
            negatives,
        ],
        dtype=torch.long,
    )


def sample_negative_citations(
    pos_edge_index,
    num_pub,
    num_neg_per_pos=1,
):

    E = pos_edge_index.size(1)

    neg_dst = torch.randint(
        0,
        num_pub,
        (
            E * num_neg_per_pos,
        ),
    )

    neg_src = (
        pos_edge_index[0]
        .repeat_interleave(
            num_neg_per_pos
        )
    )

    return torch.stack(
        [
            neg_src,
            neg_dst,
        ],
        dim=0,
    )


def get_citation_edge_index(data):

    for et in data.edge_types:

        src_type, _, dst_type = et

        if (
            src_type == "publication"
            and dst_type == "publication"
        ):

            ei = data[et].get(
                "edge_index",
                None,
            )

            if (
                ei is not None
                and ei.numel() > 0
            ):
                return ei

    raise ValueError(
        "HVGAE richiede una relazione "
        "publication -> publication."
    )


# ============================================================================
# METRICS
# ============================================================================

def dcg_at_k(
    relevances,
    k,
):

    relevances = relevances[:k]

    if len(relevances) == 0:
        return 0.0

    gains = (
        2 ** relevances
        - 1
    )

    discounts = np.log2(
        np.arange(
            2,
            len(relevances) + 2,
        )
    )

    return float(
        np.sum(
            gains / discounts
        )
    )


def ndcg_at_k(
    ranked_relevances,
    num_relevant_total,
    k,
):

    dcg = dcg_at_k(
        np.asarray(ranked_relevances),
        k,
    )

    ideal = np.ones(
        min(k, num_relevant_total)
    )

    idcg = dcg_at_k(
        ideal,
        k,
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def build_eval_pairs(
    edge_label_index,
):

    pairs = defaultdict(set)

    src = edge_label_index[0].tolist()
    dst = edge_label_index[1].tolist()

    for s, d in zip(src, dst):
        pairs[s].add(d)

    return pairs


def evaluate_ranking(
    score_fn,
    eval_pairs,
    k_list=(5, 10),
    exclude_pairs=None,
):

    mrr_at = {
        k: []
        for k in k_list
    }

    ndcg_at = {
        k: []
        for k in k_list
    }

    recall_at = {
        k: []
        for k in k_list
    }

    with torch.no_grad():

        for pub_id, relevant_set in eval_pairs.items():

            if not relevant_set:
                continue

            scores = score_fn(
                pub_id
            ).clone()

            if exclude_pairs is not None:

                known = exclude_pairs.get(
                    pub_id,
                    set(),
                )

                if known:

                    known_idx = torch.tensor(
                        list(known),
                        dtype=torch.long,
                        device=scores.device,
                    )

                    scores[
                        known_idx
                    ] = -torch.inf

            ranked_idx = torch.argsort(
                scores,
                descending=True,
            ).cpu().numpy()

            rank_first = None

            for pos, idx in enumerate(
                ranked_idx
            ):

                if idx in relevant_set:

                    rank_first = pos + 1
                    break

            for k in k_list:

                if (
                    rank_first is not None
                    and rank_first <= k
                ):

                    mrr_at[k].append(
                        1.0 / rank_first
                    )

                else:

                    mrr_at[k].append(
                        0.0
                    )

                top_k = ranked_idx[:k]

                hits = sum(
                    1
                    for idx in top_k
                    if idx in relevant_set
                )

                recall_at[k].append(
                    hits
                    /
                    min(
                        len(relevant_set),
                        k,
                    )
                )

                rel_at_k = [
                    1
                    if idx in relevant_set
                    else 0
                    for idx in top_k
                ]

                ndcg_at[k].append(
                    ndcg_at_k(
                        rel_at_k,
                        len(relevant_set),
                        k,
                    )
                )

    results = {}

    for k in k_list:

        results[
            f"MRR@{k}"
        ] = (
            float(
                np.mean(mrr_at[k])
            )
            if mrr_at[k]
            else 0.0
        )

        results[
            f"NDCG@{k}"
        ] = (
            float(
                np.mean(ndcg_at[k])
            )
            if ndcg_at[k]
            else 0.0
        )

        results[
            f"Recall@{k}"
        ] = (
            float(
                np.mean(recall_at[k])
            )
            if recall_at[k]
            else 0.0
        )

    return results


def get_eval_splits(data):

    relation = data[
        "publication",
        "cites",
        "dataset",
    ]

    result = {}

    for split in [
        "validation_trans",
        "test_trans",
        "test_semi",
        "test_ind",
    ]:

        key = (
            f"edge_label_index_{split}"
        )

        if key in relation:

            result[split] = build_eval_pairs(
                relation[key]
            )

    return result


# ============================================================================
# VGAE HELPERS
# ============================================================================

def train_vgae(
    data,
    args,
    graph,
    device,
):

    node_types = graph["node_types"]
    offsets = graph["offsets"]

    num_pub = data["publication"].num_nodes
    num_dataset = data["dataset"].num_nodes

    model = VGAEEncoder(
        node_types=node_types,
        in_channels_dict=graph[
            "in_channels_dict"
        ],
        hidden_channels=args.hidden,
        num_relations=graph[
            "num_relations"
        ],
        num_bases=args.num_bases,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    decoder = InnerProductScorer().to(
        device
    )

    x_dict = {
        nt: x.to(device)
        for nt, x in graph[
            "x_dict"
        ].items()
    }

    edge_index = graph[
        "edge_index"
    ].to(device)

    edge_type = graph[
        "edge_type"
    ].to(device)

    pub_off = offsets[
        "publication"
    ]

    ds_off = offsets[
        "dataset"
    ]

    train_edge_index_cpu = data[
        "publication",
        "cites",
        "dataset",
    ].edge_index_train.cpu()

    train_edge_index = (
        train_edge_index_cpu.to(device)
    )

    positive_sets = build_positive_sets(
        train_edge_index_cpu
    )

    negative_pool = build_negative_pool(
        train_edge_index_cpu
    )

    eval_splits = get_eval_splits(data)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_ndcg = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()
        optimizer.zero_grad()

        mu, logvar = model(
            x_dict,
            edge_index,
            edge_type,
        )

        z = model.reparameterize(
            mu,
            logvar,
        )

        pub_z = z[
            pub_off:
            pub_off + num_pub
        ]

        ds_z = z[
            ds_off:
            ds_off + num_dataset
        ]

        pos_scores = decoder(
            pub_z[
                train_edge_index[0]
            ],
            ds_z[
                train_edge_index[1]
            ],
        )

        neg_edge_index = (
            sample_filtered_negatives(
                train_edge_index_cpu,
                num_dataset,
                args.num_neg,
                positive_sets,
                negative_pool,
            ).to(device)
        )

        neg_scores = decoder(
            pub_z[
                neg_edge_index[0]
            ],
            ds_z[
                neg_edge_index[1]
            ],
        )

        recon_scores = torch.cat(
            [
                pos_scores,
                neg_scores,
            ]
        )

        recon_labels = torch.cat(
            [
                torch.ones_like(
                    pos_scores
                ),
                torch.zeros_like(
                    neg_scores
                ),
            ]
        )

        recon_loss = F.binary_cross_entropy_with_logits(
            recon_scores,
            recon_labels,
        )

        kl = model.kl_loss(
            mu,
            logvar,
        )

        warmup = min(
            1.0,
            epoch
            /
            max(
                1,
                args.kl_warmup_epochs,
            ),
        )

        beta = (
            args.kl_weight
            * warmup
        )

        loss = (
            recon_loss
            +
            beta * kl
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )

        optimizer.step()

        if (
            epoch % args.eval_every == 0
            or epoch == args.epochs
        ):

            model.eval()

            with torch.no_grad():

                mu_eval, _ = model(
                    x_dict,
                    edge_index,
                    edge_type,
                )

                pub_eval = mu_eval[
                    pub_off:
                    pub_off + num_pub
                ]

                ds_eval = mu_eval[
                    ds_off:
                    ds_off + num_dataset
                ]

                def score_fn(
                    pub_id
                ):
                    return decoder.score_all(
                        pub_eval[pub_id],
                        ds_eval,
                    )

                line = (
                    f"[vgae] "
                    f"epoch={epoch:03d} "
                    f"loss={loss.item():.5f} "
                    f"recon={recon_loss.item():.5f} "
                    f"kl={kl.item():.5f} "
                    f"beta={beta:.6f}"
                )

                if "validation_trans" in eval_splits:

                    val = evaluate_ranking(
                        score_fn,
                        eval_splits[
                            "validation_trans"
                        ],
                        k_list=(5, 10),
                        exclude_pairs=positive_sets,
                    )

                    line += (
                        f" | MRR@10={val['MRR@10']:.4f}"
                        f" NDCG@10={val['NDCG@10']:.4f}"
                        f" Recall@10={val['Recall@10']:.4f}"
                    )

                    if (
                        val["NDCG@10"]
                        > best_ndcg
                    ):

                        best_ndcg = val[
                            "NDCG@10"
                        ]

                        best_state = {
                            k: v.detach()
                            .cpu()
                            .clone()
                            for k, v
                            in model.state_dict().items()
                        }

                        patience_counter = 0

                    else:
                        patience_counter += 1

                print(line)

                if (
                    args.patience > 0
                    and patience_counter
                    >= args.patience
                ):
                    break

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    model.eval()

    final_results = {}

    with torch.no_grad():

        mu_eval, _ = model(
            x_dict,
            edge_index,
            edge_type,
        )

        pub_eval = mu_eval[
            pub_off:
            pub_off + num_pub
        ]

        ds_eval = mu_eval[
            ds_off:
            ds_off + num_dataset
        ]

        def score_fn(
            pub_id
        ):
            return decoder.score_all(
                pub_eval[pub_id],
                ds_eval,
            )

        print(
            "\n=== FINAL vgae ==="
        )

        for split in [
            "validation_trans",
            "test_trans",
            "test_semi",
            "test_ind",
        ]:

            if split not in eval_splits:
                continue

            result = evaluate_ranking(
                score_fn,
                eval_splits[split],
                k_list=(5, 10),
                exclude_pairs=positive_sets,
            )

            final_results[
                split
            ] = result

            print(
                f"[{split}] {result}"
            )

    return (
        final_results,
        model,
        decoder,
    )


# ============================================================================
# HVGAE HELPERS
# ============================================================================

def reparameterize_gaussian(
    mu,
    var,
    training=True,
):

    if training:

        std = torch.sqrt(
            var + 1e-8
        )

        eps = torch.randn_like(std)

        return mu + eps * std

    return mu


def kl_to_standard_normal(
    mu,
    var,
):

    logvar = torch.log(
        var + 1e-8
    )

    return -0.5 * torch.mean(
        torch.sum(
            1
            + logvar
            - mu.pow(2)
            - var,
            dim=1,
        )
    )


def kl_query_dataset(
    mu_q,
    var_q,
    mu_d,
    var_d,
):

    sigma_q = torch.sqrt(
        var_q + 1e-8
    )

    sigma_d = torch.sqrt(
        var_d + 1e-8
    )

    term1 = torch.log(
        sigma_d
        /
        sigma_q.unsqueeze(0)
        + 1e-8
    )

    term2 = (
        var_q.unsqueeze(0)
        +
        (
            mu_q.unsqueeze(0)
            - mu_d
        ).pow(2)
    ) / (
        2 * var_d + 1e-8
    )

    kl = (
        term1
        + term2
        - 0.5
    ).sum(dim=-1)

    # evaluation sorts descending:
    # lower KL = better => negative KL
    return -kl


def train_hvgae(
    data,
    args,
    device,
):

    num_pub = data["publication"].num_nodes
    num_dataset = data["dataset"].num_nodes

    x_p = getattr(
        data["publication"],
        "x",
        None,
    )

    if x_p is None:
        x_p = data["publication"].net_x

    x_p = x_p.float().to(device)

    citation_edge_index = (
        get_citation_edge_index(data)
        .to(device)
    )

    train_edge_index_cpu = data[
        "publication",
        "cites",
        "dataset",
    ].edge_index_train.cpu()

    train_edge_index = (
        train_edge_index_cpu.to(device)
    )

    # H_P,D^T
    H_T = torch.zeros(
        num_dataset,
        num_pub,
        device=device,
    )

    H_T[
        train_edge_index[1],
        train_edge_index[0],
    ] = 1.0

    positive_sets = build_positive_sets(
        train_edge_index_cpu
    )

    negative_pool = build_negative_pool(
        train_edge_index_cpu
    )

    paper_encoder = HVGAEPaperEncoder(
        in_channels=x_p.size(-1),
        hidden_channels=args.hidden,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    dataset_encoder = HVGAEDatasetEncoder(
        num_pub=num_pub,
        hidden_channels=args.hidden,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
    ).to(device)

    decoder = InnerProductScorer().to(
        device
    )

    optimizer = torch.optim.AdamW(
        list(
            paper_encoder.parameters()
        )
        +
        list(
            dataset_encoder.parameters()
        ),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    eval_splits = get_eval_splits(data)

    best_ndcg = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        paper_encoder.train()
        dataset_encoder.train()

        optimizer.zero_grad()

        mu_p, var_p = paper_encoder(
            x_p,
            citation_edge_index,
        )

        mu_d, var_d = dataset_encoder(
            H_T
        )

        z_p = reparameterize_gaussian(
            mu_p,
            var_p,
            True,
        )

        z_d = reparameterize_gaussian(
            mu_d,
            var_d,
            True,
        )

        # Citation graph
        pos_scores_a = decoder(
            z_p[
                citation_edge_index[0]
            ],
            z_p[
                citation_edge_index[1]
            ],
        )

        neg_edge_a = sample_negative_citations(
            citation_edge_index.cpu(),
            num_pub,
            args.num_neg,
        ).to(device)

        neg_scores_a = decoder(
            z_p[
                neg_edge_a[0]
            ],
            z_p[
                neg_edge_a[1]
            ],
        )

        scores_a = torch.cat(
            [
                pos_scores_a,
                neg_scores_a,
            ]
        )

        labels_a = torch.cat(
            [
                torch.ones_like(
                    pos_scores_a
                ),
                torch.zeros_like(
                    neg_scores_a
                ),
            ]
        )

        recon_loss_a = (
            F.binary_cross_entropy_with_logits(
                scores_a,
                labels_a,
            )
        )

        # Publication -> Dataset
        pos_scores_h = decoder(
            z_p[
                train_edge_index[0]
            ],
            z_d[
                train_edge_index[1]
            ],
        )

        neg_edge_h = sample_filtered_negatives(
            train_edge_index_cpu,
            num_dataset,
            args.num_neg,
            positive_sets,
            negative_pool,
        ).to(device)

        neg_scores_h = decoder(
            z_p[
                neg_edge_h[0]
            ],
            z_d[
                neg_edge_h[1]
            ],
        )

        scores_h = torch.cat(
            [
                pos_scores_h,
                neg_scores_h,
            ]
        )

        labels_h = torch.cat(
            [
                torch.ones_like(
                    pos_scores_h
                ),
                torch.zeros_like(
                    neg_scores_h
                ),
            ]
        )

        recon_loss_h = (
            F.binary_cross_entropy_with_logits(
                scores_h,
                labels_h,
            )
        )

        kl_p = kl_to_standard_normal(
            mu_p,
            var_p,
        )

        kl_d = kl_to_standard_normal(
            mu_d,
            var_d,
        )

        warmup = min(
            1.0,
            epoch
            /
            max(
                1,
                args.kl_warmup_epochs,
            ),
        )

        beta = (
            args.kl_weight
            * warmup
        )

        loss = (
            recon_loss_a
            + recon_loss_h
            + beta * (kl_p + kl_d)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(
                paper_encoder.parameters()
            )
            +
            list(
                dataset_encoder.parameters()
            ),
            5.0,
        )

        optimizer.step()

        if (
            epoch % args.eval_every == 0
            or epoch == args.epochs
        ):

            paper_encoder.eval()
            dataset_encoder.eval()

            with torch.no_grad():

                mu_p_eval, var_p_eval = (
                    paper_encoder(
                        x_p,
                        citation_edge_index,
                    )
                )

                mu_d_eval, var_d_eval = (
                    dataset_encoder(
                        H_T
                    )
                )

                def score_fn(
                    pub_id
                ):

                    return kl_query_dataset(
                        mu_p_eval[pub_id],
                        var_p_eval[pub_id],
                        mu_d_eval,
                        var_d_eval,
                    )

                line = (
                    f"[hvgae] "
                    f"epoch={epoch:03d} "
                    f"loss={loss.item():.5f} "
                    f"reconA={recon_loss_a.item():.5f} "
                    f"reconH={recon_loss_h.item():.5f} "
                    f"klP={kl_p.item():.5f} "
                    f"klD={kl_d.item():.5f} "
                    f"beta={beta:.6f}"
                )

                if "validation_trans" in eval_splits:

                    val = evaluate_ranking(
                        score_fn,
                        eval_splits[
                            "validation_trans"
                        ],
                        k_list=(5, 10),
                        exclude_pairs=positive_sets,
                    )

                    line += (
                        f" | MRR@10={val['MRR@10']:.4f}"
                        f" NDCG@10={val['NDCG@10']:.4f}"
                        f" Recall@10={val['Recall@10']:.4f}"
                    )

                    if (
                        val["NDCG@10"]
                        > best_ndcg
                    ):

                        best_ndcg = val[
                            "NDCG@10"
                        ]

                        best_state = {
                            "paper_encoder": {
                                k: v.detach()
                                .cpu()
                                .clone()
                                for k, v
                                in paper_encoder.state_dict().items()
                            },
                            "dataset_encoder": {
                                k: v.detach()
                                .cpu()
                                .clone()
                                for k, v
                                in dataset_encoder.state_dict().items()
                            },
                        }

                        patience_counter = 0

                    else:

                        patience_counter += 1

                print(line)

                if (
                    args.patience > 0
                    and patience_counter
                    >= args.patience
                ):
                    break

    if best_state is not None:

        paper_encoder.load_state_dict(
            best_state[
                "paper_encoder"
            ]
        )

        dataset_encoder.load_state_dict(
            best_state[
                "dataset_encoder"
            ]
        )

    paper_encoder.eval()
    dataset_encoder.eval()

    final_results = {}

    with torch.no_grad():

        mu_p_eval, var_p_eval = (
            paper_encoder(
                x_p,
                citation_edge_index,
            )
        )

        mu_d_eval, var_d_eval = (
            dataset_encoder(
                H_T
            )
        )

        def score_fn(
            pub_id
        ):

            return kl_query_dataset(
                mu_p_eval[pub_id],
                var_p_eval[pub_id],
                mu_d_eval,
                var_d_eval,
            )

        print(
            "\n=== FINAL hvgae ==="
        )

        for split in [
            "validation_trans",
            "test_trans",
            "test_semi",
            "test_ind",
        ]:

            if split not in eval_splits:
                continue

            result = evaluate_ranking(
                score_fn,
                eval_splits[split],
                k_list=(5, 10),
                exclude_pairs=positive_sets,
            )

            final_results[
                split
            ] = result

            print(
                f"[{split}] {result}"
            )

    return (
        final_results,
        (
            paper_encoder,
            dataset_encoder,
        ),
        decoder,
    )


# ============================================================================
# STATIC TRAINING
# RGCN / KGAT
# ============================================================================

def train_static_model(
    data,
    args,
    graph,
    device,
):

    node_types = graph["node_types"]
    offsets = graph["offsets"]
    total_nodes = graph["total_nodes"]

    if args.model == "rgcn":

        model = RGCNEncoder(
            node_types=node_types,
            in_channels_dict=graph[
                "in_channels_dict"
            ],
            hidden_channels=args.hidden,
            num_relations=graph[
                "num_relations"
            ],
            num_bases=args.num_bases,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)

        out_dim = args.hidden

    else:

        model = KGATEncoder(
            node_types=node_types,
            in_channels_dict=graph[
                "in_channels_dict"
            ],
            hidden_channels=args.hidden,
            num_relations=graph[
                "num_relations"
            ],
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)

        out_dim = model.out_dim

    scorer = BilinearScorer(
        out_dim
    ).to(device)

    x_dict = {
        nt: x.to(device)
        for nt, x in graph[
            "x_dict"
        ].items()
    }

    edge_index = graph[
        "edge_index"
    ].to(device)

    edge_type = graph[
        "edge_type"
    ].to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters())
        +
        list(scorer.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    pub_off = offsets[
        "publication"
    ]

    ds_off = offsets[
        "dataset"
    ]

    num_pub = data[
        "publication"
    ].num_nodes

    num_dataset = data[
        "dataset"
    ].num_nodes

    train_edge_index_cpu = data[
        "publication",
        "cites",
        "dataset",
    ].edge_index_train.cpu()

    train_edge_index = (
        train_edge_index_cpu.to(device)
    )

    positive_sets = build_positive_sets(
        train_edge_index_cpu
    )

    negative_pool = build_negative_pool(
        train_edge_index_cpu
    )

    eval_splits = get_eval_splits(
        data
    )

    best_ndcg = -1.0
    best_state = None
    patience_counter = 0

    model_params = (
        list(model.parameters())
        +
        list(scorer.parameters())
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()
        scorer.train()

        optimizer.zero_grad()

        if args.model == "kgat":

            embeddings = model(
                x_dict,
                edge_index,
                edge_type,
                total_nodes,
            )

        else:

            embeddings = model(
                x_dict,
                edge_index,
                edge_type,
            )

        pub_emb = embeddings[
            pub_off:
            pub_off + num_pub
        ]

        ds_emb = embeddings[
            ds_off:
            ds_off + num_dataset
        ]

        pos_scores = scorer(
            pub_emb[
                train_edge_index[0]
            ],
            ds_emb[
                train_edge_index[1]
            ],
        )

        neg_edge_index = (
            sample_filtered_negatives(
                train_edge_index_cpu,
                num_dataset,
                args.num_neg,
                positive_sets,
                negative_pool,
            ).to(device)
        )

        neg_scores = scorer(
            pub_emb[
                neg_edge_index[0]
            ],
            ds_emb[
                neg_edge_index[1]
            ],
        )

        pos_expanded = (
            pos_scores.repeat_interleave(
                args.num_neg
            )
        )

        loss = bpr_loss(
            pos_expanded,
            neg_scores,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model_params,
            5.0,
        )

        optimizer.step()

        if (
            epoch % args.eval_every == 0
            or epoch == args.epochs
        ):

            model.eval()
            scorer.eval()

            with torch.no_grad():

                if args.model == "kgat":

                    embeddings = model(
                        x_dict,
                        edge_index,
                        edge_type,
                        total_nodes,
                    )

                else:

                    embeddings = model(
                        x_dict,
                        edge_index,
                        edge_type,
                    )

                pub_emb = embeddings[
                    pub_off:
                    pub_off + num_pub
                ]

                ds_emb = embeddings[
                    ds_off:
                    ds_off + num_dataset
                ]

                def score_fn(
                    pub_id
                ):

                    return scorer.score_all(
                        pub_emb[pub_id],
                        ds_emb,
                    )

                line = (
                    f"[{args.model}] "
                    f"epoch={epoch:03d} "
                    f"loss={loss.item():.5f}"
                )

                if "validation_trans" in eval_splits:

                    val = evaluate_ranking(
                        score_fn,
                        eval_splits[
                            "validation_trans"
                        ],
                        k_list=(5, 10),
                        exclude_pairs=positive_sets,
                    )

                    line += (
                        f" | MRR@10={val['MRR@10']:.4f}"
                        f" NDCG@10={val['NDCG@10']:.4f}"
                        f" Recall@10={val['Recall@10']:.4f}"
                    )

                    if (
                        val["NDCG@10"]
                        > best_ndcg
                    ):

                        best_ndcg = val[
                            "NDCG@10"
                        ]

                        best_state = {
                            "model": {
                                k: v.detach()
                                .cpu()
                                .clone()
                                for k, v
                                in model.state_dict().items()
                            },
                            "scorer": {
                                k: v.detach()
                                .cpu()
                                .clone()
                                for k, v
                                in scorer.state_dict().items()
                            },
                        }

                        patience_counter = 0

                    else:

                        patience_counter += 1

                print(line)

                if (
                    args.patience > 0
                    and patience_counter
                    >= args.patience
                ):
                    break

    if best_state is not None:

        model.load_state_dict(
            best_state["model"]
        )

        scorer.load_state_dict(
            best_state["scorer"]
        )

    model.eval()
    scorer.eval()

    with torch.no_grad():

        if args.model == "kgat":

            embeddings = model(
                x_dict,
                edge_index,
                edge_type,
                total_nodes,
            )

        else:

            embeddings = model(
                x_dict,
                edge_index,
                edge_type,
            )

        pub_emb = embeddings[
            pub_off:
            pub_off + num_pub
        ]

        ds_emb = embeddings[
            ds_off:
            ds_off + num_dataset
        ]

    final_results = {}

    def score_fn(
        pub_id
    ):

        return scorer.score_all(
            pub_emb[pub_id],
            ds_emb,
        )

    print(
        f"\n=== FINAL {args.model} ==="
    )

    for split in [
        "validation_trans",
        "test_trans",
        "test_semi",
        "test_ind",
    ]:

        if split not in eval_splits:
            continue

        result = evaluate_ranking(
            score_fn,
            eval_splits[split],
            k_list=(5, 10),
            exclude_pairs=positive_sets,
        )

        final_results[
            split
        ] = result

        print(
            f"[{split}] {result}"
        )

    return (
        final_results,
        model,
        scorer,
    )


# ============================================================================
# KGCN TRAINING
# ============================================================================

def train_kgcn(
    data,
    args,
    graph,
    device,
):

    offsets = graph["offsets"]
    total_nodes = graph["total_nodes"]

    node_types = graph["node_types"]

    model = KGCNEncoder(
        node_types=node_types,
        in_channels_dict=graph[
            "in_channels_dict"
        ],
        hidden_channels=args.hidden,
        num_relations=graph[
            "num_relations"
        ],
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    x_dict = {
        nt: x.to(device)
        for nt, x in graph[
            "x_dict"
        ].items()
    }

    edge_index = graph[
        "edge_index"
    ].to(device)

    edge_type = graph[
        "edge_type"
    ].to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    pub_off = offsets[
        "publication"
    ]

    ds_off = offsets[
        "dataset"
    ]

    num_pub = data[
        "publication"
    ].num_nodes

    num_dataset = data[
        "dataset"
    ].num_nodes

    train_edge_index_cpu = data[
        "publication",
        "cites",
        "dataset",
    ].edge_index_train.cpu()

    train_pairs = build_positive_sets(
        train_edge_index_cpu
    )

    train_pubs = list(
        train_pairs.keys()
    )

    negative_pool = build_negative_pool(
        train_edge_index_cpu
    )

    eval_splits = get_eval_splits(
        data
    )

    def score_for_pub(
        pub_id,
        base_x,
    ):

        query_emb = base_x[
            pub_off + pub_id
        ]

        conditioned = (
            model.conditioned_embeddings(
                base_x,
                edge_index,
                edge_type,
                total_nodes,
                query_emb,
            )
        )

        ds_emb = conditioned[
            ds_off:
            ds_off + num_dataset
        ]

        return (
            ds_emb @ query_emb
        )

    best_ndcg = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()
        optimizer.zero_grad()

        base_x = model.base_embeddings(
            x_dict
        )

        batch_pubs = random.sample(
            train_pubs,
            min(
                args.kgcn_batch_pubs,
                len(train_pubs),
            ),
        )

        losses = []

        for pub_id in batch_pubs:

            positives = list(
                train_pairs[
                    pub_id
                ]
            )

            query_emb = base_x[
                pub_off + pub_id
            ]

            conditioned = (
                model.conditioned_embeddings(
                    base_x,
                    edge_index,
                    edge_type,
                    total_nodes,
                    query_emb,
                )
            )

            ds_emb = conditioned[
                ds_off:
                ds_off + num_dataset
            ]

            pos_idx = torch.tensor(
                positives,
                dtype=torch.long,
                device=device,
            )

            local_edge = torch.stack(
                [
                    torch.full(
                        (len(positives),),
                        pub_id,
                        dtype=torch.long,
                    ),
                    torch.tensor(
                        positives,
                        dtype=torch.long,
                    ),
                ],
                dim=0,
            )

            neg_edge = (
                sample_filtered_negatives(
                    local_edge,
                    num_dataset,
                    args.num_neg,
                    train_pairs,
                    negative_pool,
                ).to(device)
            )

            pos_scores = (
                ds_emb[pos_idx]
                @ query_emb
            )

            neg_scores = (
                ds_emb[
                    neg_edge[1]
                ]
                @ query_emb
            )

            losses.append(
                bpr_loss(
                    pos_scores.repeat_interleave(
                        args.num_neg
                    ),
                    neg_scores,
                )
            )

        loss = torch.stack(
            losses
        ).mean()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )

        optimizer.step()

        if (
            epoch % args.eval_every == 0
            or epoch == args.epochs
        ):

            model.eval()

            with torch.no_grad():

                base_x = model.base_embeddings(
                    x_dict
                )

                line = (
                    f"[kgcn] "
                    f"epoch={epoch:03d} "
                    f"loss={loss.item():.5f}"
                )

                if "validation_trans" in eval_splits:

                    validation = eval_splits[
                        "validation_trans"
                    ]

                    if (
                        len(validation)
                        > args.kgcn_val_subsample
                    ):

                        selected = random.sample(
                            list(
                                validation.keys()
                            ),
                            args.kgcn_val_subsample,
                        )

                        validation = {
                            p: validation[p]
                            for p in selected
                        }

                    def score_fn(
                        pub_id
                    ):

                        return score_for_pub(
                            pub_id,
                            base_x,
                        )

                    val = evaluate_ranking(
                        score_fn,
                        validation,
                        k_list=(5, 10),
                        exclude_pairs=train_pairs,
                    )

                    line += (
                        f" | MRR@10={val['MRR@10']:.4f}"
                        f" NDCG@10={val['NDCG@10']:.4f}"
                        f" Recall@10={val['Recall@10']:.4f}"
                    )

                    if (
                        val["NDCG@10"]
                        > best_ndcg
                    ):

                        best_ndcg = val[
                            "NDCG@10"
                        ]

                        best_state = {
                            k: v.detach()
                            .cpu()
                            .clone()
                            for k, v
                            in model.state_dict().items()
                        }

                        patience_counter = 0

                    else:

                        patience_counter += 1

                print(line)

                if (
                    args.patience > 0
                    and patience_counter
                    >= args.patience
                ):
                    break

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    model.eval()

    final_results = {}

    with torch.no_grad():

        base_x = model.base_embeddings(
            x_dict
        )

        def score_fn(
            pub_id
        ):
            return score_for_pub(
                pub_id,
                base_x,
            )

        print(
            "\n=== FINAL kgcn ==="
        )

        for split in [
            "validation_trans",
            "test_trans",
            "test_semi",
            "test_ind",
        ]:

            if split not in eval_splits:
                continue

            result = evaluate_ranking(
                score_fn,
                eval_splits[split],
                k_list=(5, 10),
                exclude_pairs=train_pairs,
            )

            final_results[
                split
            ] = result

            print(
                f"[{split}] {result}"
            )

    return (
        final_results,
        model,
        None,
    )


# ============================================================================
# DATA
# ============================================================================

def load_data(
    dataset,
    split=0,
):

    ds = loader.ScholarlyDataset(
        root=f"datasets/{dataset}/"
    )

    data = ds[0]

    if split > 0:

        num_dataset = data[
            "dataset"
        ].num_nodes

        feat_dim = data[
            "dataset"
        ].x.size(-1)

        n_sampled = int(
            split / 100.0
            * num_dataset
        )

        sampled_indices = torch.randperm(
            num_dataset
        )[:n_sampled]

        data[
            "dataset"
        ].x[
            sampled_indices
        ] = torch.ones(
            n_sampled,
            feat_dim,
            dtype=data[
                "dataset"
            ].x.dtype,
        )

        print(
            f"[load_data] "
            f"split={split}% "
            f"-> {n_sampled}/{num_dataset}"
        )

    return data


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DEFAULTS = dict(

    hidden=128,

    latent_dim=16,

    num_layers=2,

    dropout=0.10,

    lr=3e-4,

    weight_decay=1e-5,

    num_neg=5,

    epochs=150,

    eval_every=5,

    patience=20,

    num_bases=16,

    kl_weight=0.001,

    kl_warmup_epochs=20,

    kgcn_batch_pubs=8,

    kgcn_val_subsample=200,

    split=0,

    cpu=False,

    dataset="mes",

    root=None,
)


DATASET_DEFAULTS = {

    "mes": {
        "epochs": 150,
        "eval_every": 5,
        "patience": 20,
    },

    "pubmed": {
        "epochs": 100,
        "eval_every": 5,
        "patience": 10,
    },
}


MODEL_DEFAULTS = {

    "rgcn": {
        "hidden": 128,
        "num_layers": 2,
        "dropout": 0.10,
        "lr": 3e-4,
        "num_bases": 16,
    },

    "kgat": {
        "hidden": 64,
        "num_layers": 3,
        "dropout": 0.10,
        "lr": 3e-4,
    },

    "kgcn": {
        "hidden": 64,
        "num_layers": 2,
        "dropout": 0.10,
        "lr": 3e-4,
        "kgcn_batch_pubs": 8,
        "kgcn_val_subsample": 200,
    },

    "vgae": {
        "hidden": 128,
        "num_layers": 2,
        "dropout": 0.10,
        "lr": 1e-3,
        "num_bases": 16,
        "kl_weight": 0.001,
        "kl_warmup_epochs": 20,
    },

    "hvgae": {
        "hidden": 128,
        "num_layers": 2,
        "dropout": 0.10,
        "lr": 2e-3,
        "latent_dim": 16,
        "kl_weight": 0.001,
        "kl_warmup_epochs": 30,
    },
}


DATASET_MODEL_DEFAULTS = {

    ("pubmed", "kgcn"): {
        "hidden": 32,
        "kgcn_batch_pubs": 8,
        "kgcn_val_subsample": 300,
    },

    ("pubmed", "kgat"): {
        "hidden": 32,
    },

    ("pubmed", "hvgae"): {
        "hidden": 64,
        "latent_dim": 16,
    },

    ("pubmed", "vgae"): {
        "hidden": 64,
    },
}


def load_params(
    model,
    dataset,
    cli_args,
):

    params = dict(
        BASE_DEFAULTS
    )

    params.update(
        DATASET_DEFAULTS.get(
            dataset,
            {},
        )
    )

    params.update(
        MODEL_DEFAULTS.get(
            model,
            {},
        )
    )

    params.update(
        DATASET_MODEL_DEFAULTS.get(
            (
                dataset,
                model,
            ),
            {},
        )
    )

    # CLI overrides only when explicitly supplied.
    for key, value in vars(
        cli_args
    ).items():

        if value is not None:

            params[key] = value

    params["model"] = model
    params["dataset"] = dataset

    if params.get("root") is None:

        params["root"] = (
            f"datasets/{dataset}"
        )

    args = argparse.Namespace(
        **params
    )

    print(
        f"[config] dataset={dataset} "
        f"model={model}"
    )

    print(
        f"[config] hidden={args.hidden} "
        f"layers={args.num_layers} "
        f"dropout={args.dropout}"
    )

    print(
        f"[config] lr={args.lr} "
        f"wd={args.weight_decay} "
        f"neg={args.num_neg}"
    )

    print(
        f"[config] epochs={args.epochs} "
        f"eval_every={args.eval_every}"
    )

    if model == "rgcn":

        print(
            f"[config] "
            f"num_bases={args.num_bases}"
        )

    if model == "kgcn":

        print(
            f"[config] "
            f"batch_pubs={args.kgcn_batch_pubs} "
            f"val_subsample={args.kgcn_val_subsample}"
        )

    if model in (
        "vgae",
        "hvgae",
    ):

        print(
            f"[config] "
            f"kl_weight={args.kl_weight} "
            f"kl_warmup={args.kl_warmup_epochs}"
        )

    if model == "hvgae":

        print(
            f"[config] "
            f"latent_dim={args.latent_dim}"
        )

    return args


# ============================================================================
# DISPATCH
# ============================================================================

def train_and_evaluate(
    data,
    args,
):

    device = torch.device(
        "cuda"
        if (
            torch.cuda.is_available()
            and not args.cpu
        )
        else "cpu"
    )

    print(
        f"\nDevice: {device} | "
        f"Modello: {args.model}"
    )

    # HVGAE does NOT use the homogeneous KG.
    if args.model == "hvgae":

        return train_hvgae(
            data,
            args,
            device,
        )

    graph = build_homogeneous_graph(
        data
    )

    if args.model in (
        "rgcn",
        "kgat",
    ):

        return train_static_model(
            data,
            args,
            graph,
            device,
        )

    if args.model == "kgcn":

        return train_kgcn(
            data,
            args,
            graph,
            device,
        )

    if args.model == "vgae":

        return train_vgae(
            data,
            args,
            graph,
            device,
        )

    raise ValueError(
        f"Modello non supportato: "
        f"{args.model}"
    )


# ============================================================================
# CLI
# ============================================================================

def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset",
        "-dataset",
        type=str,
        default="mes",
    )

    p.add_argument(
        "--split",
        "-split",
        type=float,
        default=0,
    )

    p.add_argument(
        "--model",
        "-model",
        type=str,
        default=None,
        choices=[
            "rgcn",
            "kgat",
            "kgcn",
            "vgae",
            "hvgae",
        ],
    )

    p.add_argument(
        "--hidden",
        "-hidden",
        type=int,
        default=None,
    )

    p.add_argument(
        "--latent_dim",
        "-latent_dim",
        type=int,
        default=None,
    )

    p.add_argument(
        "--num_layers",
        "-num_layers",
        type=int,
        default=None,
    )

    p.add_argument(
        "--dropout",
        "-dropout",
        type=float,
        default=None,
    )

    p.add_argument(
        "--lr",
        "-lr",
        type=float,
        default=None,
    )

    p.add_argument(
        "--weight_decay",
        "-weight_decay",
        type=float,
        default=None,
    )

    p.add_argument(
        "--num_neg",
        "-num_neg",
        type=int,
        default=None,
    )

    p.add_argument(
        "--epochs",
        "-epochs",
        type=int,
        default=None,
    )

    p.add_argument(
        "--eval_every",
        "-eval_every",
        type=int,
        default=None,
    )

    p.add_argument(
        "--patience",
        "-patience",
        type=int,
        default=None,
    )

    p.add_argument(
        "--num_bases",
        "-num_bases",
        type=int,
        default=None,
    )

    p.add_argument(
        "--kl_weight",
        "-kl_weight",
        type=float,
        default=None,
    )

    p.add_argument(
        "--kl_warmup_epochs",
        "-kl_warmup_epochs",
        type=int,
        default=None,
    )

    p.add_argument(
        "--kgcn_batch_pubs",
        "-kgcn_batch_pubs",
        type=int,
        default=None,
    )

    p.add_argument(
        "--kgcn_val_subsample",
        "-kgcn_val_subsample",
        type=int,
        default=None,
    )

    p.add_argument(
        "--root",
        "-root",
        type=str,
        default=None,
    )

    p.add_argument(
        "--cpu",
        "-cpu",
        action="store_true",
    )

    return p.parse_args()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    base_args = parse_args()

    set_seed(SEED)

    methods = (
        [base_args.model]
        if base_args.model is not None
        else [
            "vgae",
            "hvgae",
            "kgat",
            "kgcn",
            "rgcn",


        ]
    )

    for method in methods:

        print()
        print(
            "=" * 60
        )
        print(
            f" DATASET: {base_args.dataset}"
        )
        print(
            f" MODEL:   {method}"
        )
        print(
            "=" * 60
        )

        args = load_params(
            method,
            base_args.dataset,
            base_args,
        )

        data = load_data(
            args.dataset,
            split=args.split,
        )

        results, model, scorer = (
            train_and_evaluate(
                data,
                args,
            )
        )

        print(
            f"\n[{method}] training terminato."
        )