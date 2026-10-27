"""
Utilities to turn the HeteroData produced by dataset.py (ScholarlyDataset)
into a single knowledge-graph representation:

    - a global node id space (offset per node type)
    - a global relation id space (one id per edge type in HeteroData,
      plus an explicit inverse relation for every edge type, since KGCN/KGAT
      both assume message passing flows in both directions)
    - a triples tensor of shape [3, num_triples] = (head, relation, tail)

This lets KGCN / KGAT operate on the graph as a normal multi-relational KG,
independently of how many node/edge types the heterogeneous schema has.

The target link-prediction relation is always ('publication', 'cites', 'dataset'),
consistent with the rest of the pipeline (pd_edge_index_train / *_test_* / *_validation_*).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import torch


TARGET_EDGE_TYPE = ('publication', 'cites', 'dataset')


@dataclass
class KGSchema:
    node_type_offset: Dict[str, int]
    node_type_count: Dict[str, int]
    total_nodes: int
    relation2id: Dict[Tuple[str, str, str], int]
    id2relation: Dict[int, Tuple[str, str, str]]
    num_relations: int  # already doubled to include inverses


def build_node_offsets(data) -> Tuple[Dict[str, int], Dict[str, int], int]:
    """Assign every node type a contiguous block in a single global id space."""
    offset = {}
    count = {}
    running = 0
    # deterministic order: sort node types alphabetically so runs are reproducible
    for node_type in sorted(data.node_types):
        n = data[node_type].num_nodes
        offset[node_type] = running
        count[node_type] = n
        running += n
    return offset, count, running


def build_relation_index(data) -> Tuple[Dict[Tuple[str, str, str], int], Dict[int, Tuple[str, str, str]], int]:
    """
    Assign an id to every edge_type present in HeteroData, plus its inverse.
    Only edge_types that actually carry an `edge_index` are used (this
    excludes the multiple *_train/_test/_validation_* variants of the
    publication-dataset link, which are handled separately as the
    supervision/target relation and its own message-passing edge).
    """
    relation2id = {}
    id2relation = {}
    next_id = 0

    aux_edge_types = [et for et in data.edge_types if et != TARGET_EDGE_TYPE]

    for et in sorted(aux_edge_types):
        relation2id[et] = next_id
        id2relation[next_id] = et
        next_id += 1
        inv_et = (et[2], f'inv_{et[1]}', et[0])
        relation2id[inv_et] = next_id
        id2relation[next_id] = inv_et
        next_id += 1

    # target relation (message passing side only, i.e. the *_train edges used
    # as observed graph structure) gets its own id + inverse too
    relation2id[TARGET_EDGE_TYPE] = next_id
    id2relation[next_id] = TARGET_EDGE_TYPE
    next_id += 1
    inv_target = (TARGET_EDGE_TYPE[2], f'inv_{TARGET_EDGE_TYPE[1]}', TARGET_EDGE_TYPE[0])
    relation2id[inv_target] = next_id
    id2relation[next_id] = inv_target
    next_id += 1

    return relation2id, id2relation, next_id


def build_kg_schema(data) -> KGSchema:
    offset, count, total = build_node_offsets(data)
    rel2id, id2rel, num_rel = build_relation_index(data)
    return KGSchema(
        node_type_offset=offset,
        node_type_count=count,
        total_nodes=total,
        relation2id=rel2id,
        id2relation=id2rel,
        num_relations=num_rel,
    )


def _to_global(edge_index: torch.Tensor, src_type: str, dst_type: str, schema: KGSchema) -> torch.Tensor:
    src_off = schema.node_type_offset[src_type]
    dst_off = schema.node_type_offset[dst_type]
    out = edge_index.clone()
    out[0] = out[0] + src_off
    out[1] = out[1] + dst_off
    return out


def build_triples(data, schema: KGSchema, message_passing_edge_index_train=None) -> torch.Tensor:
    """
    Build the full triples tensor [3, N] = (head_global, relation, tail_global)
    used as the KG structure for embedding propagation (KGCN / KGAT).

    All auxiliary edge types (hasauthor, hastopic, hasentity, cites (pub-pub),
    cites (data-data), hasvenue, haskeyword, hasorganization, ...) are added
    in both directions. The target relation publication-cites-dataset is added
    using ONLY the training split (message_passing_edge_index_train, i.e.
    data['publication','cites','dataset'].edge_index_train) to avoid leaking
    validation/test edges into the propagation structure.
    """
    triples = []

    aux_edge_types = [et for et in data.edge_types if et != TARGET_EDGE_TYPE]
    for et in aux_edge_types:
        src_type, rel_name, dst_type = et
        ei = data[et].edge_index
        if ei is None or ei.numel() == 0:
            continue
        rel_id = schema.relation2id[et]
        inv_rel_id = schema.relation2id[(dst_type, f'inv_{rel_name}', src_type)]

        g_ei = _to_global(ei, src_type, dst_type, schema)
        rel_col = torch.full((g_ei.size(1),), rel_id, dtype=torch.long)
        triples.append(torch.stack([g_ei[0], rel_col, g_ei[1]], dim=0))

        inv_rel_col = torch.full((g_ei.size(1),), inv_rel_id, dtype=torch.long)
        triples.append(torch.stack([g_ei[1], inv_rel_col, g_ei[0]], dim=0))

    # target relation: message-passing (train) edges only
    if message_passing_edge_index_train is None:
        message_passing_edge_index_train = data[TARGET_EDGE_TYPE].edge_index_train

    src_type, rel_name, dst_type = TARGET_EDGE_TYPE
    rel_id = schema.relation2id[TARGET_EDGE_TYPE]
    inv_rel_id = schema.relation2id[(dst_type, f'inv_{rel_name}', src_type)]

    g_ei = _to_global(message_passing_edge_index_train, src_type, dst_type, schema)
    rel_col = torch.full((g_ei.size(1),), rel_id, dtype=torch.long)
    triples.append(torch.stack([g_ei[0], rel_col, g_ei[1]], dim=0))

    inv_rel_col = torch.full((g_ei.size(1),), inv_rel_id, dtype=torch.long)
    triples.append(torch.stack([g_ei[1], inv_rel_col, g_ei[0]], dim=0))

    return torch.cat(triples, dim=1)


def global_ids_for_split(data, schema: KGSchema, split: str):
    """
    Returns (pos_edge_index_global, neg_edge_index_global) for a given split,
    where split in {'train', 'validation_trans', 'test_trans',
                     'test_semi', 'test_ind'} matching the attribute
    naming convention used in dataset.py:
        edge_label_index_{split}   and   negative_edge_label_index_{split}
    For split == 'train' the attribute is simply
        edge_label_index_train / (no negatives precomputed -> sampled on the fly)
    """
    src_type, _, dst_type = TARGET_EDGE_TYPE
    src_off = schema.node_type_offset[src_type]
    dst_off = schema.node_type_offset[dst_type]

    pos_attr = f'edge_label_index_{split}'
    neg_attr = f'negative_edge_label_index_{split}'

    pos = getattr(data[TARGET_EDGE_TYPE], pos_attr)
    pos_g = pos.clone()
    pos_g[0] += src_off
    pos_g[1] += dst_off

    neg_g = None
    if hasattr(data[TARGET_EDGE_TYPE], neg_attr):
        neg = getattr(data[TARGET_EDGE_TYPE], neg_attr)
        neg_g = neg.clone()
        neg_g[0] += src_off
        neg_g[1] += dst_off

    return pos_g, neg_g


def initial_node_features(data, schema: KGSchema, feature_kind: str = 'net_x') -> torch.Tensor:
    """
    Stack per-type features (net_x = node2vec embeddings, or x = content/text
    embeddings where available) into one [total_nodes, dim] tensor in the
    global id ordering. Node types lacking a given feature_kind fall back to
    the other one; if neither exists, random init is used (rare: only happens
    if a node type has no textual/keyword field AND node2vec failed).
    """
    dims = []
    per_type_feats = {}
    for nt in sorted(data.node_types):
        feat = getattr(data[nt], feature_kind, None)
        if feat is None:
            feat = getattr(data[nt], 'x' if feature_kind == 'net_x' else 'net_x', None)
        if feat is None:
            feat = torch.randn(schema.node_type_count[nt], 128)
        per_type_feats[nt] = feat.float()
        dims.append(feat.size(-1))

    # if dims differ across types, project everything to the min dim via a
    # simple learned-free truncation/pad (keeps this utility dependency-free;
    # a Linear projection per type is applied in the model itself instead)
    max_dim = max(dims)
    total = schema.total_nodes
    out = torch.zeros(total, max_dim)
    for nt in sorted(data.node_types):
        off = schema.node_type_offset[nt]
        n = schema.node_type_count[nt]
        feat = per_type_feats[nt]
        d = feat.size(-1)
        out[off:off + n, :d] = feat
    return out
