"""
KGCN -- Knowledge Graph Convolutional Network
(Wang et al., "Knowledge Graph Convolutional Networks for Recommender
Systems", WWW 2019).

Core idea: for the "item" side of a user-item recommendation (here: the
"dataset" node is the item, "publication" is the user/query side, following
the publication-cites-dataset target relation), for every candidate item
compute a receptive field by sampling its KG neighbors, weight each neighbor
by an attention score conditioned on the *query* embedding (the publication
in our case) and the relation embedding, aggregate, and combine with the
item's own embedding through K hops (here we use `num_hops` GNN layers).

Since our task is (publication, dataset) link prediction rather than
(user, item) recommendation, the "query" vector is the publication embedding,
and KGCN neighbor aggregation is applied on the dataset side, propagating
through the full multi-relational KG built in kg_utils.py.

This implementation samples a fixed-size neighborhood per node per hop
(as in the original paper) rather than using the full neighborhood, for
scalability on large heterogeneous graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Aggregator(nn.Module):
    """Aggregates entity's neighborhood representation into its own
    embedding, following one of KGCN's three aggregator variants."""

    def __init__(self, dim, aggregator_type='sum'):
        super().__init__()
        self.aggregator_type = aggregator_type
        if aggregator_type == 'concat':
            self.linear = nn.Linear(dim * 2, dim)
        else:
            self.linear = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, self_vectors, neighbor_vectors):
        if self.aggregator_type == 'sum':
            out = self_vectors + neighbor_vectors
            out = self.linear(out)
        elif self.aggregator_type == 'concat':
            out = torch.cat([self_vectors, neighbor_vectors], dim=-1)
            out = self.linear(out)
        elif self.aggregator_type == 'neighbor':
            out = self.linear(neighbor_vectors)
        else:
            raise ValueError(f'Unknown aggregator_type {self.aggregator_type}')
        return self.act(out)


class KGCN(nn.Module):
    def __init__(self, num_nodes, num_relations, in_dim, embed_dim=64,
                 num_hops=2, neighbor_sample_size=8, aggregator_type='sum',
                 dropout=0.1):
        super().__init__()
        self.num_hops = num_hops
        self.neighbor_sample_size = neighbor_sample_size
        self.embed_dim = embed_dim

        self.input_proj = nn.Linear(in_dim, embed_dim)
        self.relation_emb = nn.Embedding(num_relations, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.aggregators = nn.ModuleList([
            Aggregator(embed_dim, aggregator_type) for _ in range(num_hops)
        ])

        # neighbor adjacency built lazily via set_graph()
        self.adj_entity = None   # [num_nodes, neighbor_sample_size] long
        self.adj_relation = None  # [num_nodes, neighbor_sample_size] long
        self.num_nodes = num_nodes

    def set_graph(self, triples: torch.Tensor):
        """
        triples: [3, N] = (head, relation, tail). Builds a fixed-size,
        randomly-sampled-with-replacement neighbor table per node (as in the
        original KGCN paper's `_construct_adj`).
        """
        device = triples.device
        head, rel, tail = triples[0], triples[1], triples[2]

        neighbor_lists = [[] for _ in range(self.num_nodes)]
        for h, r, t in zip(head.tolist(), rel.tolist(), tail.tolist()):
            neighbor_lists[h].append((r, t))

        S = self.neighbor_sample_size
        adj_entity = torch.zeros(self.num_nodes, S, dtype=torch.long)
        adj_relation = torch.zeros(self.num_nodes, S, dtype=torch.long)

        for node in range(self.num_nodes):
            neighbors = neighbor_lists[node]
            if len(neighbors) == 0:
                # self-loop fallback with a dummy relation id 0
                adj_entity[node] = node
                adj_relation[node] = 0
                continue
            if len(neighbors) >= S:
                idx = torch.randperm(len(neighbors))[:S]
            else:
                idx = torch.randint(0, len(neighbors), (S,))
            chosen = [neighbors[i] for i in idx.tolist()]
            adj_relation[node] = torch.tensor([c[0] for c in chosen])
            adj_entity[node] = torch.tensor([c[1] for c in chosen])

        self.adj_entity = adj_entity
        self.adj_relation = adj_relation

    def _get_neighbors(self, seed_ids: torch.Tensor):
        """
        Builds the multi-hop receptive field for a batch of seed node ids.
        Returns:
            entities: list of length (num_hops+1), entities[0] = seeds,
                      entities[i] = neighbor ids at hop i, shape [batch, S^i]
            relations: list of length num_hops, relation ids at each hop
        """
        entities = [seed_ids.unsqueeze(1)]  # [B, 1]
        relations = []
        device = seed_ids.device
        for h in range(self.num_hops):
            cur = entities[-1]  # [B, prev_width]
            flat = cur.reshape(-1)
            neigh_e = self.adj_entity.to(device)[flat]     # [B*prev_width, S]
            neigh_r = self.adj_relation.to(device)[flat]   # [B*prev_width, S]
            B = seed_ids.size(0)
            neigh_e = neigh_e.view(B, -1)
            neigh_r = neigh_r.view(B, -1)
            entities.append(neigh_e)
            relations.append(neigh_r)
        return entities, relations

    def _aggregate(self, query_vec, entities, relations, node_features):
        """
        query_vec: [B, dim] -- embedding of the "query" side (publication)
        entities: from _get_neighbors
        relations: from _get_neighbors
        node_features: [num_nodes, in_dim] raw features, will be projected
        """
        device = query_vec.device
        entity_vectors = [self.input_proj(node_features.to(device)[e]) for e in entities]
        relation_vectors = [self.relation_emb(r) for r in relations]

        for hop in range(self.num_hops):
            next_entity_vectors = []
            agg = self.aggregators[hop]
            for depth in range(self.num_hops - hop):
                self_v = entity_vectors[depth]  # [B, W, dim]
                B, W, D = self_v.shape
                neigh_v = entity_vectors[depth + 1].view(B, W, self.neighbor_sample_size, D)
                neigh_r = relation_vectors[depth].view(B, W, self.neighbor_sample_size, D)

                # attention: query . relation, softmax over sampled neighbors
                q_exp = query_vec.view(B, 1, 1, D)
                attn_logits = (q_exp * neigh_r).sum(-1)  # [B, W, S]
                attn = F.softmax(attn_logits, dim=-1).unsqueeze(-1)  # [B,W,S,1]
                neighbor_agg = (attn * neigh_v).sum(dim=2)  # [B, W, D]

                out = agg(self_v, neighbor_agg)
                out = self.dropout(out)
                next_entity_vectors.append(out)
            entity_vectors = next_entity_vectors

        return entity_vectors[0].squeeze(1)  # [B, dim] final item(seed) representation

    def encode(self, query_ids: torch.Tensor, item_ids: torch.Tensor, node_features: torch.Tensor):
        """
        query_ids: publication (global) ids, [B]
        item_ids: dataset (global) ids, [B] -- the entities we run KGCN
                  neighborhood aggregation on, conditioned on the query
        """
        query_vec = self.input_proj(node_features[query_ids])
        entities, relations = self._get_neighbors(item_ids)
        item_repr = self._aggregate(query_vec, entities, relations, node_features)
        return query_vec, item_repr

    def score(self, head_ids: torch.Tensor, tail_ids: torch.Tensor):
        """Public scoring API used by metrics.evaluate_ranking.
        Expects self._cached_features to be set (see train.py) since the
        evaluator only passes node ids, not raw features."""
        assert hasattr(self, '_cached_features'), \
            'call model.set_features(node_features) before scoring'
        q, item_repr = self.encode(head_ids, tail_ids, self._cached_features)
        return (q * item_repr).sum(dim=-1)

    def set_features(self, node_features: torch.Tensor):
        self._cached_features = node_features

    def forward(self, head_ids, tail_ids, node_features):
        q, item_repr = self.encode(head_ids, tail_ids, node_features)
        return (q * item_repr).sum(dim=-1)
