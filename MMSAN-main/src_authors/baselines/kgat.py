"""
KGAT -- Knowledge Graph Attention Network
(Wang et al., "KGAT: Knowledge Graph Attention Network for Recommendation",
KDD 2019).

Two components, trained jointly:
  1. TransR-style KG embedding layer: learns entity embeddings `e` and
     relation embeddings `r` (+ a relation-specific projection matrix W_r)
     via a translation-based scoring function, trained with a margin/BPR
     loss on (h, r, t) vs (h, r, t') triples (t' corrupted).
  2. Attentive embedding propagation layers: aggregate neighbor entity
     embeddings weighted by attention scores computed from the TransR
     scoring function (knowledge-aware attention), stacked over L layers,
     then concatenate all-layer representations (as in the paper) to form
     the final entity embedding used for CF (here: link) scoring.

Final prediction score for (publication, dataset) = inner product of their
final multi-layer entity embeddings (as in the paper's CF/prediction layer).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KGAT(nn.Module):
    def __init__(self, num_nodes, num_relations, in_dim, embed_dim=64,
                 relation_dim=64, num_layers=2, dropout=0.1, aggregator='bi'):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.aggregator = aggregator

        self.input_proj = nn.Linear(in_dim, embed_dim)

        # TransR components
        self.relation_emb = nn.Embedding(num_relations, relation_dim)
        self.W_r = nn.Parameter(torch.empty(num_relations, embed_dim, relation_dim))
        nn.init.xavier_uniform_(self.W_r)

        # propagation layers
        self.W1 = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])
        self.W2 = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # sparse adjacency (built by set_graph), stored as edge list for
        # attention-based propagation (recomputed each forward since
        # attention depends on current embeddings)
        self.register_buffer('_dummy', torch.zeros(1), persistent=False)
        self.edge_index = None   # [2, E]  (head, tail) global ids
        self.edge_type = None    # [E]

    def set_graph(self, triples: torch.Tensor):
        """triples: [3, N] = (head, relation, tail)."""
        self.edge_index = torch.stack([triples[0], triples[2]], dim=0)
        self.edge_type = triples[1]

    # ---- TransR scoring, used both for KG-embedding loss and attention ----
    def _transr_score(self, h_emb, r_id, t_emb):
        """h_emb, t_emb: [E, embed_dim]; r_id: [E]. Returns scalar score per edge
        (negative L2 distance in relation space, higher = more plausible)."""
        Wr = self.W_r[r_id]                      # [E, embed_dim, relation_dim]
        r_vec = self.relation_emb(r_id)           # [E, relation_dim]
        h_proj = torch.bmm(h_emb.unsqueeze(1), Wr).squeeze(1)  # [E, relation_dim]
        t_proj = torch.bmm(t_emb.unsqueeze(1), Wr).squeeze(1)  # [E, relation_dim]
        score = -torch.norm(h_proj + r_vec - t_proj, p=2, dim=-1)
        return score

    def kg_embedding_loss(self, node_features, num_neg=1):
        """
        BPR-style TransR loss over the stored KG edges: for each positive
        (h, r, t), sample a corrupted tail t' and push
        score(h, r, t) > score(h, r, t').
        """
        device = self.edge_index.device
        h_ids, t_ids = self.edge_index[0], self.edge_index[1]
        r_ids = self.edge_type

        ent_emb = self.input_proj(node_features.to(device))
        h_emb = ent_emb[h_ids]
        t_emb = ent_emb[t_ids]

        pos_score = self._transr_score(h_emb, r_ids, t_emb)

        neg_t_ids = torch.randint(0, self.num_nodes, (h_ids.size(0) * num_neg,), device=device)
        neg_t_emb = ent_emb[neg_t_ids]
        h_emb_rep = h_emb.repeat_interleave(num_neg, dim=0)
        r_ids_rep = r_ids.repeat_interleave(num_neg, dim=0)
        neg_score = self._transr_score(h_emb_rep, r_ids_rep, neg_t_emb)
        neg_score = neg_score.view(h_ids.size(0), num_neg).mean(dim=1)

        loss = F.softplus(neg_score - pos_score).mean()
        return loss

    # -------- attentive propagation --------
    def _attention_weights(self, ent_emb):
        h_ids, t_ids = self.edge_index[0], self.edge_index[1]
        r_ids = self.edge_type
        h_emb = ent_emb[h_ids]
        t_emb = ent_emb[t_ids]
        raw_attn = self._transr_score(h_emb, r_ids, t_emb)  # [E]

        # softmax over incoming edges per target node (segment softmax)
        attn = raw_attn - raw_attn.max()
        attn = attn.exp()
        denom = torch.zeros(self.num_nodes, device=ent_emb.device).scatter_add_(
            0, t_ids, attn) + 1e-10
        attn = attn / denom[t_ids]
        return attn

    def _propagate_layer(self, ent_emb, layer_idx):
        h_ids, t_ids = self.edge_index[0], self.edge_index[1]
        attn = self._attention_weights(ent_emb)  # [E]

        h_emb = ent_emb[h_ids]
        weighted = attn.unsqueeze(-1) * h_emb  # [E, dim]

        agg = torch.zeros_like(ent_emb).scatter_add_(
            0, t_ids.unsqueeze(-1).expand(-1, ent_emb.size(-1)), weighted)

        if self.aggregator == 'bi':
            sum_term = self.W1[layer_idx](ent_emb + agg)
            bi_term = self.W2[layer_idx](ent_emb * agg)
            out = self.leaky_relu(sum_term) + self.leaky_relu(bi_term)
        else:  # 'gcn'-style
            out = self.leaky_relu(self.W1[layer_idx](ent_emb + agg))

        out = self.dropout(out)
        out = F.normalize(out, p=2, dim=-1)
        return out

    def encode_all(self, node_features: torch.Tensor):
        """Returns final entity embeddings for ALL nodes (concat of every
        propagation layer's output, as in the KGAT paper)."""
        device = self.edge_index.device
        ent_emb = self.input_proj(node_features.to(device))
        layer_outs = [ent_emb]
        cur = ent_emb
        for l in range(self.num_layers):
            cur = self._propagate_layer(cur, l)
            layer_outs.append(cur)
        final = torch.cat(layer_outs, dim=-1)  # [num_nodes, embed_dim*(num_layers+1)]
        return final

    def set_features(self, node_features: torch.Tensor):
        """Precompute and cache full-graph propagated embeddings, used by
        `score()` for evaluation. Call again after each training epoch."""
        with torch.no_grad():
            self._cached_embeddings = self.encode_all(node_features)

    def score(self, head_ids: torch.Tensor, tail_ids: torch.Tensor):
        assert hasattr(self, '_cached_embeddings'), \
            'call model.set_features(node_features) before scoring'
        h = self._cached_embeddings[head_ids]
        t = self._cached_embeddings[tail_ids]
        return (h * t).sum(dim=-1)

    def cf_loss(self, node_features, pos_edge_index, neg_edge_index):
        """BPR loss on the target (publication, dataset) relation using the
        full attentive final embeddings (recomputed with grad this time)."""
        final_emb = self.encode_all(node_features)
        h_pos = final_emb[pos_edge_index[0]]
        t_pos = final_emb[pos_edge_index[1]]
        h_neg = final_emb[neg_edge_index[0]]
        t_neg = final_emb[neg_edge_index[1]]

        pos_score = (h_pos * t_pos).sum(dim=-1)
        neg_score = (h_neg * t_neg).sum(dim=-1)
        loss = F.softplus(neg_score - pos_score).mean()
        return loss
