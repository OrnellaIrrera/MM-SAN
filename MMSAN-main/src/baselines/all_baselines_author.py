"""
KGCN / KGAT / VGAE / HVGAE / RGCN / SAGE / GAT / HAN / HGT per DATASET
RECOMMENDATION SU AUTORE (author -> dataset), su ScholarlyDataset.

Questo file e' l'adattamento di kg_dataset_recommendation.py (che
raccomandava dataset a partire da una PUBLICATION come query) al task in
cui la query e' invece un AUTORE: dato un autore, ordinare i dataset che
potrebbe voler usare/citare. Il loader.py fornito espone gia' nativamente
questo task: la relazione ('author', 'isauthor', 'dataset') contiene tutti
gli split di training/validation/test (transductive, semi-inductive,
full-inductive) pre-costruiti dal loader stesso (edge_index_train,
edge_label_index_validation_trans, edge_label_index_test_trans,
edge_label_index_test_semi, edge_label_index_test_ind), esattamente come
prima accadeva per ('publication', 'cites', 'dataset') nello script di
riferimento.

Rispetto allo script originale, le differenze concettuali sono:

  1) Il ruolo di "user"/query per tutti i modelli a embedding statici
     (RGCN, KGAT, SAGE, GAT, HAN, HGT) e per KGCN passa da 'publication' ad
     'author'. La relazione target della link prediction (con
     leakage-avoidance nel grafo di message-passing) passa da
     ('publication','cites','dataset') ad ('author','isauthor','dataset').

  2) La relazione ('publication','cites','dataset') NON e' piu' il target:
     nel loader ha un edge_index "normale" (pd_edge_index, senza split),
     quindi ora e' semplicemente una relazione ausiliaria del KG come le
     altre (paper-citation, paper-topic, ecc.), usata solo per il message
     passing in RGCN/KGAT/SAGE/GAT/HAN/HGT/KGCN/VGAE.

  3) HVGAE (Altaf et al., ICDM 2019) nel paper originale usa tre input
     specifici: contenuto testuale del paper (X_P), grafo di citazione fra
     paper (A_P,P) e bipartito paper-dataset (H_P,D). Il loader NON
     contiene pero' una relazione author-author nativa (a differenza di
     publication-publication). Per restare fedeli all'architettura del
     paper (che richiede un grafo omogeneo fra le entita' "query" per il
     GraphSAGE dell'encoder), qui il grafo di citazione A_P,P e' sostituito
     da un grafo di CO-AUTORIALITA' A_A,A, costruito a runtime come
     proiezione del bipartito author-publication (author-isauthor-
     publication): due autori sono collegati se compaiono come coautori di
     almeno una stessa publication. E' l'analogo piu' diretto di un grafo
     "chi e' collegato a chi" fra le entita' che qui giocano il ruolo di
     query, al posto delle citazioni fra paper. Il resto dell'architettura
     (encoder autore = GraphSAGE + head mu/var, encoder dataset = MLP su
     H_A,D^T, decoder a prodotto scalare, scoring via KL-divergenza) resta
     identico al paper.

Uso (identico allo script originale, stesso CLI):
    python kg_author_dataset_recommendation.py -model kgcn -dataset mes -epochs 100
    python kg_author_dataset_recommendation.py -model kgat -dataset mes -epochs 100
    python kg_author_dataset_recommendation.py -model vgae -dataset mes -epochs 100
    python kg_author_dataset_recommendation.py -model hvgae -dataset mes -epochs 100
    python kg_author_dataset_recommendation.py -model sage -dataset mes -epochs 100
    python kg_author_dataset_recommendation.py -model gat -dataset mes -epochs 100
    python kg_author_dataset_recommendation.py -model han -dataset mes -epochs 100
    python kg_author_dataset_recommendation.py -model hgt -dataset mes -epochs 100

Assume che `loader.py` sia importabile come modulo con la classe
ScholarlyDataset (stesso contratto dati fornito, che espone gia'
nativamente gli split per ('author','isauthor','dataset')).
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
from torch_geometric.nn import (
    RGCNConv,
    SAGEConv,
    GATConv,
    HANConv,
    HGTConv,
    Linear as PyGLinear,
)
from torch_geometric.utils import softmax as scatter_softmax

# ----------------------------------------------------------------------------
# Riproducibilita'
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
# COSTRUZIONE GRAFO OMOGENEO
# ----------------------------------------------------------------------------
def build_homogeneous_graph(data: HeteroData, feature_key='x'):
    """
    Converte HeteroData in:
      - x_dict: {ntype: tensor features}
      - edge_index (2, E) globale, edge_type (E,)
      - offsets: {ntype: start_index} per mappare id locali -> id globali
      - node_types ordinati

    Usa solo gli edge_index "di message passing" (edge_index_train per
    author-dataset, e i restanti edge_index normali per le altre
    relazioni, inclusa publication-dataset che qui NON e' piu' il target
    della link prediction), per evitare data leakage dai link di
    validation/test dell'autore.

    NOTA: usata da RGCN, KGCN, KGAT, SAGE e GAT. SAGE e GAT non sono
    relation-aware, ma riusano comunque questo grafo (ignorando edge_type)
    per beneficiare della stessa leakage-avoidance e dello stesso set di
    nodi/archi degli altri modelli. Il modello HVGAE, seguendo fedelmente
    il paper di riferimento (adattato ad autore, si veda il docstring del
    modulo), NON la usa: costruisce invece i propri input (X_A, A_A,A,
    H_A,D) direttamente in train_hvgae(). HAN e HGT, essendo nativamente
    eterogenei, usano invece build_heterogeneous_graph().
    """
    node_types = list(data.node_types)
    num_nodes_dict = {nt: data[nt].num_nodes for nt in node_types}

    offsets = {}
    cum = 0
    for nt in node_types:
        offsets[nt] = cum
        cum += num_nodes_dict[nt]
    total_nodes = cum

    x_dict = {}
    in_channels_dict = {}
    for nt in node_types:
        feat = getattr(data[nt], feature_key, None)
        if feat is None:
            feat = data[nt].net_x
        feat = feat.float()
        x_dict[nt] = feat
        in_channels_dict[nt] = feat.size(-1)

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
        # relazione target della link prediction (autore -> dataset):
        # SOLO gli edge di training per evitare leakage da val/test
        if src_type == 'author' and dst_type == 'dataset' and rel_name == 'isauthor':
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


def build_heterogeneous_graph(data: HeteroData, feature_key='x'):
    """
    Costruisce le strutture "nativamente eterogenee" (x_dict, edge_index_dict,
    metadata) necessarie per HAN e HGT.

    Stessa logica di leakage-avoidance di build_homogeneous_graph: per la
    relazione author-isauthor-dataset si usa SOLO edge_index_train, le
    altre relazioni (inclusa publication-cites-dataset) usano l'edge_index
    normale.
    """
    node_types = list(data.node_types)

    x_dict = {}
    in_channels_dict = {}
    for nt in node_types:
        feat = getattr(data[nt], feature_key, None)
        if feat is None:
            feat = data[nt].net_x
        feat = feat.float()
        x_dict[nt] = feat
        in_channels_dict[nt] = feat.size(-1)

    edge_index_dict = {}

    def add_relation(src_type, rel_name, dst_type, edge_index, bidirectional=True):
        if edge_index is None or edge_index.numel() == 0:
            return
        edge_index_dict[(src_type, rel_name, dst_type)] = edge_index
        if bidirectional:
            edge_index_dict[(dst_type, rel_name + '_rev', src_type)] = edge_index.flip(0)

    for et in data.edge_types:
        src_type, rel_name, dst_type = et
        if src_type == 'author' and dst_type == 'dataset' and rel_name == 'isauthor':
            ei = data[et].get('edge_index_train', None)
            add_relation(src_type, rel_name, dst_type, ei)
            continue

        ei = data[et].get('edge_index', None)
        if ei is None:
            continue
        add_relation(src_type, rel_name, dst_type, ei)

    metadata = (node_types, list(edge_index_dict.keys()))

    return {
        'x_dict': x_dict,
        'in_channels_dict': in_channels_dict,
        'node_types': node_types,
        'edge_index_dict': edge_index_dict,
        'metadata': metadata,
    }


def get_coauthorship_edge_index(data: HeteroData):
    """
    Costruisce un grafo di CO-AUTORIALITA' A_A,A fra autori, analogo del
    grafo di citazione A_P,P usato dall'HVGAE originale fra i paper.

    Il loader.py fornito non contiene una relazione author-author nativa;
    la coautorialita' viene quindi derivata qui a runtime come proiezione
    del bipartito author-isauthor-publication (data['author', 'isauthor',
    'publication'].edge_index, SEMPRE l'edge_index completo: questa
    relazione non e' il target della link prediction e non richiede quindi
    alcuna leakage-avoidance): due autori sono collegati se compaiono
    entrambi fra gli autori di almeno una stessa publication.

    Ritorna un edge_index [2, E] (senza duplicati, gia' orientato in
    entrambe le direzioni).
    """
    et = ('author', 'isauthor', 'publication')
    if et not in data.edge_types:
        raise ValueError(
            "Nessuna relazione author-isauthor-publication trovata in "
            "HeteroData. Il modello hvgae (adattato ad autore) richiede "
            "questa relazione per costruire il grafo di co-autorialita' "
            "A_A,A. Verifica il contratto dati del tuo loader.py."
        )
    ei = data[et].edge_index
    if ei is None or ei.numel() == 0:
        raise ValueError(
            "La relazione author-isauthor-publication e' vuota: impossibile "
            "costruire il grafo di co-autorialita' richiesto da hvgae."
        )

    pub_to_authors = defaultdict(list)
    src, dst = ei[0].tolist(), ei[1].tolist()
    for a, p in zip(src, dst):
        pub_to_authors[p].append(a)

    edges_src, edges_dst = [], []
    for _, authors in pub_to_authors.items():
        if len(authors) < 2:
            continue
        for i in range(len(authors)):
            for j in range(len(authors)):
                if i != j:
                    edges_src.append(authors[i])
                    edges_dst.append(authors[j])

    if len(edges_src) == 0:
        # nessuna coautorialita' osservata (es. dataset con un solo autore
        # per publication): ritorna un grafo vuoto, il GraphSAGE dell'
        # encoder autore degradera' semplicemente all'aggregazione del solo
        # self-loop implicito di SAGEConv.
        return torch.zeros((2, 0), dtype=torch.long)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_index = torch.unique(edge_index.t(), dim=0).t().contiguous()
    return edge_index


# ----------------------------------------------------------------------------
# BilinearScorer (usato da RGCN, KGAT, SAGE, GAT, HAN, HGT: embedding
# statici per tutti i nodi)
# ----------------------------------------------------------------------------
class BilinearScorer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, src_emb, dst_emb):
        src_emb = F.normalize(src_emb, p=2, dim=-1)
        dst_emb = F.normalize(dst_emb, p=2, dim=-1)
        return (src_emb @ self.weight * dst_emb).sum(dim=-1)

    def score_all(self, src_emb, dst_all_emb):
        src_emb = F.normalize(src_emb, p=2, dim=-1)
        dst_all_emb = F.normalize(dst_all_emb, p=2, dim=-1)
        return dst_all_emb @ (self.weight.t() @ src_emb)


# ----------------------------------------------------------------------------
# RGCN (baseline)
# ----------------------------------------------------------------------------
class RGCNEncoder(nn.Module):
    def __init__(self, node_types, in_channels_dict, hidden_channels,
                 num_relations, num_bases=30, num_layers=2, dropout=0.2):
        super().__init__()
        self.node_types = node_types
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.input_proj = nn.ModuleDict()
        for ntype in node_types:
            self.input_proj[ntype] = nn.Linear(in_channels_dict[ntype], hidden_channels)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases)
            )

    def forward(self, x_dict, edge_index, edge_type):
        xs = [self.input_proj[nt](x_dict[nt]) for nt in self.node_types]
        x = torch.cat(xs, dim=0)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ----------------------------------------------------------------------------
# SAGE / GAT: encoder "classici", non relation-aware, sul grafo omogeneo
# ----------------------------------------------------------------------------
class SAGEEncoderKG(nn.Module):
    def __init__(self, node_types, in_channels_dict, hidden_channels,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.node_types = node_types
        self.dropout = dropout

        self.input_proj = nn.ModuleDict()
        for ntype in node_types:
            self.input_proj[ntype] = nn.Linear(in_channels_dict[ntype], hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

    def forward(self, x_dict, edge_index):
        xs = [self.input_proj[nt](x_dict[nt]) for nt in self.node_types]
        x = torch.cat(xs, dim=0)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATEncoderKG(nn.Module):
    def __init__(self, node_types, in_channels_dict, hidden_channels,
                 num_layers=2, heads=4, dropout=0.2):
        super().__init__()
        self.node_types = node_types
        self.dropout = dropout

        self.input_proj = nn.ModuleDict()
        for ntype in node_types:
            self.input_proj[ntype] = nn.Linear(in_channels_dict[ntype], hidden_channels)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            concat = i < num_layers - 1
            out_per_head = hidden_channels // heads if concat else hidden_channels
            self.convs.append(
                GATConv(hidden_channels, out_per_head, heads=heads, concat=concat, dropout=dropout)
            )

    def forward(self, x_dict, edge_index):
        xs = [self.input_proj[nt](x_dict[nt]) for nt in self.node_types]
        x = torch.cat(xs, dim=0)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ----------------------------------------------------------------------------
# KGCN: aggregazione condizionata dalla query (autore), per tipo di relazione
# ----------------------------------------------------------------------------
class KGCNEncoder(nn.Module):
    """
    Adattamento di KGCN (Wang et al. 2019) al caso author -> dataset:
      - 'author' = user, 'dataset' = item, il resto del KG sono entita'.
      - Per ogni query (embedding dell'autore) si ricalcolano gli embedding
        di TUTTI i nodi del grafo, propagando l'informazione dei vicini
        pesata da un'attenzione sul TIPO di relazione:
            alpha_r = softmax_r( rel_emb_r . query )
      - Il punteggio finale e' il prodotto scalare tra la query e
        l'embedding finale (condizionato) del dataset.
    """

    def __init__(self, node_types, in_channels_dict, hidden_channels,
                 num_relations, num_layers=2, dropout=0.2):
        super().__init__()
        self.node_types = node_types
        self.hidden_channels = hidden_channels
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.dropout = dropout

        self.input_proj = nn.ModuleDict()
        for ntype in node_types:
            self.input_proj[ntype] = nn.Linear(in_channels_dict[ntype], hidden_channels)

        self.rel_emb = nn.Embedding(num_relations, hidden_channels)
        self.self_transform = nn.ModuleList(
            [nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )
        self.neigh_transform = nn.ModuleList(
            [nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )

    def base_embeddings(self, x_dict):
        xs = [self.input_proj[nt](x_dict[nt]) for nt in self.node_types]
        return torch.cat(xs, dim=0)

    def conditioned_embeddings(self, x0, edge_index, edge_type, total_nodes, query_emb):
        src, dst = edge_index
        rel_scores = self.rel_emb.weight @ query_emb  # [R]
        alpha = F.softmax(rel_scores, dim=0)  # [R]
        edge_weight = alpha[edge_type]  # [E]

        x = x0
        for l in range(self.num_layers):
            weighted_src = x[src] * edge_weight.unsqueeze(-1)  # [E, H]
            message = x.new_zeros(total_nodes, self.hidden_channels)
            message.index_add_(0, dst, weighted_src)

            x_self = self.self_transform[l](x)
            x_neigh = self.neigh_transform[l](message)
            x = F.relu(x_self + x_neigh)
            if l < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ----------------------------------------------------------------------------
# KGAT: propagazione ad alto ordine con attenzione stile TransR + bi-interaction
# ----------------------------------------------------------------------------
class KGATEncoder(nn.Module):
    def __init__(self, node_types, in_channels_dict, hidden_channels,
                 num_relations, num_layers=3, dropout=0.2):
        super().__init__()
        self.node_types = node_types
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.out_dim = hidden_channels * (num_layers + 1)

        self.input_proj = nn.ModuleDict()
        for ntype in node_types:
            self.input_proj[ntype] = nn.Linear(in_channels_dict[ntype], hidden_channels)

        self.rel_emb = nn.Embedding(num_relations, hidden_channels)
        self.rel_transform = nn.Embedding(num_relations, hidden_channels)
        nn.init.ones_(self.rel_transform.weight)

        self.attn_vec = nn.ModuleList(
            [nn.Linear(hidden_channels, 1, bias=False) for _ in range(num_layers)]
        )
        self.w1 = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers)])
        self.w2 = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers)])
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x_dict, edge_index, edge_type, total_nodes):
        xs = [self.input_proj[nt](x_dict[nt]) for nt in self.node_types]
        x = torch.cat(xs, dim=0)
        src, dst = edge_index

        all_layers = [x]
        for l in range(self.num_layers):
            rel_scale = self.rel_transform(edge_type)
            rel_vec = self.rel_emb(edge_type)

            e_h = x[src] * rel_scale
            e_t = x[dst] * rel_scale

            hidden = torch.tanh(e_h + rel_vec)
            score = self.attn_vec[l](hidden).squeeze(-1) + (e_t * hidden).sum(dim=-1)

            alpha = scatter_softmax(score, src, num_nodes=total_nodes)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

            messages = alpha.unsqueeze(-1) * e_t
            neigh_agg = x.new_zeros(total_nodes, self.hidden_channels)
            neigh_agg.index_add_(0, src, messages)

            bi1 = self.leaky_relu(self.w1[l](x + neigh_agg))
            bi2 = self.leaky_relu(self.w2[l](x * neigh_agg))
            x = bi1 + bi2
            x = F.dropout(x, p=self.dropout, training=self.training)
            all_layers.append(x)

        return torch.cat(all_layers, dim=-1)


# ----------------------------------------------------------------------------
# HAN / HGT: encoder nativamente eterogenei (x_dict / edge_index_dict)
# ----------------------------------------------------------------------------
class HANEncoderKG(nn.Module):
    def __init__(self, in_channels_dict, hidden_channels, metadata, heads=4, dropout=0.2):
        super().__init__()
        self.han_conv = HANConv(in_channels_dict, hidden_channels, heads=heads,
                                 dropout=dropout, metadata=metadata)

    def forward(self, x_dict, edge_index_dict):
        return self.han_conv(x_dict, edge_index_dict)


class HGTEncoderKG(nn.Module):
    def __init__(self, node_types, in_channels_dict, hidden_channels,
                 metadata, heads=8, num_layers=1):
        super().__init__()
        self.lin_dict = nn.ModuleDict()
        for ntype in node_types:
            self.lin_dict[ntype] = PyGLinear(in_channels_dict[ntype], hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HGTConv(hidden_channels, hidden_channels, metadata, heads))

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        return x_dict


# ----------------------------------------------------------------------------
# VGAE: encoder relazionale (mu, logvar) + decoder a prodotto scalare
# ----------------------------------------------------------------------------
class VGAEEncoder(nn.Module):
    def __init__(self, node_types, in_channels_dict, hidden_channels,
                 num_relations, num_bases=30, num_layers=2, dropout=0.2):
        super().__init__()
        self.node_types = node_types
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.input_proj = nn.ModuleDict()
        for ntype in node_types:
            self.input_proj[ntype] = nn.Linear(in_channels_dict[ntype], hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(max(num_layers - 1, 0)):
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases)
            )

        self.conv_mu = RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases)
        self.conv_logvar = RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases)

    def forward(self, x_dict, edge_index, edge_type):
        xs = [self.input_proj[nt](x_dict[nt]) for nt in self.node_types]
        x = torch.cat(xs, dim=0)

        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_type))
            x = F.dropout(x, p=self.dropout, training=self.training)

        mu = self.conv_mu(x, edge_index, edge_type)
        logvar = self.conv_logvar(x, edge_index, edge_type)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    @staticmethod
    def kl_loss(mu, logvar):
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


class InnerProductScorer(nn.Module):
    def forward(self, src_emb, dst_emb):
        src_emb = F.normalize(src_emb, p=2, dim=-1)
        dst_emb = F.normalize(dst_emb, p=2, dim=-1)
        return (src_emb * dst_emb).sum(dim=-1)

    def score_all(self, src_emb, dst_all_emb):
        src_emb = F.normalize(src_emb, p=2, dim=-1)
        dst_all_emb = F.normalize(dst_all_emb, p=2, dim=-1)
        return dst_all_emb @ src_emb


# ----------------------------------------------------------------------------
# HVGAE (Altaf et al., ICDM 2019) adattato ad AUTORE - encoder autore
# (GraphSAGE su grafo di co-autorialita') + encoder dataset (MLP su H^T) +
# decoder a prodotto scalare + scoring via KL
# ----------------------------------------------------------------------------
class HVGAEAuthorEncoder(nn.Module):
    """
    Encoder degli autori, analogo dell'encoder dei paper del paper HVGAE
    (Sezione III-A, Eq. 2-3): GraphSAGE con k=2 iterazioni di default
    (Combine=sum, Aggregate=mean), applicato pero' al grafo di
    CO-AUTORIALITA' A_A,A (get_coauthorship_edge_index) invece che al grafo
    di citazione A_P,P, seguito da due head lineari che producono media e
    varianza della distribuzione gaussiana di ogni autore:

        h_a^k = Combine(h_a^{k-1}, Aggregate({h_a'^{k-1}, a' in N(a)}))
        mu_a      = ReLU(W h_a + b)
        sigma_a^2 = exp(ReLU(W h_a + b))
    """

    def __init__(self, in_channels, hidden_channels, latent_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        c_in = in_channels
        for _ in range(num_layers):
            self.convs.append(SAGEConv(c_in, hidden_channels, aggr='mean'))
            c_in = hidden_channels

        self.mu_head = nn.Linear(hidden_channels, latent_dim)
        self.var_head = nn.Linear(hidden_channels, latent_dim)

    def forward(self, x, edge_index):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(h)

        mu = F.relu(self.mu_head(h))
        var = torch.exp(F.relu(self.var_head(h)))
        return mu, var


class HVGAEDatasetEncoder(nn.Module):
    """
    Encoder dei dataset (Eq. 5 del paper HVGAE): MLP a due layer che prende
    in input H_A,D^T, cioe' per ogni dataset il vettore binario degli
    autori (di training) che lo hanno usato/citato:

        h_d = MLP(H_A,D^T)
        mu_d      = ReLU(W h_d + b)
        sigma_d^2 = exp(ReLU(W h_d + b))
    """

    def __init__(self, num_author, hidden_channels, latent_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(num_author, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = dropout

        self.mu_head = nn.Linear(hidden_channels, latent_dim)
        self.var_head = nn.Linear(hidden_channels, latent_dim)

    def forward(self, H_T):
        h = F.relu(self.fc1(H_T))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.fc2(h))

        mu = F.relu(self.mu_head(h))
        var = torch.exp(F.relu(self.var_head(h)))
        return mu, var


def reparameterize_gaussian(mu, var, training):
    if training:
        std = torch.sqrt(var + 1e-8)
        eps = torch.randn_like(std)
        return mu + eps * std
    return mu


def kl_to_standard_normal(mu, var):
    logvar = torch.log(var + 1e-8)
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - var, dim=1))


def kl_query_dataset(mu_q, var_q, mu_d, var_d):
    """
    Punteggio di rilevanza query-dataset s(Q, d) (Eq. 15 del paper HVGAE),
    qui con Q = un singolo autore. Piu' BASSA e' la KL, piu' il dataset e'
    rilevante; il valore ritornato e' NEGATO, cosi' che punteggi piu' alti
    restino "piu' rilevanti", in linea con la convenzione usata da
    evaluate_ranking_with_scorefn (ordinamento decrescente) e dagli altri
    modelli (score_all()).
    """
    sigma_q = torch.sqrt(var_q + 1e-8)
    sigma_d = torch.sqrt(var_d + 1e-8)

    term1 = torch.log(sigma_d / sigma_q.unsqueeze(0) + 1e-8)
    term2 = (var_q.unsqueeze(0) + (mu_q.unsqueeze(0) - mu_d).pow(2)) / (2 * var_d + 1e-8)
    kl = (term1 + term2 - 0.5).sum(dim=-1)
    return -kl


def aggregate_query(mu_a_query, var_a_query):
    """Rappresentazione della query come distribuzione gaussiana (Eq. 14):
    media element-wise di mu e di sigma^2 sugli autori della query."""
    mu_Q = mu_a_query.mean(dim=0)
    var_Q = var_a_query.mean(dim=0)
    return mu_Q, var_Q


# ----------------------------------------------------------------------------
# Sampling negativi
# ----------------------------------------------------------------------------
def sample_negative_datasets(pos_edge_index, num_src, num_dataset, num_neg_per_pos=1):
    E = pos_edge_index.size(1)
    neg_dst = torch.randint(0, num_dataset, (E * num_neg_per_pos,))
    neg_src = pos_edge_index[0].repeat_interleave(num_neg_per_pos)
    return torch.stack([neg_src, neg_dst], dim=0)


def sample_negative_coauthor_pairs(pos_edge_index, num_author, num_neg_per_pos=1):
    """Negativi per la ricostruzione di A_A,A (usata da HVGAE): per ogni
    coppia di coautori positiva (a_i -> a_j), campiona un a_j' casuale come
    destinazione negativa."""
    E = pos_edge_index.size(1)
    neg_dst = torch.randint(0, num_author, (E * num_neg_per_pos,))
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


def build_eval_pairs(edge_label_index):
    """Da edge_index [2, E] (author, dataset) costruisce dict author -> set(dataset)."""
    pairs = defaultdict(set)
    src = edge_label_index[0].tolist()
    dst = edge_label_index[1].tolist()
    for s, d in zip(src, dst):
        pairs[s].add(d)
    return pairs


def evaluate_ranking_with_scorefn(score_fn, eval_pairs, k_list=(5, 10)):
    """
    score_fn(author_id) -> tensor [num_dataset] con i punteggi di quell'
    autore contro tutti i dataset. Riusabile sia per modelli a embedding
    statici (RGCN, KGAT, SAGE, GAT, HAN, HGT) sia per modelli
    query-dipendenti (KGCN), sia per modelli basati su KL-divergenza
    (HVGAE).
    """
    mrr_at = {k: [] for k in k_list}
    ndcg_at = {k: [] for k in k_list}
    recall_at = {k: [] for k in k_list}

    with torch.no_grad():
        for author_id, relevant_set in eval_pairs.items():
            if len(relevant_set) == 0:
                continue

            scores = score_fn(author_id)
            ranked_idx = torch.argsort(scores, descending=True).cpu().numpy()

            rank_of_first_relevant = None
            for pos, idx in enumerate(ranked_idx):
                if idx in relevant_set:
                    rank_of_first_relevant = pos + 1
                    break

            for k in k_list:
                if rank_of_first_relevant is not None and rank_of_first_relevant <= k:
                    mrr_at[k].append(1.0 / rank_of_first_relevant)
                else:
                    mrr_at[k].append(0.0)

                top_k_idx = ranked_idx[:k]
                hits = sum(1 for idx in top_k_idx if idx in relevant_set)
                recall_at[k].append(hits / min(len(relevant_set), k))

                rel_at_k = [1 if idx in relevant_set else 0 for idx in top_k_idx]
                ndcg_at[k].append(ndcg_at_k(rel_at_k, len(relevant_set), k))

    results = {}
    for k in k_list:
        results[f'MRR@{k}'] = float(np.mean(mrr_at[k])) if mrr_at[k] else 0.0
        results[f'NDCG@{k}'] = float(np.mean(ndcg_at[k])) if ndcg_at[k] else 0.0
        results[f'Recall@{k}'] = float(np.mean(recall_at[k])) if recall_at[k] else 0.0
    return results


# ----------------------------------------------------------------------------
# TRAINING: modelli a embedding statici sul grafo omogeneo (RGCN, KGAT,
# SAGE, GAT)
# ----------------------------------------------------------------------------
def train_static_model(data, args, graph, device):
    node_types = graph['node_types']
    offsets = graph['offsets']
    total_nodes = graph['total_nodes']

    if args.model == 'rgcn':
        model = RGCNEncoder(
            node_types=node_types,
            in_channels_dict=graph['in_channels_dict'],
            hidden_channels=args.hidden,
            num_relations=graph['num_relations'],
            num_bases=min(30, graph['num_relations']),
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
        out_dim = args.hidden
    elif args.model == 'kgat':
        model = KGATEncoder(
            node_types=node_types,
            in_channels_dict=graph['in_channels_dict'],
            hidden_channels=args.hidden,
            num_relations=graph['num_relations'],
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
        out_dim = model.out_dim
    elif args.model == 'sage':
        model = SAGEEncoderKG(
            node_types=node_types,
            in_channels_dict=graph['in_channels_dict'],
            hidden_channels=args.hidden,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
        out_dim = args.hidden
    elif args.model == 'gat':
        model = GATEncoderKG(
            node_types=node_types,
            in_channels_dict=graph['in_channels_dict'],
            hidden_channels=args.hidden,
            num_layers=args.num_layers,
            heads=args.gat_heads,
            dropout=args.dropout,
        ).to(device)
        out_dim = args.hidden
    else:
        raise ValueError(f'Modello statico sconosciuto: {args.model}')

    scorer = BilinearScorer(out_dim).to(device)

    x_dict = {nt: graph['x_dict'][nt].to(device) for nt in node_types}
    edge_index = graph['edge_index'].to(device)
    edge_type = graph['edge_type'].to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(scorer.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    author_off = offsets['author']
    ds_off = offsets['dataset']
    num_author = data['author'].num_nodes
    num_dataset = data['dataset'].num_nodes

    train_edge_index = data['author', 'isauthor', 'dataset'].edge_index_train.to(device)

    eval_splits = {}
    for split in ['validation_trans', 'test_trans', 'test_semi', 'test_ind']:
        key = f'edge_label_index_{split}'
        if key in data['author', 'isauthor', 'dataset']:
            eli = data['author', 'isauthor', 'dataset'][key]
            eval_splits[split] = build_eval_pairs(eli)

    def forward_embeddings():
        if args.model == 'rgcn':
            return model(x_dict, edge_index, edge_type)
        elif args.model in ('sage', 'gat'):
            return model(x_dict, edge_index)
        else:  # kgat
            return model(x_dict, edge_index, edge_type, total_nodes)

    best_val_metric = -1
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        scorer.train()
        optimizer.zero_grad()

        emb = forward_embeddings()
        author_emb_all = emb[author_off: author_off + num_author]
        ds_emb_all = emb[ds_off: ds_off + num_dataset]

        pos_src = train_edge_index[0]
        pos_dst = train_edge_index[1]
        pos_scores = scorer(author_emb_all[pos_src], ds_emb_all[pos_dst])

        neg_edge_index = sample_negative_datasets(
            train_edge_index.cpu(), num_author, num_dataset, num_neg_per_pos=args.num_neg
        ).to(device)
        neg_scores = scorer(author_emb_all[neg_edge_index[0]], ds_emb_all[neg_edge_index[1]])

        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        scores = torch.cat([pos_scores, neg_scores])
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(scorer.parameters()), max_norm=5.0
        )
        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            scorer.eval()
            with torch.no_grad():
                emb = forward_embeddings()
                author_emb_all = emb[author_off: author_off + num_author]
                ds_emb_all = emb[ds_off: ds_off + num_dataset]

            log_line = f'[{args.model}] Epoch {epoch:03d} | loss {loss.item():.4f}'
            if 'validation_trans' in eval_splits:
                score_fn = lambda aid: scorer.score_all(author_emb_all[aid], ds_emb_all)
                val_results = evaluate_ranking_with_scorefn(
                    score_fn, eval_splits['validation_trans'], k_list=(5, 10)
                )
                log_line += f" | val MRR@10 {val_results['MRR@10']:.4f} NDCG@10 {val_results['NDCG@10']:.4f}"

                if val_results['NDCG@10'] > best_val_metric:
                    best_val_metric = val_results['NDCG@10']
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

    if best_state is not None:
        model.load_state_dict(best_state['model'])
        scorer.load_state_dict(best_state['scorer'])

    model.eval()
    scorer.eval()
    with torch.no_grad():
        emb = forward_embeddings()
        author_emb_all = emb[author_off: author_off + num_author]
        ds_emb_all = emb[ds_off: ds_off + num_dataset]

    print('\n=== Risultati finali ===')
    final_results = {}
    for split in ['test_trans', 'test_semi', 'test_ind']:
        if split in eval_splits:
            score_fn = lambda aid: scorer.score_all(author_emb_all[aid], ds_emb_all)
            res = evaluate_ranking_with_scorefn(score_fn, eval_splits[split], k_list=(5, 10))
            final_results[split] = res
            print(f'[{split}]', {k: round(v, 4) for k, v in res.items()})

    return final_results, model, scorer


# ----------------------------------------------------------------------------
# TRAINING: modelli nativamente eterogenei (HAN, HGT)
# ----------------------------------------------------------------------------
def train_hetero_model(data, args, graph, device):
    node_types = graph['node_types']
    metadata = graph['metadata']

    x_dict = {nt: graph['x_dict'][nt].to(device) for nt in node_types}
    edge_index_dict = {et: ei.to(device) for et, ei in graph['edge_index_dict'].items()}

    if args.model == 'han':
        model = HANEncoderKG(
            in_channels_dict=graph['in_channels_dict'],
            hidden_channels=args.hidden,
            metadata=metadata,
            heads=args.han_heads,
            dropout=args.dropout,
        ).to(device)
    elif args.model == 'hgt':
        model = HGTEncoderKG(
            node_types=node_types,
            in_channels_dict=graph['in_channels_dict'],
            hidden_channels=args.hidden,
            metadata=metadata,
            heads=args.hgt_heads,
            num_layers=args.num_layers,
        ).to(device)
    else:
        raise ValueError(f'Modello eterogeneo sconosciuto: {args.model}')

    scorer = BilinearScorer(args.hidden).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(scorer.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    num_author = data['author'].num_nodes
    num_dataset = data['dataset'].num_nodes
    train_edge_index = data['author', 'isauthor', 'dataset'].edge_index_train.to(device)

    eval_splits = {}
    for split in ['validation_trans', 'test_trans', 'test_semi', 'test_ind']:
        key = f'edge_label_index_{split}'
        if key in data['author', 'isauthor', 'dataset']:
            eli = data['author', 'isauthor', 'dataset'][key]
            eval_splits[split] = build_eval_pairs(eli)

    def forward_embeddings():
        out_dict = model(x_dict, edge_index_dict)
        return out_dict['author'], out_dict['dataset']

    best_val_metric = -1
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        scorer.train()
        optimizer.zero_grad()

        author_emb_all, ds_emb_all = forward_embeddings()

        pos_src = train_edge_index[0]
        pos_dst = train_edge_index[1]
        pos_scores = scorer(author_emb_all[pos_src], ds_emb_all[pos_dst])

        neg_edge_index = sample_negative_datasets(
            train_edge_index.cpu(), num_author, num_dataset, num_neg_per_pos=args.num_neg
        ).to(device)
        neg_scores = scorer(author_emb_all[neg_edge_index[0]], ds_emb_all[neg_edge_index[1]])

        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        scores = torch.cat([pos_scores, neg_scores])
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(scorer.parameters()), max_norm=5.0
        )
        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            scorer.eval()
            with torch.no_grad():
                author_emb_all, ds_emb_all = forward_embeddings()

            log_line = f'[{args.model}] Epoch {epoch:03d} | loss {loss.item():.4f}'
            if 'validation_trans' in eval_splits:
                score_fn = lambda aid: scorer.score_all(author_emb_all[aid], ds_emb_all)
                val_results = evaluate_ranking_with_scorefn(
                    score_fn, eval_splits['validation_trans'], k_list=(5, 10)
                )
                log_line += f" | val MRR@10 {val_results['MRR@10']:.4f} NDCG@10 {val_results['NDCG@10']:.4f}"

                if val_results['NDCG@10'] > best_val_metric:
                    best_val_metric = val_results['NDCG@10']
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

    if best_state is not None:
        model.load_state_dict(best_state['model'])
        scorer.load_state_dict(best_state['scorer'])

    model.eval()
    scorer.eval()
    with torch.no_grad():
        author_emb_all, ds_emb_all = forward_embeddings()

    print('\n=== Risultati finali ===')
    final_results = {}
    for split in ['test_trans', 'test_semi', 'test_ind']:
        if split in eval_splits:
            score_fn = lambda aid: scorer.score_all(author_emb_all[aid], ds_emb_all)
            res = evaluate_ranking_with_scorefn(score_fn, eval_splits[split], k_list=(5, 10))
            final_results[split] = res
            print(f'[{split}]', {k: round(v, 4) for k, v in res.items()})

    return final_results, model, scorer


# ----------------------------------------------------------------------------
# TRAINING: KGCN (embedding condizionati dalla query, per-batch di autori)
# ----------------------------------------------------------------------------
def train_kgcn(data, args, graph, device):
    node_types = graph['node_types']
    offsets = graph['offsets']
    total_nodes = graph['total_nodes']

    model = KGCNEncoder(
        node_types=node_types,
        in_channels_dict=graph['in_channels_dict'],
        hidden_channels=args.hidden,
        num_relations=graph['num_relations'],
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    x_dict = {nt: graph['x_dict'][nt].to(device) for nt in node_types}
    edge_index = graph['edge_index'].to(device)
    edge_type = graph['edge_type'].to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    author_off = offsets['author']
    ds_off = offsets['dataset']
    num_author = data['author'].num_nodes
    num_dataset = data['dataset'].num_nodes

    train_edge_index = data['author', 'isauthor', 'dataset'].edge_index_train
    train_pairs = build_eval_pairs(train_edge_index)
    train_authors = list(train_pairs.keys())

    eval_splits = {}
    for split in ['validation_trans', 'test_trans', 'test_semi', 'test_ind']:
        key = f'edge_label_index_{split}'
        if key in data['author', 'isauthor', 'dataset']:
            eli = data['author', 'isauthor', 'dataset'][key]
            eval_splits[split] = build_eval_pairs(eli)

    # Sottocampiona la validation per i controlli PERIODICI durante il
    # training: in KGCN ogni valutazione richiede una forward pass
    # full-graph per autore, quindi valutare su TUTTA la validation ogni
    # --eval_every epoche costerebbe piu' del training stesso. La
    # validation COMPLETA (e i test set) vengono comunque valutati una sola
    # volta, a fine training, sul modello migliore.
    val_pairs_full = eval_splits.get('validation_trans', None)
    val_pairs_periodic = val_pairs_full
    if val_pairs_full is not None and len(val_pairs_full) > args.kgcn_val_subsample:
        subsample_keys = random.sample(list(val_pairs_full.keys()), args.kgcn_val_subsample)
        val_pairs_periodic = {k: val_pairs_full[k] for k in subsample_keys}
        print(f'[kgcn] Validation sottocampionata per i controlli periodici: '
              f'{len(val_pairs_periodic)}/{len(val_pairs_full)} autori')

    def score_for_author(author_local_id, x0):
        query_emb = x0[author_off + author_local_id]
        conditioned = model.conditioned_embeddings(x0, edge_index, edge_type, total_nodes, query_emb)
        ds_emb = conditioned[ds_off: ds_off + num_dataset]
        return ds_emb @ query_emb

    best_val_metric = -1
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        x0 = model.base_embeddings(x_dict)

        batch_authors = random.sample(train_authors, min(args.kgcn_batch_authors, len(train_authors)))

        total_loss = 0.0
        for author_id in batch_authors:
            pos_datasets = list(train_pairs[author_id])
            query_emb = x0[author_off + author_id]
            conditioned = model.conditioned_embeddings(x0, edge_index, edge_type, total_nodes, query_emb)
            ds_emb = conditioned[ds_off: ds_off + num_dataset]

            pos_idx = torch.tensor(pos_datasets, dtype=torch.long, device=device)
            neg_idx = torch.randint(0, num_dataset, (len(pos_datasets) * args.num_neg,), device=device)

            pos_scores = ds_emb[pos_idx] @ query_emb
            neg_scores = ds_emb[neg_idx] @ query_emb

            labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
            scores = torch.cat([pos_scores, neg_scores])
            total_loss = total_loss + F.binary_cross_entropy_with_logits(scores, labels)

        loss = total_loss / len(batch_authors)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            log_line = f'[kgcn] Epoch {epoch:03d} | loss {loss.item():.4f}'
            if val_pairs_periodic is not None:
                with torch.no_grad():
                    x0_eval = model.base_embeddings(x_dict)
                    score_fn = lambda aid: score_for_author(aid, x0_eval)
                    val_results = evaluate_ranking_with_scorefn(
                        score_fn, val_pairs_periodic, k_list=(5, 10)
                    )
                log_line += f" | val(sub) MRR@10 {val_results['MRR@10']:.4f} NDCG@10 {val_results['NDCG@10']:.4f}"

                if val_results['NDCG@10'] > best_val_metric:
                    best_val_metric = val_results['NDCG@10']
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
            print(log_line)

            if args.patience > 0 and patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch}')
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    print('\n=== Risultati finali ===')
    final_results = {}
    with torch.no_grad():
        x0_eval = model.base_embeddings(x_dict)
        score_fn = lambda aid: score_for_author(aid, x0_eval)

        if val_pairs_full is not None:
            val_res_full = evaluate_ranking_with_scorefn(score_fn, val_pairs_full, k_list=(5, 10))
            final_results['validation_trans'] = val_res_full
            print('[validation_trans (completa)]', {k: round(v, 4) for k, v in val_res_full.items()})

        for split in ['test_trans', 'test_semi', 'test_ind']:
            if split in eval_splits:
                res = evaluate_ranking_with_scorefn(score_fn, eval_splits[split], k_list=(5, 10))
                final_results[split] = res
                print(f'[{split}]', {k: round(v, 4) for k, v in res.items()})

    return final_results, model, None


# ----------------------------------------------------------------------------
# TRAINING: VGAE (mu/logvar + reparametrizzazione + KL)
# ----------------------------------------------------------------------------
def train_vgae(data, args, graph, device):
    node_types = graph['node_types']
    offsets = graph['offsets']

    model = VGAEEncoder(
        node_types=node_types,
        in_channels_dict=graph['in_channels_dict'],
        hidden_channels=args.hidden,
        num_relations=graph['num_relations'],
        num_bases=min(30, graph['num_relations']),
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    scorer = InnerProductScorer().to(device)

    x_dict = {nt: graph['x_dict'][nt].to(device) for nt in node_types}
    edge_index = graph['edge_index'].to(device)
    edge_type = graph['edge_type'].to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    author_off = offsets['author']
    ds_off = offsets['dataset']
    num_author = data['author'].num_nodes
    num_dataset = data['dataset'].num_nodes

    train_edge_index = data['author', 'isauthor', 'dataset'].edge_index_train.to(device)

    eval_splits = {}
    for split in ['validation_trans', 'test_trans', 'test_semi', 'test_ind']:
        key = f'edge_label_index_{split}'
        if key in data['author', 'isauthor', 'dataset']:
            eli = data['author', 'isauthor', 'dataset'][key]
            eval_splits[split] = build_eval_pairs(eli)

    best_val_metric = -1
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        mu, logvar = model(x_dict, edge_index, edge_type)
        z = model.reparameterize(mu, logvar)
        author_emb_all = z[author_off: author_off + num_author]
        ds_emb_all = z[ds_off: ds_off + num_dataset]

        pos_src = train_edge_index[0]
        pos_dst = train_edge_index[1]
        pos_scores = scorer(author_emb_all[pos_src], ds_emb_all[pos_dst])

        neg_edge_index = sample_negative_datasets(
            train_edge_index.cpu(), num_author, num_dataset, num_neg_per_pos=args.num_neg
        ).to(device)
        neg_scores = scorer(author_emb_all[neg_edge_index[0]], ds_emb_all[neg_edge_index[1]])

        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        scores = torch.cat([pos_scores, neg_scores])
        recon_loss = F.binary_cross_entropy_with_logits(scores, labels)
        kl = model.kl_loss(mu, logvar)
        loss = recon_loss + args.kl_weight * kl

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                mu_eval, _ = model(x_dict, edge_index, edge_type)
                author_emb_all = mu_eval[author_off: author_off + num_author]
                ds_emb_all = mu_eval[ds_off: ds_off + num_dataset]

            log_line = (f'[vgae] Epoch {epoch:03d} | loss {loss.item():.4f} '
                        f'(recon {recon_loss.item():.4f}, kl {kl.item():.4f})')
            if 'validation_trans' in eval_splits:
                score_fn = lambda aid: scorer.score_all(author_emb_all[aid], ds_emb_all)
                val_results = evaluate_ranking_with_scorefn(
                    score_fn, eval_splits['validation_trans'], k_list=(5, 10)
                )
                log_line += f" | val MRR@10 {val_results['MRR@10']:.4f} NDCG@10 {val_results['NDCG@10']:.4f}"

                if val_results['NDCG@10'] > best_val_metric:
                    best_val_metric = val_results['NDCG@10']
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
            print(log_line)

            if args.patience > 0 and patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch}')
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        mu_eval, _ = model(x_dict, edge_index, edge_type)
        author_emb_all = mu_eval[author_off: author_off + num_author]
        ds_emb_all = mu_eval[ds_off: ds_off + num_dataset]

    print('\n=== Risultati finali ===')
    final_results = {}
    for split in ['test_trans', 'test_semi', 'test_ind']:
        if split in eval_splits:
            score_fn = lambda aid: scorer.score_all(author_emb_all[aid], ds_emb_all)
            res = evaluate_ranking_with_scorefn(score_fn, eval_splits[split], k_list=(5, 10))
            final_results[split] = res
            print(f'[{split}]', {k: round(v, 4) for k, v in res.items()})

    return final_results, model, scorer


# ----------------------------------------------------------------------------
# TRAINING: HVGAE (Altaf et al., ICDM 2019), adattato ad AUTORE
# ----------------------------------------------------------------------------
def train_hvgae(data, args, device):
    """
    Training di HVGAE adattato al task author -> dataset. NON usa il grafo
    omogeneo multi-relazionale (build_homogeneous_graph), ma tre input
    analoghi a quelli del paper originale:
      - X_A     : data['author'].x (embedding testuale del fullname, via
                  KeywordEncoder nel loader)
      - A_A,A   : grafo di co-autorialita' (get_coauthorship_edge_index)
      - H_A,D   : bipartito author-dataset, SOLO edge di training
                  (data['author','isauthor','dataset'].edge_index_train)

    Segue Eq. (2)-(3) per l'encoder autore, Eq. (5) per l'encoder dataset,
    Eq. (7)-(8) per i due decoder (ricostruzione di A_A,A e di H_A,D),
    Eq. (13) per la loss (ELBO: ricostruzione - KL) ed Eq. (15) per lo
    scoring finale query-dataset via KL-divergenza tra le due gaussiane.

    Come nello script originale, la query Q e' composta da un solo autore
    (aggregate_query() e' comunque fornita per un uso con |Q|>1).
    """
    num_author = data['author'].num_nodes
    num_dataset = data['dataset'].num_nodes

    x_a = getattr(data['author'], 'x', None)
    if x_a is None:
        x_a = data['author'].net_x
    x_a = x_a.float().to(device)

    coauthorship_edge_index = get_coauthorship_edge_index(data).to(device)
    train_edge_index = data['author', 'isauthor', 'dataset'].edge_index_train.to(device)

    # H_A,D^T di SOLO training (Eq. 5): ogni riga rappresenta un dataset
    # come vettore binario degli autori di training che lo hanno usato.
    H_T = torch.zeros(num_dataset, num_author, device=device)
    H_T[train_edge_index[1], train_edge_index[0]] = 1.0

    author_encoder = HVGAEAuthorEncoder(
        in_channels=x_a.size(-1),
        hidden_channels=args.hidden,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    dataset_encoder = HVGAEDatasetEncoder(
        num_author=num_author,
        hidden_channels=args.hidden,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
    ).to(device)
    decoder = InnerProductScorer().to(device)

    optimizer = torch.optim.Adam(
        list(author_encoder.parameters()) + list(dataset_encoder.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    eval_splits = {}
    for split in ['validation_trans', 'test_trans', 'test_semi', 'test_ind']:
        key = f'edge_label_index_{split}'
        if key in data['author', 'isauthor', 'dataset']:
            eli = data['author', 'isauthor', 'dataset'][key]
            eval_splits[split] = build_eval_pairs(eli)

    def make_score_fn(mu_a, var_a, mu_d, var_d):
        def score_fn(author_id):
            return kl_query_dataset(mu_a[author_id], var_a[author_id], mu_d, var_d)
        return score_fn

    best_val_metric = -1
    best_state = None
    patience_counter = 0

    has_coauthor_edges = coauthorship_edge_index.numel() > 0

    for epoch in range(1, args.epochs + 1):
        author_encoder.train()
        dataset_encoder.train()
        optimizer.zero_grad()

        mu_a, var_a = author_encoder(x_a, coauthorship_edge_index)
        mu_d, var_d = dataset_encoder(H_T)

        z_a = reparameterize_gaussian(mu_a, var_a, author_encoder.training)
        z_d = reparameterize_gaussian(mu_d, var_d, dataset_encoder.training)

        # --- ricostruzione di A_A,A, il grafo di co-autorialita' (Eq. 7) ---
        if has_coauthor_edges:
            pos_src, pos_dst = coauthorship_edge_index
            pos_scores_a = decoder(z_a[pos_src], z_a[pos_dst])
            neg_edge_a = sample_negative_coauthor_pairs(
                coauthorship_edge_index.cpu(), num_author, num_neg_per_pos=args.num_neg
            ).to(device)
            neg_scores_a = decoder(z_a[neg_edge_a[0]], z_a[neg_edge_a[1]])
            labels_a = torch.cat([torch.ones_like(pos_scores_a), torch.zeros_like(neg_scores_a)])
            scores_a = torch.cat([pos_scores_a, neg_scores_a])
            recon_loss_a = F.binary_cross_entropy_with_logits(scores_a, labels_a)
        else:
            # nessuna coppia di coautori osservata nel grafo: il termine di
            # ricostruzione di A_A,A e' nullo (non contribuisce alla loss)
            recon_loss_a = torch.zeros((), device=device)

        # --- ricostruzione di H_A,D, SOLO edge di training (Eq. 8) ---
        pos_scores_h = decoder(z_a[train_edge_index[0]], z_d[train_edge_index[1]])
        neg_edge_h = sample_negative_datasets(
            train_edge_index.cpu(), num_author, num_dataset, num_neg_per_pos=args.num_neg
        ).to(device)
        neg_scores_h = decoder(z_a[neg_edge_h[0]], z_d[neg_edge_h[1]])
        labels_h = torch.cat([torch.ones_like(pos_scores_h), torch.zeros_like(neg_scores_h)])
        scores_h = torch.cat([pos_scores_h, neg_scores_h])
        recon_loss_h = F.binary_cross_entropy_with_logits(scores_h, labels_h)

        # --- termini di KL rispetto a N(0, I), uno per Z_A e uno per Z_D (Eq. 13) ---
        kl_a = kl_to_standard_normal(mu_a, var_a)
        kl_d = kl_to_standard_normal(mu_d, var_d)

        loss = recon_loss_a + recon_loss_h + args.kl_weight * (kl_a + kl_d)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(author_encoder.parameters()) + list(dataset_encoder.parameters()), max_norm=5.0
        )
        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            author_encoder.eval()
            dataset_encoder.eval()
            with torch.no_grad():
                mu_a_eval, var_a_eval = author_encoder(x_a, coauthorship_edge_index)
                mu_d_eval, var_d_eval = dataset_encoder(H_T)

            log_line = (f'[hvgae] Epoch {epoch:03d} | loss {loss.item():.4f} '
                        f'(recon_A {float(recon_loss_a):.4f}, recon_H {recon_loss_h.item():.4f}, '
                        f'kl_a {kl_a.item():.4f}, kl_d {kl_d.item():.4f})')
            if 'validation_trans' in eval_splits:
                score_fn = make_score_fn(mu_a_eval, var_a_eval, mu_d_eval, var_d_eval)
                val_results = evaluate_ranking_with_scorefn(
                    score_fn, eval_splits['validation_trans'], k_list=(5, 10)
                )
                log_line += f" | val MRR@10 {val_results['MRR@10']:.4f} NDCG@10 {val_results['NDCG@10']:.4f}"

                if val_results['NDCG@10'] > best_val_metric:
                    best_val_metric = val_results['NDCG@10']
                    best_state = {
                        'author_encoder': {k: v.clone() for k, v in author_encoder.state_dict().items()},
                        'dataset_encoder': {k: v.clone() for k, v in dataset_encoder.state_dict().items()},
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1
            print(log_line)

            if args.patience > 0 and patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch}')
                break

    if best_state is not None:
        author_encoder.load_state_dict(best_state['author_encoder'])
        dataset_encoder.load_state_dict(best_state['dataset_encoder'])

    author_encoder.eval()
    dataset_encoder.eval()
    with torch.no_grad():
        mu_a_eval, var_a_eval = author_encoder(x_a, coauthorship_edge_index)
        mu_d_eval, var_d_eval = dataset_encoder(H_T)

    print('\n=== Risultati finali ===')
    final_results = {}
    for split in ['test_trans', 'test_semi', 'test_ind']:
        if split in eval_splits:
            score_fn = make_score_fn(mu_a_eval, var_a_eval, mu_d_eval, var_d_eval)
            res = evaluate_ranking_with_scorefn(score_fn, eval_splits[split], k_list=(5, 10))
            final_results[split] = res
            print(f'[{split}]', {k: round(v, 4) for k, v in res.items()})

    return final_results, (author_encoder, dataset_encoder), decoder


# ----------------------------------------------------------------------------
# DISPATCH
# ----------------------------------------------------------------------------
def train_and_evaluate(data: HeteroData, args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f'Device: {device} | Modello: {args.model}')

    if args.model == 'hvgae':
        return train_hvgae(data, args, device)

    if args.model in ('han', 'hgt'):
        graph = build_heterogeneous_graph(data, feature_key='x')
        return train_hetero_model(data, args, graph, device)

    graph = build_homogeneous_graph(data, feature_key='x')

    if args.model in ('rgcn', 'kgat', 'sage', 'gat'):
        return train_static_model(data, args, graph, device)
    elif args.model == 'kgcn':
        return train_kgcn(data, args, graph, device)
    elif args.model == 'vgae':
        return train_vgae(data, args, graph, device)
    else:
        raise ValueError(f'Modello sconosciuto: {args.model}')


# ----------------------------------------------------------------------------
# CONFIGURAZIONE PARAMETRI PER DATASET / MODELLO
# ----------------------------------------------------------------------------
BASE_DEFAULTS = dict(
    hidden=128,
    latent_dim=16,
    num_layers=2,
    dropout=0.2,
    lr=0.005,
    split=0,
    weight_decay=1e-5,
    epochs=100,
    eval_every=5,
    patience=20,
    num_neg=5,
    kl_weight=0.01,
    kgcn_batch_authors=64,
    kgcn_val_subsample=200,
    gat_heads=4,
    han_heads=4,
    hgt_heads=8,
    cpu=False,
    root=None,
)

DATASET_DEFAULTS = {
    'mes': {
        'epochs': 150,
        'eval_every': 5,
        'patience': 20,
        'kgcn_batch_authors': 64,
        'kgcn_val_subsample': 200,
    },
    'pubmed': {
        'epochs': 100,
        'eval_every': 2,
        'patience': 8,
        'kgcn_batch_authors': 128,
        'kgcn_val_subsample': 500,
    },
}

MODEL_DEFAULTS = {
    'rgcn': {
        'num_layers': 2,
        'hidden': 128,
        'lr': 0.001,
        'dropout': 0.2,
    },
    'kgcn': {
        'num_layers': 2,
        'hidden': 64,
        'lr': 0.005,
        'dropout': 0.2,
    },
    'kgat': {
        'num_layers': 3,
        'hidden': 64,
        'lr': 0.0005,
        'dropout': 0.1,
    },
    'vgae': {
        'num_layers': 2,
        'hidden': 128,
        'lr': 0.001,
        'kl_weight': 0.001,
    },
    'hvgae': {
        'num_layers': 2,
        'hidden': 128,
        'latent_dim': 16,
        'lr': 0.005,
        'kl_weight': 0.001,
    },
    'sage': {
        'num_layers': 2,
        'hidden': 128,
        'lr': 0.0001,
        'dropout': 0.2,
    },
    'gat': {
        'num_layers': 2,
        'hidden': 128,
        'lr': 0.0001,
        'dropout': 0.2,
        'gat_heads': 4,
    },
    'han': {
        'hidden': 128,
        'lr': 0.001,
        'dropout': 0.2,
        'han_heads': 4,
    },
    'hgt': {
        'num_layers': 1,
        'hidden': 128,
        'lr': 0.001,
        'hgt_heads': 8,
    },
}

DATASET_MODEL_DEFAULTS = {
    ('pubmed', 'kgcn'): {
        'hidden': 32,
        'num_layers': 2,
        'kgcn_batch_authors': 32,
        'kgcn_val_subsample': 300,
    },
    ('pubmed', 'kgat'): {
        'hidden': 32,
        'num_layers': 2,
    },
    ('pubmed', 'hvgae'): {
        'hidden': 64,
        'latent_dim': 16,
    },
    ('pubmed', 'vgae'): {
        'hidden': 64,
    },
    ('pubmed', 'sage'): {
        'hidden': 64,
    },
    ('pubmed', 'gat'): {
        'hidden': 64,
    },
    ('pubmed', 'han'): {
        'hidden': 64,
    },
    ('pubmed', 'hgt'): {
        'hidden': 64,
    },
}


def load_params(model, dataset, base_args=None):
    """
    Costruisce un argparse.Namespace di iperparametri per la coppia
    (model, dataset), applicando in ordine:
        BASE_DEFAULTS -> DATASET_DEFAULTS[dataset] -> MODEL_DEFAULTS[model]
                       -> DATASET_MODEL_DEFAULTS[(dataset, model)]
    """
    merged = dict(BASE_DEFAULTS)
    if base_args is not None:
        merged.update(vars(base_args))

    merged.update(DATASET_DEFAULTS.get(dataset, {}))
    merged.update(MODEL_DEFAULTS.get(model, {}))
    merged.update(DATASET_MODEL_DEFAULTS.get((dataset, model), {}))

    merged['dataset'] = dataset
    merged['model'] = model
    if merged.get('root') is None:
        merged['root'] = f'datasets/{dataset}'

    args = argparse.Namespace(**merged)

    print(f"[load_params] dataset={dataset} model={model} -> "
          f"hidden={args.hidden}, num_layers={args.num_layers}, lr={args.lr}, "
          f"dropout={args.dropout}, epochs={args.epochs}, eval_every={args.eval_every}, "
          f"patience={args.patience}, kl_weight={args.kl_weight}"
          + (f", kgcn_batch_authors={args.kgcn_batch_authors}, kgcn_val_subsample={args.kgcn_val_subsample}"
             if model == 'kgcn' else "")
          + (f", latent_dim={args.latent_dim}" if model == 'hvgae' else "")
          + (f", gat_heads={args.gat_heads}" if model == 'gat' else "")
          + (f", han_heads={args.han_heads}" if model == 'han' else "")
          + (f", hgt_heads={args.hgt_heads}" if model == 'hgt' else ""))

    return args


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-dataset', type=str, default='mes',
                    help='nome cartella dataset dentro ../datasets/<dataset>')
    p.add_argument('-split', type=float, default=0,
                   help='percentuale (0-100) di nodi dataset a cui azzerare il contenuto '
                        'testuale (data["dataset"].x viene sostituito con un vettore '
                        'costante), per testare la robustezza in assenza di metadati '
                        'testuali')
    p.add_argument('-root', type=str, default=None)
    p.add_argument('-model', type=str, default='hvgae',
                    choices=['rgcn', 'kgcn', 'kgat', 'vgae', 'hvgae', 'sage', 'gat', 'han', 'hgt'])
    p.add_argument('-hidden', type=int, default=128)
    p.add_argument('-latent_dim', type=int, default=16,
                    help='dimensione dello spazio latente per HVGAE (il paper '
                         'originale usa L=16)')
    p.add_argument('-num_layers', type=int, default=2)
    p.add_argument('-dropout', type=float, default=0.2)
    p.add_argument('-lr', type=float, default=0.005)
    p.add_argument('-weight_decay', type=float, default=1e-5)
    p.add_argument('-epochs', type=int, default=100)
    p.add_argument('-eval_every', type=int, default=5)
    p.add_argument('-patience', type=int, default=20)
    p.add_argument('-num_neg', type=int, default=1)
    p.add_argument('-kl_weight', type=float, default=0.01,
                    help='peso del termine di KL-divergence in VGAE/HVGAE')
    p.add_argument('-kgcn_batch_authors', type=int, default=64,
                    help='numero di autori (query) campionati per epoca in KGCN, per '
                         'contenere il costo (ogni query richiede un forward pass '
                         'condizionato sull\'intero grafo)')
    p.add_argument('-kgcn_val_subsample', type=int, default=200,
                    help='numero di autori della validation usati nei controlli '
                         'PERIODICI durante il training di KGCN (early stopping / best '
                         'model selection). La validation completa (e i test set) '
                         'vengono comunque valutati per intero una sola volta, a fine '
                         'training, sul modello migliore.')
    p.add_argument('-gat_heads', type=int, default=4,
                    help='numero di head di attenzione per GAT (hidden deve essere '
                         'divisibile per questo valore)')
    p.add_argument('-han_heads', type=int, default=4,
                    help='numero di head di attenzione node-level per HAN (HANConv)')
    p.add_argument('-hgt_heads', type=int, default=8,
                    help='numero di head di attenzione per HGT (HGTConv)')
    p.add_argument('-cpu', action='store_true')
    return p.parse_args()


import loader_authors as loader


def load_data(dataset, split=0):
    ds = loader.ScholarlyDataset(root=f'datasets/{dataset}/')
    data = ds[0]

    if split and split > 0:
        num_dataset_nodes = data['dataset'].num_nodes
        feat_dim = data['dataset'].x.size(-1)
        n_sampled = int(split / 100 * num_dataset_nodes)
        sampled_indices = torch.randperm(num_dataset_nodes)[:n_sampled]

        data['dataset'].x[sampled_indices] = torch.ones(n_sampled, feat_dim)

        print(f'[load_data] split={split}%: azzerato il testo di '
              f'{n_sampled}/{num_dataset_nodes} nodi dataset')

    return data


if __name__ == '__main__':
    base_args = parse_args()
    set_seed(SEED)

    print()
    print(base_args, flush=True)
    for method in ['kgat']:
        print(f'dataset: {base_args.dataset}', flush=True)
        print(f'method: {method}', flush=True)
        print(f"load method params for dataset: {base_args.dataset}", flush=True)

        args = load_params(method, base_args.dataset, base_args)
        args.model = method
        args.split = base_args.split
        print(f'split: {args.split}')

        # ------------------------------------------------------------------
        # DATA LOADING: usa il tuo ScholarlyDataset esistente (dal loader).
        # ------------------------------------------------------------------
        data = load_data(args.dataset, split=args.split)
        results, model, scorer = train_and_evaluate(data, args)

        print('\nFatto.')