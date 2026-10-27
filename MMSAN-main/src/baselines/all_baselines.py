"""
KGCN / KGAT / VGAE / HVGAE / RGCN / SAGE / GAT / HAN / HGT per dataset
recommendation su ScholarlyDataset.

Adattamento dello script R-GCN fornito: stessa pipeline (loader, metriche,
training loop, split di valutazione), ma con encoder alternativi basati
su Knowledge Graph, su Variational (Graph) Autoencoder e su GNN "classiche"
(GraphSAGE, GAT, HAN, HGT) integrate nella STESSA pipeline unificata invece
di restare script a sé stanti con la propria logica di training/valutazione
(come erano negli script baseline forniti separatamente).

  --model kgcn : Knowledge Graph Convolutional Network (Wang et al., 2019)
                 adattato: la 'publication' gioca il ruolo dello "user" e il
                 'dataset' quello dell'"item". Gli embedding degli item (e di
                 tutti i nodi del KG) vengono aggregati layer-per-layer con
                 pesi di attenzione sulle RELAZIONI, condizionati dalla query
                 (l'embedding della publication). Per efficienza, l'attenzione
                 è calcolata a livello di tipo di relazione (non per singolo
                 vicino), il che permette di ricalcolare gli embedding
                 dell'intero grafo condizionati su una query con un solo
                 einsum per layer, invece di un giro esplicito sui vicini.

  --model kgat : Knowledge Graph Attention Network (Wang et al., 2019)
                 adattato: propagazione ad alto ordine su un unico grafo
                 omogeneo (nodi + relazioni tipizzate), con attenzione in
                 stile TransR (qui alleggerita: trasformazione relazione-
                 specifica come scaling diagonale anziché matrice piena, per
                 tenere sotto controllo il numero di parametri) e aggregatore
                 bi-interaction. L'embedding finale di ogni nodo è la
                 concatenazione degli embedding di tutti i layer di
                 propagazione (come nel paper originale).

  --model vgae : Variational Graph Auto-Encoder (Kipf & Welling, 2016)
                 adattato al grafo eterogeneo multi-relazionale: un encoder
                 relazionale (RGCNConv, come nel tuo script di riferimento)
                 produce media (mu) e log-varianza (logvar) per ogni nodo;
                 in training si campiona z = mu + eps*std (reparametrization
                 trick), in eval si usa direttamente mu. Il decoder è un
                 prodotto scalare puro (inner product, senza matrice
                 apprendibile, come nel paper originale). La loss aggiunge
                 al termine di ricostruzione (BCE su edge positivi/negativi)
                 il termine di KL-divergence rispetto a una gaussiana
                 standard, pesato da --kl_weight.

  --model rgcn : mantenuto identico allo script di riferimento, come baseline.

  --model hvgae : Heterogeneous Variational Graph AutoEncoder (Altaf et al.,
                 ICDM 2019, "Dataset Recommendation via Variational Graph
                 Autoencoder"). A differenza degli altri modelli, NON usa il
                 grafo omogeneo multi-relazionale: segue l'architettura del
                 paper, che usa solo tre input:
                   - X_P: contenuto testuale dei paper (data['publication'].x)
                   - A_P,P: grafo di citazione tra paper
                   - H_P,D: bipartito paper-dataset (solo edge di training)
                 L'encoder dei paper è un GraphSAGE seguito da due head
                 (media, varianza) come in Eq.(2)-(3) del paper; l'encoder dei
                 dataset è un MLP a 2 layer che prende in input la trasposta
                 della matrice bipartita H_P,D (Eq. 5). Il decoder ricostruisce
                 sia A_P,P sia H_P,D con un prodotto scalare (Eq. 7-8). Lo
                 scoring finale query-dataset non è un prodotto scalare ma la
                 KL-divergenza fra le due distribuzioni gaussiane (Eq. 15),
                 coerentemente col fatto che qui gli embedding sono
                 distribuzioni, non vettori singoli.

  --model sage : GraphSAGE (Hamilton et al., 2017) adattato dallo script
                 baseline fornito: opera sullo stesso grafo omogeneo
                 multi-relazionale usato da RGCN/KGAT (costruito con
                 leakage-avoidance da build_homogeneous_graph), ma
                 SAGEConv non è relation-aware, quindi il tipo di relazione
                 viene ignorato (esattamente come nel to_homogeneous() +
                 SAGEConv dello script baseline). Score finale via
                 BilinearScorer, stessa pipeline di training/valutazione di
                 RGCN/KGAT (train_static_model).

  --model gat  : Graph Attention Network (Velickovic et al., 2018), stesso
                 adattamento di sage ma con GATConv multi-head al posto di
                 SAGEConv. Anche qui il tipo di relazione viene ignorato.

  --model han  : Heterogeneous Graph Attention Network (Wang et al., 2019),
                 adattato dallo script baseline fornito. A differenza di
                 RGCN/KGAT/SAGE/GAT, HAN è nativamente eterogeneo: lavora
                 su x_dict/edge_index_dict (non su un grafo omogeneo unico),
                 quindi usa una costruzione dati dedicata
                 (build_heterogeneous_graph) e un training loop dedicato
                 (train_hetero_model), che replica però la stessa logica di
                 leakage-avoidance, stesso negative sampling, stesse metriche
                 e stesso protocollo di early stopping degli altri modelli.

  --model hgt  : Heterogeneous Graph Transformer (Hu et al., 2020), stesso
                 adattamento "nativamente eterogeneo" di han (build_
                 heterogeneous_graph + train_hetero_model), con HGTConv al
                 posto di HANConv, come nello script baseline fornito
                 (proiezione per-tipo con Linear + stack di HGTConv).

Uso:
    python kg_dataset_recommendation.py --model kgcn --dataset mes --epochs 100
    python kg_dataset_recommendation.py --model kgat --dataset mes --epochs 100
    python kg_dataset_recommendation.py --model vgae --dataset mes --epochs 100
    python kg_dataset_recommendation.py --model hvgae --dataset mes --epochs 100
    python kg_dataset_recommendation.py --model sage --dataset mes --epochs 100
    python kg_dataset_recommendation.py --model gat --dataset mes --epochs 100
    python kg_dataset_recommendation.py --model han --dataset mes --epochs 100
    python kg_dataset_recommendation.py --model hgt --dataset mes --epochs 100

Assume che `loader.py` sia importabile come modulo con la classe
ScholarlyDataset (stesso contratto dati dello script R-GCN fornito). Se il
tuo entrypoint differisce, adatta la sezione "DATA LOADING" in fondo al file.
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
# COSTRUZIONE GRAFO OMOGENEO (identica allo script R-GCN di riferimento)
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

    NOTA: questa funzione è usata da RGCN, KGCN, KGAT, SAGE e GAT (e dalla
    vecchia versione di VGAE su grafo eterogeneo). SAGE e GAT non sono
    relation-aware, ma riusano comunque questo grafo (ignorando edge_type)
    per beneficiare della stessa leakage-avoidance e dello stesso set di
    nodi/archi degli altri modelli, invece di ricostruire un
    to_homogeneous() indipendente come nello script baseline originale.
    Il modello HVGAE, seguendo fedelmente il paper di riferimento, NON la
    usa: costruisce invece i propri input (X_P, A_P,P, H_P,D) direttamente
    in train_hvgae(). HAN e HGT, essendo nativamente eterogenei, usano
    invece build_heterogeneous_graph().
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
        if src_type == 'publication' and dst_type == 'dataset' and rel_name == 'cites':
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
    metadata) necessarie per encoder come HAN e HGT che, a differenza di
    RGCN/KGCN/KGAT/SAGE/GAT, non lavorano su un grafo omogeneo unico ma
    su x_dict/edge_index_dict separati per tipo (l'interfaccia nativa di
    HANConv/HGTConv in PyG, la stessa usata negli script baseline forniti
    con data.x_dict/data.edge_index_dict).

    Segue la STESSA logica di leakage-avoidance di build_homogeneous_graph:
    per la relazione publication-cites-dataset si usa SOLO edge_index_train,
    le altre relazioni usano l'edge_index normale. Gli archi vengono resi
    bidirezionali aggiungendo la relazione inversa (rel_name + '_rev'),
    coerentemente con add_relation() di build_homogeneous_graph (necessario
    perché HAN/HGT, come RGCN/KGAT, propagano solo lungo gli archi presenti
    in edge_index_dict).
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
        if src_type == 'publication' and dst_type == 'dataset' and rel_name == 'cites':
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


def get_citation_edge_index(data: HeteroData):
    """
    Estrae il grafo di citazione fra paper A_P,P (Sezione III, Fig. 2 del
    paper HVGAE) dallo HeteroData, cercando la relazione
    publication -> publication. Questa è diversa dalla relazione
    publication -> dataset (H_P,D) usata dagli altri modelli.

    Se il tuo loader chiama diversamente questa relazione (es. un edge_type
    con rel_name != quello atteso), questa funzione la trova comunque perché
    filtra solo su src_type/dst_type == 'publication', non sul nome della
    relazione.
    """
    for et in data.edge_types:
        src_type, rel_name, dst_type = et
        if src_type == 'publication' and dst_type == 'publication':
            ei = data[et].get('edge_index', None)
            if ei is not None and ei.numel() > 0:
                return ei
    raise ValueError(
        "Nessuna relazione publication-publication (grafo di citazione A_P,P) "
        "trovata in HeteroData. Il modello hvgae richiede esplicitamente questo "
        "grafo, in aggiunta al bipartito publication-dataset. Verifica il "
        "contratto dati del tuo loader.py."
    )


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

class BilinearScorer_old(nn.Module):
    """Score(pub, dataset) = pub^T W dataset (DistMult-style, relazione unica)."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, src_emb, dst_emb):
        return (src_emb @ self.weight * dst_emb).sum(dim=-1)

    def score_all(self, src_emb, dst_all_emb):
        return dst_all_emb @ (self.weight.t() @ src_emb)


# ----------------------------------------------------------------------------
# RGCN (baseline, identico allo script di riferimento)
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
    """
    GraphSAGE (Hamilton et al., 2017), adattato dallo script baseline
    fornito (data.to_homogeneous() + SAGEConv). A differenza di RGCN/KGAT,
    SAGEConv non è relation-aware: il tipo di relazione (edge_type) viene
    quindi ignorato, esattamente come nello script baseline. La differenza
    rispetto al baseline è che qui si riusa il grafo omogeneo COSTRUITO CON
    LEAKAGE-AVOIDANCE da build_homogeneous_graph (solo edge_index_train per
    publication-dataset), invece di un to_homogeneous() indipendente che
    userebbe RandomLinkSplit per gestire il leakage.
    """

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
    """
    Graph Attention Network (Velickovic et al., 2018), stesso adattamento di
    SAGEEncoderKG ma con GATConv multi-head al posto di SAGEConv, come nello
    script baseline fornito. Anche qui il tipo di relazione viene ignorato
    (GATConv "vede" un unico grafo omogeneo). Sui layer intermedi le head
    vengono concatenate (concat=True) con out_per_head = hidden/heads, così
    che la dimensione resti hidden_channels ad ogni layer; sull'ultimo layer
    le head vengono mediate (concat=False) per non far esplodere out_dim.
    """

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
# KGCN: aggregazione condizionata dalla query (publication), per tipo di relazione
# ----------------------------------------------------------------------------
class KGCNEncoder(nn.Module):
    """
    Adattamento di KGCN (Wang et al. 2019) al caso publication -> dataset:
      - 'publication' = user, 'dataset' = item, il resto del KG sono entità.
      - Per ogni query (embedding della publication) si ricalcolano gli
        embedding di TUTTI i nodi del grafo, propagando l'informazione dei
        vicini pesata da un'attenzione sul TIPO di relazione:
            alpha_r = softmax_r( rel_emb_r . query )
        Questo evita di iterare esplicitamente sui vicini di ogni nodo: si
        precalcola, per ogni layer, la somma dei messaggi in arrivo per
        (nodo destinazione, tipo di relazione) e poi si combina con i pesi
        alpha_r validi per la query corrente in un solo einsum.
      - Il punteggio finale è il prodotto scalare tra la query e l'embedding
        finale (condizionato) del dataset, come nel KGCN originale
        (score(u,i) = user_emb . item_emb_condizionato_su_u).
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
        """Proiezione iniziale delle feature (indipendente dalla query)."""
        xs = [self.input_proj[nt](x_dict[nt]) for nt in self.node_types]
        return torch.cat(xs, dim=0)

    def conditioned_embeddings(self, x0, edge_index, edge_type, total_nodes, query_emb):
        """
        Ricalcola gli embedding di TUTTI i nodi condizionati su una singola
        query (l'embedding della publication che sta interrogando il grafo).
        query_emb: tensore [H].

        NOTA implementativa: l'attenzione sul tipo di relazione (alpha) dipende
        solo da rel_emb e dalla query, non dalle feature correnti dei nodi:
        viene quindi calcolata una sola volta (non ad ogni layer). Il messaggio
        per ogni nodo destinazione è ottenuto con UNO scatter pesato sugli
        edge (costo O(E*H)) invece di materializzare un tensore denso
        [N, num_relazioni, H] (costo O(N*R*H)), che su grafi con molti nodi
        (es. entità estratte via NER) sarebbe proibitivo in memoria.
        """
        src, dst = edge_index
        rel_scores = self.rel_emb.weight @ query_emb  # [R]
        alpha = F.softmax(rel_scores, dim=0)  # [R]
        edge_weight = alpha[edge_type]  # [E], peso per edge in base al tipo di relazione

        x = x0
        for l in range(self.num_layers):
            weighted_src = x[src] * edge_weight.unsqueeze(-1)  # [E, H]
            message = x.new_zeros(total_nodes, self.hidden_channels)
            message.index_add_(0, dst, weighted_src)  # somma pesata per nodo destinazione

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
    """
    Adattamento di KGAT (Wang et al. 2019): un unico grafo omogeneo (nodi +
    relazioni tipizzate) su cui si propaga con attenzione.

    Semplificazione rispetto al paper originale: la trasformazione
    relazione-specifica W_r (TransR) è implementata come uno scaling
    diagonale (element-wise) invece che come matrice piena hidden x hidden,
    per tenere il numero di parametri sotto controllo quando le relazioni
    sono molte. L'attenzione e l'aggregatore bi-interaction seguono invece
    fedelmente lo schema originale:

        pi(h,r,t) = a^T tanh(W_r e_h + e_r) + (W_r e_t) . tanh(W_r e_h + e_r)
        alpha(h,r,t) = softmax su tutti i vicini di h
        e_Nh = sum alpha(h,r,t) * (W_r e_t)
        e_h^(l+1) = LeakyReLU(W1 (e_h + e_Nh)) + LeakyReLU(W2 (e_h * e_Nh))

    L'embedding finale di ogni nodo è la concatenazione degli embedding di
    tutti i layer (compreso il layer 0), come nel paper originale.
    """

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
        # trasformazione relazione-specifica alleggerita (scaling diagonale, stile TransR)
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
        src, dst = edge_index  # h = src (nodo che aggrega), t = dst (vicino)

        all_layers = [x]
        for l in range(self.num_layers):
            rel_scale = self.rel_transform(edge_type)  # [E, H] scaling diagonale W_r
            rel_vec = self.rel_emb(edge_type)           # [E, H] e_r

            e_h = x[src] * rel_scale   # W_r e_h
            e_t = x[dst] * rel_scale   # W_r e_t

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

        return torch.cat(all_layers, dim=-1)  # [N, hidden*(num_layers+1)]


# ----------------------------------------------------------------------------
# HAN / HGT: encoder nativamente eterogenei (x_dict / edge_index_dict)
# ----------------------------------------------------------------------------
class HANEncoderKG(nn.Module):
    """
    Adattamento di HAN (Wang et al., 2019) allo stesso schema publication/
    dataset degli altri modelli, riprendendo la struttura dello script
    baseline fornito (HANConv + due Linear di output per paper/dataset).
    A differenza di RGCN/KGAT/SAGE/GAT, HANConv lavora nativamente su
    x_dict/edge_index_dict (nessun grafo omogeneo, nessun offset globale):
    l'attenzione a due livelli (node-level + semantic-level, come nel paper
    originale) è già interamente gestita dall'implementazione di PyG.

    A differenza dello script baseline (che esponeva solo 'publication' e
    'dataset' con due Linear dedicate), qui restituiamo l'embedding di
    TUTTI i tipi di nodo (dict), così il chiamante (train_hetero_model) può
    estrarre publication/dataset con la stessa convenzione usata per HGT,
    senza bisogno di head di proiezione ad-hoc nell'encoder.
    """

    def __init__(self, in_channels_dict, hidden_channels, metadata, heads=4, dropout=0.2):
        super().__init__()
        self.han_conv = HANConv(in_channels_dict, hidden_channels, heads=heads,
                                 dropout=dropout, metadata=metadata)

    def forward(self, x_dict, edge_index_dict):
        return self.han_conv(x_dict, edge_index_dict)


class HGTEncoderKG(nn.Module):
    """
    Adattamento di HGT (Hu et al., 2020), come nello script baseline
    fornito: ogni tipo di nodo viene prima proiettato in uno spazio hidden
    comune con una Linear dedicata (self.lin_dict), poi si impilano
    num_layers HGTConv (che gestiscono internamente l'attenzione
    eterogenea tipo-consapevole del paper). Restituisce, come HANEncoderKG,
    l'embedding di TUTTI i tipi di nodo (dict).
    """

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
    """
    Adattamento di VGAE (Kipf & Welling, 2016) al grafo eterogeneo
    multi-relazionale: la parte di encoding usa RGCNConv (stessa scelta del
    tuo script R-GCN di riferimento) per gestire relazioni tipizzate; sopra
    l'ultimo hidden layer condiviso si aggiungono due "head" separate
    (conv_mu, conv_logvar) che producono media e log-varianza per ogni nodo,
    come nel VGAE originale (lì l'encoder era un GCN semplice su grafo
    omogeneo; qui va generalizzato perché abbiamo più tipi di nodo/relazione).
    """

    def __init__(self, node_types, in_channels_dict, hidden_channels,
                 num_relations, num_bases=30, num_layers=2, dropout=0.2):
        super().__init__()
        self.node_types = node_types
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.input_proj = nn.ModuleDict()
        for ntype in node_types:
            self.input_proj[ntype] = nn.Linear(in_channels_dict[ntype], hidden_channels)

        # layer condivisi (tutti tranne l'ultimo): rappresentazione comune
        self.convs = nn.ModuleList()
        for _ in range(max(num_layers - 1, 0)):
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases)
            )

        # ultimo layer sdoppiato in due head, come nel VGAE originale
        self.conv_mu = RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases)
        self.conv_logvar = RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases)

    def forward_old(self, x_dict, edge_index, edge_type):
        xs = [self.input_proj[nt](x_dict[nt]) for nt in self.node_types]
        x = torch.cat(xs, dim=0)

        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_type))
            x = F.dropout(x, p=self.dropout, training=self.training)

        mu = self.conv_mu(x, edge_index, edge_type)
        logvar = self.conv_logvar(x, edge_index, edge_type)
        return mu, logvar

    def forward(self, x_dict, edge_index, edge_type):
        xs = [self.input_proj[nt](x_dict[nt]) for nt in self.node_types]
        x = torch.cat(xs, dim=0)

        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_type))
            x = F.dropout(x, p=self.dropout, training=self.training)

        mu = self.conv_mu(x, edge_index, edge_type)
        logvar = self.conv_logvar(x, edge_index, edge_type)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)   # <-- aggiunto
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # in eval si usa la media, deterministica

    @staticmethod
    def kl_loss(mu, logvar):
        """KL-divergence rispetto a una gaussiana standard N(0, I),
        media sui nodi e somma sulle dimensioni latenti (come in VGAE)."""
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


class InnerProductScorer_old(nn.Module):
    """Decoder a prodotto scalare puro, senza matrice appresa (fedele al
    decoder originale di VGAE: p(A_ij=1) = sigmoid(z_i . z_j)).

    Riusato anche da HVGAE per ricostruire sia A_P,P sia H_P,D (Eq. 7-8 del
    paper Altaf et al.), che nel paper originale sono entrambe definite come
    un prodotto scalare tra embedding seguito da sigmoide."""

    def forward(self, src_emb, dst_emb):
        return (src_emb * dst_emb).sum(dim=-1)

    def score_all(self, src_emb, dst_all_emb):
        return dst_all_emb @ src_emb


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
# HVGAE (Altaf et al., ICDM 2019) - encoder paper (GraphSAGE) + encoder
# dataset (MLP su H^T) + decoder a prodotto scalare + scoring via KL
# ----------------------------------------------------------------------------
class HVGAEPaperEncoder(nn.Module):
    """
    Encoder dei paper del paper HVGAE (Sezione III-A, Eq. 2-3): GraphSAGE
    con k=2 iterazioni di default (Combine=sum, Aggregate=mean, come da
    paper), seguito da due head lineari con attivazione ReLU che producono
    media e varianza della distribuzione gaussiana di ogni paper:

        h_p^k = Combine(h_p^{k-1}, Aggregate({h_p'^{k-1}, p' in N(p)}))
        mu_p     = ReLU(W h_p + b)
        sigma_p^2 = exp(ReLU(W h_p + b))

    NOTA: usiamo SAGEConv di PyG con aggr='mean' come implementazione di
    GraphSAGE (Combine=sum è implicito nella combinazione self+neighbor di
    SAGEConv). Manteniamo esplicitamente la varianza (non il logvar) per
    restare fedeli alla formula sigma^2 = exp(ReLU(.)) del paper, invece di
    riusare la parametrizzazione logvar dello script VGAE di riferimento.
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
        var = torch.exp(F.relu(self.var_head(h)))  # sigma_p^2 = exp(ReLU(.)), Eq. 2
        return mu, var


class HVGAEDatasetEncoder(nn.Module):
    """
    Encoder dei dataset del paper HVGAE (Eq. 5): MLP densamente connesso a
    due layer che prende in input H_P,D^T, cioè per ogni dataset il vettore
    binario delle publication (di training) che lo usano:

        h_d = MLP(H_P,D^T)
        mu_d      = ReLU(W h_d + b)
        sigma_d^2 = exp(ReLU(W h_d + b))

    mu_d e sigma_d^2 condividono i parametri del backbone MLP a due layer e
    sono poi prodotti da due head separate, esattamente come nel paper.
    """

    def __init__(self, num_pub, hidden_channels, latent_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(num_pub, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = dropout

        self.mu_head = nn.Linear(hidden_channels, latent_dim)
        self.var_head = nn.Linear(hidden_channels, latent_dim)

    def forward(self, H_T):
        h = F.relu(self.fc1(H_T))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.fc2(h))

        mu = F.relu(self.mu_head(h))
        var = torch.exp(F.relu(self.var_head(h)))  # sigma_d^2 = exp(ReLU(.)), Eq. 5
        return mu, var


def reparameterize_gaussian(mu, var, training):
    """Reparametrization trick (Kingma & Welling, 2014) nella parametrizzazione
    a varianza esplicita usata dal paper HVGAE: z = mu + eps * sigma,
    con eps ~ N(0, I). In eval si usa direttamente mu (deterministico)."""
    if training:
        std = torch.sqrt(var + 1e-8)
        eps = torch.randn_like(std)
        return mu + eps * std
    return mu


def kl_to_standard_normal(mu, var):
    """KL[N(mu, diag(var)) || N(0, I)], media sui nodi e somma sulle
    dimensioni latenti (termine di regolarizzazione dell'ELBO, Eq. 13)."""
    logvar = torch.log(var + 1e-8)
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - var, dim=1))


def kl_query_dataset(mu_q, var_q, mu_d, var_d):
    """
    Punteggio di rilevanza query-dataset s(Q, d) del paper HVGAE (Eq. 15):

        s(Q, d) = log(sigma_d / sigma_Q) + (sigma_Q^2 + (mu_Q - mu_d)^2) / (2 sigma_d^2) - 1/2

    sommato sulle dimensioni latenti (KL-divergenza multivariata a
    covarianza diagonale). Più BASSA è la KL, più il dataset è rilevante
    (Sezione III-C: si selezionano i top-k dataset con s(Q,d) più piccola).

    Qui mu_q, var_q sono i parametri della distribuzione di UNA query
    (già aggregata secondo Eq. 14 se |Q|>1, oppure la distribuzione di una
    singola publication se la si usa direttamente come query, come nella
    valutazione per-publication di questo script), con shape [H].
    mu_d, var_d hanno shape [num_dataset, H] (tutti i dataset candidati).

    Ritorna il punteggio NEGATO ([num_dataset]), perché
    evaluate_ranking_with_scorefn ordina i punteggi in modo DECRESCENTE
    (punteggio più alto = più rilevante): negando la KL, il dataset con KL
    più piccola (più rilevante secondo il paper) ottiene il punteggio più
    alto, mantenendo la stessa convenzione di score_all() usata dagli altri
    modelli (RGCN, KGAT, KGCN, VGAE).
    """
    sigma_q = torch.sqrt(var_q + 1e-8)          # [H]
    sigma_d = torch.sqrt(var_d + 1e-8)          # [num_dataset, H]

    term1 = torch.log(sigma_d / sigma_q.unsqueeze(0) + 1e-8)
    term2 = (var_q.unsqueeze(0) + (mu_q.unsqueeze(0) - mu_d).pow(2)) / (2 * var_d + 1e-8)
    kl = (term1 + term2 - 0.5).sum(dim=-1)      # [num_dataset]
    return -kl


def aggregate_query(mu_p_query, var_p_query):
    """
    Rappresentazione della query come distribuzione gaussiana (Eq. 14):
    media element-wise di mu e di sigma^2 sui paper della query.
    mu_p_query, var_p_query: [|Q|, H] (embedding dei paper che compongono Q).
    Ritorna (mu_Q, var_Q) con shape [H].
    """
    mu_Q = mu_p_query.mean(dim=0)
    var_Q = var_p_query.mean(dim=0)
    return mu_Q, var_Q


# ----------------------------------------------------------------------------
# Sampling negativi (identico allo script di riferimento, + variante per A_P,P)
# ----------------------------------------------------------------------------
def sample_negative_datasets(pos_edge_index, num_pub, num_dataset, num_neg_per_pos=1):
    E = pos_edge_index.size(1)
    neg_dst = torch.randint(0, num_dataset, (E * num_neg_per_pos,))
    neg_src = pos_edge_index[0].repeat_interleave(num_neg_per_pos)
    return torch.stack([neg_src, neg_dst], dim=0)


def sample_negative_citations(pos_edge_index, num_pub, num_neg_per_pos=1):
    """Negativi per la ricostruzione di A_P,P (usata da HVGAE): per ogni
    edge di citazione positivo (p_i -> p_j), campiona un p_j' casuale come
    destinazione negativa (stesso schema di sample_negative_datasets)."""
    E = pos_edge_index.size(1)
    neg_dst = torch.randint(0, num_pub, (E * num_neg_per_pos,))
    neg_src = pos_edge_index[0].repeat_interleave(num_neg_per_pos)
    return torch.stack([neg_src, neg_dst], dim=0)


# ----------------------------------------------------------------------------
# METRICHE DI RANKING: MRR, NDCG@k, Recall@k (identiche allo script di riferimento)
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
    """Da edge_index [2, E] (pub, dataset) costruisce dict pub -> set(dataset)."""
    pairs = defaultdict(set)
    src = edge_label_index[0].tolist()
    dst = edge_label_index[1].tolist()
    for s, d in zip(src, dst):
        pairs[s].add(d)
    return pairs


def evaluate_ranking_with_scorefn(score_fn, eval_pairs, k_list=(5, 10)):
    """
    Versione generalizzata della valutazione: invece di richiedere una
    matrice fissa di embedding per i dataset, riceve una funzione
    `score_fn(pub_id) -> tensor [num_dataset]` con i punteggi di quella
    publication contro tutti i dataset. Questo permette di riusare la stessa
    funzione di valutazione sia per modelli a embedding statici (RGCN, KGAT,
    SAGE, GAT, HAN, HGT) sia per modelli query-dipendenti come KGCN, sia per
    modelli basati su KL-divergenza come HVGAE (che restituisce -KL come
    punteggio, si veda kl_query_dataset).
    """
    mrr_at = {k: [] for k in k_list}
    ndcg_at = {k: [] for k in k_list}
    recall_at = {k: [] for k in k_list}

    with torch.no_grad():
        for pub_id, relevant_set in eval_pairs.items():
            if len(relevant_set) == 0:
                continue

            scores = score_fn(pub_id)
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

    pub_off = offsets['publication']
    ds_off = offsets['dataset']
    num_pub = data['publication'].num_nodes
    num_dataset = data['dataset'].num_nodes

    train_edge_index = data['publication', 'cites', 'dataset'].edge_index_train.to(device)

    eval_splits = {}
    for split in ['validation_trans', 'test_trans', 'test_semi', 'test_ind']:
        key = f'edge_label_index_{split}'
        if key in data['publication', 'cites', 'dataset']:
            eli = data['publication', 'cites', 'dataset'][key]
            eval_splits[split] = build_eval_pairs(eli)

    def forward_embeddings():
        if args.model == 'rgcn':
            return model(x_dict, edge_index, edge_type)
        elif args.model in ('sage', 'gat'):
            # SAGEConv/GATConv non sono relation-aware: edge_type ignorato
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
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(scorer.parameters()), max_norm=5.0
        )
        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            scorer.eval()
            with torch.no_grad():
                emb = forward_embeddings()
                pub_emb_all = emb[pub_off: pub_off + num_pub]
                ds_emb_all = emb[ds_off: ds_off + num_dataset]

            log_line = f'[{args.model}] Epoch {epoch:03d} | loss {loss.item():.4f}'
            if 'validation_trans' in eval_splits:
                score_fn = lambda pid: scorer.score_all(pub_emb_all[pid], ds_emb_all)
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
        pub_emb_all = emb[pub_off: pub_off + num_pub]
        ds_emb_all = emb[ds_off: ds_off + num_dataset]

    print('\n=== Risultati finali ===')
    final_results = {}
    for split in ['test_trans', 'test_semi', 'test_ind']:
        if split in eval_splits:
            score_fn = lambda pid: scorer.score_all(pub_emb_all[pid], ds_emb_all)
            res = evaluate_ranking_with_scorefn(score_fn, eval_splits[split], k_list=(5, 10))
            final_results[split] = res
            print(f'[{split}]', {k: round(v, 4) for k, v in res.items()})

    return final_results, model, scorer


# ----------------------------------------------------------------------------
# TRAINING: modelli nativamente eterogenei (HAN, HGT)
# ----------------------------------------------------------------------------
def train_hetero_model(data, args, graph, device):
    """
    Training loop per HAN e HGT, che lavorano su x_dict/edge_index_dict
    (costruiti da build_heterogeneous_graph) invece che sul grafo omogeneo
    unico usato da RGCN/KGAT/SAGE/GAT. A parte questo, ricalca esattamente
    train_static_model: stesso negative sampling, stesso BilinearScorer,
    stesse metriche, stesso protocollo di best-model-selection su
    validation_trans e stesso early stopping.
    """
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

    num_pub = data['publication'].num_nodes
    num_dataset = data['dataset'].num_nodes
    train_edge_index = data['publication', 'cites', 'dataset'].edge_index_train.to(device)

    eval_splits = {}
    for split in ['validation_trans', 'test_trans', 'test_semi', 'test_ind']:
        key = f'edge_label_index_{split}'
        if key in data['publication', 'cites', 'dataset']:
            eli = data['publication', 'cites', 'dataset'][key]
            eval_splits[split] = build_eval_pairs(eli)

    def forward_embeddings():
        out_dict = model(x_dict, edge_index_dict)
        return out_dict['publication'], out_dict['dataset']

    best_val_metric = -1
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        scorer.train()
        optimizer.zero_grad()

        pub_emb_all, ds_emb_all = forward_embeddings()

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
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(scorer.parameters()), max_norm=5.0
        )
        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            scorer.eval()
            with torch.no_grad():
                pub_emb_all, ds_emb_all = forward_embeddings()

            log_line = f'[{args.model}] Epoch {epoch:03d} | loss {loss.item():.4f}'
            if 'validation_trans' in eval_splits:
                score_fn = lambda pid: scorer.score_all(pub_emb_all[pid], ds_emb_all)
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
        pub_emb_all, ds_emb_all = forward_embeddings()

    print('\n=== Risultati finali ===')
    final_results = {}
    for split in ['test_trans', 'test_semi', 'test_ind']:
        if split in eval_splits:
            score_fn = lambda pid: scorer.score_all(pub_emb_all[pid], ds_emb_all)
            res = evaluate_ranking_with_scorefn(score_fn, eval_splits[split], k_list=(5, 10))
            final_results[split] = res
            print(f'[{split}]', {k: round(v, 4) for k, v in res.items()})

    return final_results, model, scorer


# ----------------------------------------------------------------------------
# TRAINING: KGCN (embedding condizionati dalla query, per-batch di publication)
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

    pub_off = offsets['publication']
    ds_off = offsets['dataset']
    num_pub = data['publication'].num_nodes
    num_dataset = data['dataset'].num_nodes

    train_edge_index = data['publication', 'cites', 'dataset'].edge_index_train
    # raggruppa gli edge di training per publication (query)
    train_pairs = build_eval_pairs(train_edge_index)
    train_pubs = list(train_pairs.keys())

    eval_splits = {}
    for split in ['validation_trans', 'test_trans', 'test_semi', 'test_ind']:
        key = f'edge_label_index_{split}'
        if key in data['publication', 'cites', 'dataset']:
            eli = data['publication', 'cites', 'dataset'][key]
            eval_splits[split] = build_eval_pairs(eli)

    # Sottocampiona la validation per i controlli PERIODICI durante il training:
    # in KGCN ogni valutazione richiede una forward pass full-graph per
    # publication, quindi valutare su TUTTA la validation ogni --eval_every
    # epoche costerebbe più del training stesso. Il sottoinsieme è fissato
    # una volta sola (stesso seed) così il segnale di early-stopping resta
    # comparabile epoca dopo epoca. La validation COMPLETA (e i test set)
    # vengono comunque valutati una sola volta, a fine training, sul modello
    # migliore selezionato dai controlli periodici.
    val_pairs_full = eval_splits.get('validation_trans', None)
    val_pairs_periodic = val_pairs_full
    if val_pairs_full is not None and len(val_pairs_full) > args.kgcn_val_subsample:
        subsample_keys = random.sample(list(val_pairs_full.keys()), args.kgcn_val_subsample)
        val_pairs_periodic = {k: val_pairs_full[k] for k in subsample_keys}
        print(f'[kgcn] Validation sottocampionata per i controlli periodici: '
              f'{len(val_pairs_periodic)}/{len(val_pairs_full)} publication')

    def score_for_pub(pub_local_id, x0):
        """Ricalcola gli embedding condizionati sulla query pub e ritorna
        i punteggi contro tutti i dataset."""
        query_emb = x0[pub_off + pub_local_id]
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

        # mini-batch di publication di training per limitare il costo
        # (ogni publication richiede un forward pass condizionato sull'intero grafo)
        batch_pubs = random.sample(train_pubs, min(args.kgcn_batch_pubs, len(train_pubs)))

        total_loss = 0.0
        for pub_id in batch_pubs:
            pos_datasets = list(train_pairs[pub_id])
            query_emb = x0[pub_off + pub_id]
            conditioned = model.conditioned_embeddings(x0, edge_index, edge_type, total_nodes, query_emb)
            ds_emb = conditioned[ds_off: ds_off + num_dataset]

            pos_idx = torch.tensor(pos_datasets, dtype=torch.long, device=device)
            neg_idx = torch.randint(0, num_dataset, (len(pos_datasets) * args.num_neg,), device=device)

            pos_scores = ds_emb[pos_idx] @ query_emb
            neg_scores = ds_emb[neg_idx] @ query_emb

            labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
            scores = torch.cat([pos_scores, neg_scores])
            total_loss = total_loss + F.binary_cross_entropy_with_logits(scores, labels)

        loss = total_loss / len(batch_pubs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            log_line = f'[kgcn] Epoch {epoch:03d} | loss {loss.item():.4f}'
            if val_pairs_periodic is not None:
                with torch.no_grad():
                    x0_eval = model.base_embeddings(x_dict)
                    score_fn = lambda pid: score_for_pub(pid, x0_eval)
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
        score_fn = lambda pid: score_for_pub(pid, x0_eval)

        # valutazione COMPLETA della validation, una sola volta, a fine training
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
    total_nodes = graph['total_nodes']

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

    pub_off = offsets['publication']
    ds_off = offsets['dataset']
    num_pub = data['publication'].num_nodes
    num_dataset = data['dataset'].num_nodes

    train_edge_index = data['publication', 'cites', 'dataset'].edge_index_train.to(device)

    eval_splits = {}
    for split in ['validation_trans', 'test_trans', 'test_semi', 'test_ind']:
        key = f'edge_label_index_{split}'
        if key in data['publication', 'cites', 'dataset']:
            eli = data['publication', 'cites', 'dataset'][key]
            eval_splits[split] = build_eval_pairs(eli)

    best_val_metric = -1
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        mu, logvar = model(x_dict, edge_index, edge_type)
        z = model.reparameterize(mu, logvar)  # campionamento in training
        pub_emb_all = z[pub_off: pub_off + num_pub]
        ds_emb_all = z[ds_off: ds_off + num_dataset]

        pos_src = train_edge_index[0]
        pos_dst = train_edge_index[1]
        pos_scores = scorer(pub_emb_all[pos_src], ds_emb_all[pos_dst])

        neg_edge_index = sample_negative_datasets(
            train_edge_index.cpu(), num_pub, num_dataset, num_neg_per_pos=args.num_neg
        ).to(device)
        neg_scores = scorer(pub_emb_all[neg_edge_index[0]], ds_emb_all[neg_edge_index[1]])

        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        scores = torch.cat([pos_scores, neg_scores])
        recon_loss = F.binary_cross_entropy_with_logits(scores, labels)
        kl = model.kl_loss(mu, logvar)
        loss = recon_loss + args.kl_weight * kl

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # <-- aggiunto
        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                mu_eval, _ = model(x_dict, edge_index, edge_type)
                pub_emb_all = mu_eval[pub_off: pub_off + num_pub]  # in eval si usa mu (deterministico)
                ds_emb_all = mu_eval[ds_off: ds_off + num_dataset]

            log_line = (f'[vgae] Epoch {epoch:03d} | loss {loss.item():.4f} '
                        f'(recon {recon_loss.item():.4f}, kl {kl.item():.4f})')
            if 'validation_trans' in eval_splits:
                score_fn = lambda pid: scorer.score_all(pub_emb_all[pid], ds_emb_all)
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
        pub_emb_all = mu_eval[pub_off: pub_off + num_pub]
        ds_emb_all = mu_eval[ds_off: ds_off + num_dataset]

    print('\n=== Risultati finali ===')
    final_results = {}
    for split in ['test_trans', 'test_semi', 'test_ind']:
        if split in eval_splits:
            score_fn = lambda pid: scorer.score_all(pub_emb_all[pid], ds_emb_all)
            res = evaluate_ranking_with_scorefn(score_fn, eval_splits[split], k_list=(5, 10))
            final_results[split] = res
            print(f'[{split}]', {k: round(v, 4) for k, v in res.items()})

    return final_results, model, scorer


# ----------------------------------------------------------------------------
# TRAINING: HVGAE (Altaf et al., ICDM 2019)
# ----------------------------------------------------------------------------
def train_hvgae(data, args, device):
    """
    Training di HVGAE, fedele all'architettura del paper: NON usa il grafo
    omogeneo multi-relazionale (build_homogeneous_graph), ma solo i tre
    input citati nel paper:
      - X_P     : data['publication'].x (contenuto testuale dei paper)
      - A_P,P   : grafo di citazione tra paper (get_citation_edge_index)
      - H_P,D   : bipartito paper-dataset, SOLO edge di training
                  (data['publication','cites','dataset'].edge_index_train)

    Segue Eq. (2)-(3) per l'encoder dei paper, Eq. (5) per l'encoder dei
    dataset, Eq. (7)-(8) per i due decoder (ricostruzione di A_P,P e di
    H_P,D), Eq. (13) per la loss (ELBO: ricostruzione - KL) ed Eq. (15) per
    lo scoring finale query-dataset via KL-divergenza tra le due gaussiane.

    Per coerenza con il resto dello script (dove ogni "query" di
    valutazione è una singola publication, si veda build_eval_pairs), la
    query Q qui è composta da un solo paper: l'aggregazione di Eq. (14)
    collassa quindi banalmente su mu_p, sigma_p^2 di quella publication
    (aggregate_query() è comunque fornita per un uso con |Q|>1).
    """
    num_pub = data['publication'].num_nodes
    num_dataset = data['dataset'].num_nodes

    x_p = getattr(data['publication'], 'x', None)
    if x_p is None:
        x_p = data['publication'].net_x
    x_p = x_p.float().to(device)

    citation_edge_index = get_citation_edge_index(data).to(device)
    train_edge_index = data['publication', 'cites', 'dataset'].edge_index_train.to(device)

    # H_P,D^T di SOLO training (Eq. 5): ogni riga rappresenta un dataset
    # come vettore binario delle publication di training che lo usano.
    # NOTA: matrice densa [num_dataset, num_pub]; per grafi molto grandi
    # conviene sostituirla con una rappresentazione sparsa (torch.sparse),
    # ma la teniamo densa per fedeltà diretta a MLP(H_P,D^T) del paper.
    H_T = torch.zeros(num_dataset, num_pub, device=device)
    H_T[train_edge_index[1], train_edge_index[0]] = 1.0

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
    decoder = InnerProductScorer().to(device)

    optimizer = torch.optim.Adam(
        list(paper_encoder.parameters()) + list(dataset_encoder.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    eval_splits = {}
    for split in ['validation_trans', 'test_trans', 'test_semi', 'test_ind']:
        key = f'edge_label_index_{split}'
        if key in data['publication', 'cites', 'dataset']:
            eli = data['publication', 'cites', 'dataset'][key]
            eval_splits[split] = build_eval_pairs(eli)

    def make_score_fn(mu_p, var_p, mu_d, var_d):
        def score_fn(pub_id):
            # query = singola publication (Eq. 14 con |Q|=1)
            return kl_query_dataset(mu_p[pub_id], var_p[pub_id], mu_d, var_d)
        return score_fn

    best_val_metric = -1
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        paper_encoder.train()
        dataset_encoder.train()
        optimizer.zero_grad()

        mu_p, var_p = paper_encoder(x_p, citation_edge_index)
        mu_d, var_d = dataset_encoder(H_T)

        z_p = reparameterize_gaussian(mu_p, var_p, paper_encoder.training)
        z_d = reparameterize_gaussian(mu_d, var_d, dataset_encoder.training)

        # --- ricostruzione di A_P,P, il grafo di citazione (Eq. 7) ---
        pos_src, pos_dst = citation_edge_index
        pos_scores_a = decoder(z_p[pos_src], z_p[pos_dst])
        neg_edge_a = sample_negative_citations(
            citation_edge_index.cpu(), num_pub, num_neg_per_pos=args.num_neg
        ).to(device)
        neg_scores_a = decoder(z_p[neg_edge_a[0]], z_p[neg_edge_a[1]])
        labels_a = torch.cat([torch.ones_like(pos_scores_a), torch.zeros_like(neg_scores_a)])
        scores_a = torch.cat([pos_scores_a, neg_scores_a])
        recon_loss_a = F.binary_cross_entropy_with_logits(scores_a, labels_a)

        # --- ricostruzione di H_P,D, SOLO edge di training (Eq. 8) ---
        pos_scores_h = decoder(z_p[train_edge_index[0]], z_d[train_edge_index[1]])
        neg_edge_h = sample_negative_datasets(
            train_edge_index.cpu(), num_pub, num_dataset, num_neg_per_pos=args.num_neg
        ).to(device)
        neg_scores_h = decoder(z_p[neg_edge_h[0]], z_d[neg_edge_h[1]])
        labels_h = torch.cat([torch.ones_like(pos_scores_h), torch.zeros_like(neg_scores_h)])
        scores_h = torch.cat([pos_scores_h, neg_scores_h])
        recon_loss_h = F.binary_cross_entropy_with_logits(scores_h, labels_h)

        # --- termini di KL rispetto a N(0, I), uno per Z_P e uno per Z_D (Eq. 13) ---
        kl_p = kl_to_standard_normal(mu_p, var_p)
        kl_d = kl_to_standard_normal(mu_d, var_d)

        loss = recon_loss_a + recon_loss_h + args.kl_weight * (kl_p + kl_d)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(paper_encoder.parameters()) + list(dataset_encoder.parameters()), max_norm=5.0
        )
        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            paper_encoder.eval()
            dataset_encoder.eval()
            with torch.no_grad():
                mu_p_eval, var_p_eval = paper_encoder(x_p, citation_edge_index)
                mu_d_eval, var_d_eval = dataset_encoder(H_T)

            log_line = (f'[hvgae] Epoch {epoch:03d} | loss {loss.item():.4f} '
                        f'(recon_A {recon_loss_a.item():.4f}, recon_H {recon_loss_h.item():.4f}, '
                        f'kl_p {kl_p.item():.4f}, kl_d {kl_d.item():.4f})')
            if 'validation_trans' in eval_splits:
                score_fn = make_score_fn(mu_p_eval, var_p_eval, mu_d_eval, var_d_eval)
                val_results = evaluate_ranking_with_scorefn(
                    score_fn, eval_splits['validation_trans'], k_list=(5, 10)
                )
                log_line += f" | val MRR@10 {val_results['MRR@10']:.4f} NDCG@10 {val_results['NDCG@10']:.4f}"

                if val_results['NDCG@10'] > best_val_metric:
                    best_val_metric = val_results['NDCG@10']
                    best_state = {
                        'paper_encoder': {k: v.clone() for k, v in paper_encoder.state_dict().items()},
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
        paper_encoder.load_state_dict(best_state['paper_encoder'])
        dataset_encoder.load_state_dict(best_state['dataset_encoder'])

    paper_encoder.eval()
    dataset_encoder.eval()
    with torch.no_grad():
        mu_p_eval, var_p_eval = paper_encoder(x_p, citation_edge_index)
        mu_d_eval, var_d_eval = dataset_encoder(H_T)

    print('\n=== Risultati finali ===')
    final_results = {}
    for split in ['test_trans', 'test_semi', 'test_ind']:
        if split in eval_splits:
            score_fn = make_score_fn(mu_p_eval, var_p_eval, mu_d_eval, var_d_eval)
            res = evaluate_ranking_with_scorefn(score_fn, eval_splits[split], k_list=(5, 10))
            final_results[split] = res
            print(f'[{split}]', {k: round(v, 4) for k, v in res.items()})

    return final_results, (paper_encoder, dataset_encoder), decoder


# ----------------------------------------------------------------------------
# DISPATCH
# ----------------------------------------------------------------------------
def train_and_evaluate(data: HeteroData, args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f'Device: {device} | Modello: {args.model}')

    if args.model == 'hvgae':
        # HVGAE non usa il grafo omogeneo multi-relazionale: costruisce i
        # propri input (X_P, A_P,P, H_P,D di training) direttamente.
        return train_hvgae(data, args, device)

    if args.model in ('han', 'hgt'):
        # HAN/HGT sono nativamente eterogenei: usano x_dict/edge_index_dict
        # invece del grafo omogeneo unico usato da RGCN/KGAT/SAGE/GAT.
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
# Idea: un set di iperparametri "di base", corretto poi da eventuali
# specializzazioni per DATASET (dipende solo dalla scala del grafo: mes è
# poche migliaia di nodi/archi, pubmed ha ~300k archi pub-dataset), da
# specializzazioni per MODELLO (dipende dall'architettura: es. KGCN fa un
# forward pass full-graph per ogni query, KGAT concatena gli embedding di
# tutti i layer, HAN/HGT hanno il proprio numero di head di attenzione,
# ecc.) e infine da eventuali override specifici per la combinazione
# (dataset, modello), che hanno sempre la priorità più alta perché sono i
# più mirati.
#
# Ordine di merge (dal meno al più specifico):
#   BASE_DEFAULTS -> DATASET_DEFAULTS[dataset] -> MODEL_DEFAULTS[model]
#                  -> DATASET_MODEL_DEFAULTS[(dataset, model)]
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
    kgcn_batch_pubs=64,
    kgcn_val_subsample=200,
    gat_heads=4,
    han_heads=4,
    hgt_heads=8,
    cpu=False,
    root=None,
)

# --- specializzazioni per DATASET (dipendono dalla scala del grafo) -------
DATASET_DEFAULTS = {
    'mes': {
        # poche migliaia di nodi/archi: training full-batch è economico,
        # ci si può permettere più epoche e una validation valutata spesso
        'epochs': 150,
        'eval_every': 5,
        'patience': 20,
        'kgcn_batch_pubs': 64,     # può coprire quasi tutte le publication di training
        'kgcn_val_subsample': 200,  # spesso >= alla validation intera: nessun sottocampionamento reale
    },
    'pubmed': {
        # ~300k archi pub-dataset: ogni epoca (e ogni forward full-graph)
        # costa molto di più, quindi meno epoche ma controllate più spesso
        # in valutazione "leggera" (soprattutto per KGCN); patience più
        # aggressiva per non sprecare tempo se non migliora.
        'epochs': 100,
        'eval_every': 2,
        'patience': 8,
        'kgcn_batch_pubs': 128,
        'kgcn_val_subsample': 500,
    },
}

# --- specializzazioni per MODELLO (dipendono dall'architettura) -----------
MODEL_DEFAULTS = {
    'rgcn': {
        'num_layers': 2,
        'hidden': 128,
        'lr': 0.001,
        'dropout': 0.2,
    },
    'kgcn': {
        # ogni query = un forward pass condizionato sull'intero grafo:
        # hidden più piccolo per contenere il costo per query
        'num_layers': 2,
        'hidden': 64,
        'lr': 0.005,
        'dropout': 0.2,
    },
    'kgat': {
        # l'embedding finale concatena hidden*(num_layers+1): con hidden
        # grande e più layer la dimensione esplode in fretta, quindi hidden
        # più contenuto; il bi-interaction aggregator è più instabile in
        # training, quindi lr più basso rispetto a RGCN/VGAE
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
        # il paper usa esplicitamente uno spazio latente piccolo (L=16)
        'num_layers': 2,
        'hidden': 128,
        'latent_dim': 16,
        'lr': 0.005,
        'kl_weight': 0.001,
    },
    'sage': {
        # nello script baseline il lr usato era molto conservativo
        # (1e-5, probabilmente per il grafo COMPLETO non filtrato dal
        # leakage): qui, essendo il training già protetto da
        # build_homogeneous_graph, un lr un po' più alto converge meglio.
        # Riportalo a 1e-5 se osservi instabilità.
        'num_layers': 2,
        'hidden': 128,
        'lr': 0.0001,
        'dropout': 0.2,
    },
    'gat': {
        # stesso discorso di sage per il lr; heads=4 come valore di
        # partenza ragionevole (hidden deve essere divisibile per heads).
        'num_layers': 2,
        'hidden': 128,
        'lr': 0.0001,
        'dropout': 0.2,
        'gat_heads': 4,
    },
    'han': {
        # HANConv gestisce già attenzione a due livelli (nodo + semantica):
        # niente num_layers esplicito, la profondità è fissata a un solo
        # HANConv come nello script baseline fornito.
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

# --- override mirati per combinazioni (dataset, modello) specifiche -------
# Qui il criterio guida è: su pubmed, i modelli che ricalcolano embedding
# per query (kgcn) o che tengono in memoria strutture dense proporzionali
# al numero di publication (hvgae, la cui H_T è [num_dataset, num_pub])
# vanno ridimensionati più aggressivamente degli altri. Lo stesso vale per
# HAN/HGT su pubmed, dove il numero di archi eterogenei cresce molto.
DATASET_MODEL_DEFAULTS = {
    ('pubmed', 'kgcn'): {
        'hidden': 32,
        'num_layers': 2,
        'kgcn_batch_pubs': 32,   # meno query per epoca: ognuna è un forward full-graph
        'kgcn_val_subsample': 300,
    },
    ('pubmed', 'kgat'): {
        'hidden': 32,
        'num_layers': 2,
    },
    ('pubmed', 'hvgae'): {
        # H_T è densa [num_dataset, num_pub]: con ~300k archi pub-dataset,
        # num_pub può essere grande. Riduciamo hidden/latent per contenere
        # la MLP che la consuma (fc1: Linear(num_pub, hidden)).
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
    Ogni livello sovrascrive solo le chiavi che definisce esplicitamente;
    tutto il resto resta quello del livello precedente.

    Se viene passato `base_args` (es. il Namespace prodotto da parse_args()
    con eventuali override da riga di comando come --cpu o --root), i suoi
    campi vengono copiati per primi cosicché restino ATTIVI anche dopo il
    merge, a meno che il dataset/modello non li sovrascriva esplicitamente
    più avanti nella catena (utile per flag come --cpu che non dipendono
    da dataset/modello e che l'utente vuole mantenere fissi in tutte le run
    del loop).
    """
    merged = dict(BASE_DEFAULTS)
    if base_args is not None:
        merged.update(vars(base_args))

    merged.update(DATASET_DEFAULTS.get(dataset, {}))
    merged.update(MODEL_DEFAULTS.get(model, {}))
    merged.update(DATASET_MODEL_DEFAULTS.get((dataset, model), {}))

    # questi due campi sono sempre determinati dalla chiamata corrente,
    # indipendentemente da cosa contenesse base_args
    merged['dataset'] = dataset
    merged['model'] = model
    if merged.get('root') is None:
        merged['root'] = f'datasets/{dataset}'

    args = argparse.Namespace(**merged)

    print(f"[load_params] dataset={dataset} model={model} -> "
          f"hidden={args.hidden}, num_layers={args.num_layers}, lr={args.lr}, "
          f"dropout={args.dropout}, epochs={args.epochs}, eval_every={args.eval_every}, "
          f"patience={args.patience}, kl_weight={args.kl_weight}"
          + (f", kgcn_batch_pubs={args.kgcn_batch_pubs}, kgcn_val_subsample={args.kgcn_val_subsample}"
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
                        'costante), stessa logica usata in main.py/Trainer per testare '
                        'la robustezza in assenza di metadati testuali')
    p.add_argument('-root', type=str, default=None)
    p.add_argument('-model', type=str, default='hvgae',
                    choices=['rgcn', 'kgcn', 'kgat', 'vgae', 'hvgae', 'sage', 'gat', 'han', 'hgt'])
    p.add_argument('-hidden', type=int, default=128)
    p.add_argument('-latent_dim', type=int, default=16,
                    help='dimensione dello spazio latente per HVGAE (il paper '
                         'originale usa L=16, molto più piccolo del --hidden '
                         'usato dagli altri modelli: gli autoencoder '
                         'preferiscono dimensioni latenti basse, si veda la '
                         'sensitivity analysis del paper)')
    p.add_argument('-num_layers', type=int, default=2)
    p.add_argument('-dropout', type=float, default=0.2)
    p.add_argument('-lr', type=float, default=0.005)
    p.add_argument('-weight_decay', type=float, default=1e-5)
    p.add_argument('-epochs', type=int, default=100)
    p.add_argument('-eval_every', type=int, default=5)
    p.add_argument('-patience', type=int, default=20)
    p.add_argument('-num_neg', type=int, default=1)
    p.add_argument('-kl_weight', type=float, default=0.01,
                    help='peso del termine di KL-divergence in VGAE/HVGAE (bilancia '
                         'ricostruzione vs regolarizzazione; default basso perché '
                         'la KL su embedding ad alta dimensione tende a dominare)')
    p.add_argument('-kgcn_batch_pubs', type=int, default=64,
                    help='numero di publication (query) campionate per epoca in KGCN, '
                         'per contenere il costo (ogni query richiede un forward pass '
                         'condizionato sull\'intero grafo)')
    p.add_argument('-kgcn_val_subsample', type=int, default=200,
                    help='numero di publication della validation usate nei controlli '
                         'PERIODICI durante il training di KGCN (early stopping / best '
                         'model selection). Essendo ogni valutazione query-dipendente '
                         '(una forward pass full-graph per publication), valutare su '
                         'tutta la validation ad ogni --eval_every epoche sarebbe più '
                         'lento del training stesso. La validation completa (e i test '
                         'set) vengono comunque valutati per intero una sola volta, a '
                         'fine training, sul modello migliore.')
    p.add_argument('-gat_heads', type=int, default=4,
                    help='numero di head di attenzione per GAT (hidden deve essere '
                         'divisibile per questo valore)')
    p.add_argument('-han_heads', type=int, default=4,
                    help='numero di head di attenzione node-level per HAN (HANConv)')
    p.add_argument('-hgt_heads', type=int, default=8,
                    help='numero di head di attenzione per HGT (HGTConv)')
    p.add_argument('-cpu', action='store_true')
    return p.parse_args()


import loader
def load_data_old(dataset):
    dataset = loader.ScholarlyDataset(root=f'datasets/{dataset}/')
    data = dataset[0]
    return data


import loader

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
    print(base_args,flush=True)
    for method in ['hvgae', 'vgae', 'kgcn', 'kgat', 'rgcn', 'sage', 'gat', 'han', 'hgt']:
        print(f'dataset: {base_args.dataset}',flush=True)
        print(f'method: {method}',flush=True)
        print(f"load method params for dataset: {base_args.dataset}",flush=True)

        # base_args viene passato per preservare eventuali override da CLI
        # (es. --cpu) validi per tutte le run del loop, senza però perdere
        # le specializzazioni per dataset/modello definite in load_params.
        args = load_params(method, base_args.dataset, base_args)
        args.model = method
        args.split = base_args.split
        print(f'split: {args.split}')

        # ------------------------------------------------------------------
        # DATA LOADING: usa il tuo ScholarlyDataset esistente (dal loader).
        # Adatta il path se il modulo si chiama diversamente.
        # ------------------------------------------------------------------
        data = load_data(args.dataset, split=args.split)
        results, model, scorer = train_and_evaluate(data, args)

        print('\nFatto.')