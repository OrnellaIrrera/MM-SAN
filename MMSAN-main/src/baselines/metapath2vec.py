"""
Dataset Recommendation con MetaPath2Vec sul grafo eterogeneo ScholarlyDataset.

Riusa direttamente il tuo loader (ScholarlyDataset) e i tuoi split già
pronti: train / validation_trans / test_trans / test_semi / test_ind
sull'arco ('publication', 'cites', 'dataset').

Import da adattare:
    from <tuo_modulo_loader> import ScholarlyDataset
    from args_list import get_args

FIX rispetto alla versione precedente:
    MetaPath2Vec (PyG) va in IndexError quando, per una relazione usata nel
    metapath, esiste un nodo sorgente con out-degree 0 (es. una pubblicazione
    che compare solo in validation/test e non ha archi 'cites' in train, o
    una pubblicazione senza topic/autori/entità). Questo perché il
    campionamento CSR interno calcola un indice pari esattamente a
    col.numel() in quel caso, fuori range di uno.
    La fix aggiunge un arco fittizio verso il nodo 0 di destinazione per
    ogni nodo sorgente isolato, PRIMA di ToUndirected, per tutte le
    relazioni effettivamente attraversate dal metapath (dirette e inverse).
"""
import argparse
import torch
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.nn import MetaPath2Vec
from loader import ScholarlyDataset
import loader

# ----------------------------------------------------------------------
# 1. Caricamento dataset (riusa il tuo loader così com'è)
# ----------------------------------------------------------------------
def load_data(args):
    dataset = loader.ScholarlyDataset(root=f'datasets/{args.dataset}/')
    data = dataset[0]
    return data


# ----------------------------------------------------------------------
# 2. Utility: individua e "sistema" i nodi sorgente con out-degree 0
# ----------------------------------------------------------------------
def find_isolated_sources(edge_index, num_src_nodes):
    """Ritorna gli indici dei nodi sorgente [0, num_src_nodes) che non
    compaiono mai come sorgente in edge_index."""
    has_edge = torch.zeros(num_src_nodes, dtype=torch.bool)
    if edge_index.numel() > 0:
        has_edge[edge_index[0].unique()] = True
    return (~has_edge).nonzero(as_tuple=True)[0]


def pad_isolated_sources(edge_index, num_src_nodes, src_type, dst_type,
                          rel_name="", seed=42):
    """Aggiunge un arco fittizio per ogni nodo sorgente con out-degree 0,
    evitando il crash del campionatore CSR di MetaPath2Vec quando incontra
    un nodo senza archi uscenti per quella relazione.

    A differenza di un semplice "collega tutto al nodo 0" (che crea un hub
    artificiale e distorce l'embedding di quel singolo nodo), qui la
    destinazione fittizia viene campionata dalla DISTRIBUZIONE REALE delle
    destinazioni gia' esistenti in quella relazione: e' equivalente a
    un'imputazione statistica, non all'invenzione di una relazione privilegiata
    con un nodo arbitrario. Il rumore introdotto viene cosi' "spalmato" in modo
    coerente con la struttura osservata, invece di concentrarsi tutto su un
    unico nodo.

    Se la relazione non ha NESSUN arco (caso degenere): se src_type==dst_type
    si usa un self-loop (i, i) per ogni nodo isolato; altrimenti si solleva
    un errore esplicito, perche' non esiste alcuna distribuzione da cui
    campionare e un self-loop non avrebbe senso semantico (tipi diversi).
    """
    isolated = find_isolated_sources(edge_index, num_src_nodes)
    if isolated.numel() == 0:
        return edge_index

    print(f"  [pad] {rel_name}: {isolated.numel()}/{num_src_nodes} nodi "
          f"sorgente senza archi uscenti")

    if edge_index.numel() == 0:
        if src_type == dst_type:
            print(f"    -> relazione senza alcun arco: uso self-loop (i, i) "
                  f"per i nodi isolati")
            dummy_dst = isolated.clone()
        else:
            raise ValueError(
                f"Relazione '{rel_name}' senza alcun arco e con tipi diversi "
                f"({src_type} != {dst_type}): impossibile imputare una "
                f"destinazione plausibile. Controlla il dataset/split per "
                f"questa relazione prima di procedere."
            )
    else:
        # campiona (con seed fisso, riproducibile) dalla distribuzione reale
        # delle destinazioni gia' osservate in questa relazione
        g = torch.Generator().manual_seed(seed)
        real_dst_pool = edge_index[1]
        sample_idx = torch.randint(
            0, real_dst_pool.numel(), (isolated.numel(),), generator=g
        )
        dummy_dst = real_dst_pool[sample_idx]
        print(f"    -> destinazione imputata campionando dalla distribuzione "
              f"reale delle {real_dst_pool.numel()} destinazioni gia' esistenti")

    dummy_edges = torch.stack([isolated, dummy_dst], dim=0)
    edge_index = torch.cat([edge_index, dummy_edges], dim=1)
    return edge_index


def check_all_relations(mp_data, relations):
    """Diagnostica: stampa eventuali nodi sorgente isolati per una lista di
    relazioni (src_type, rel, dst_type) già presenti in mp_data."""
    any_isolated = False
    for src_type, rel, dst_type in relations:
        if (src_type, rel, dst_type) not in mp_data.edge_types:
            continue
        ei = mp_data[src_type, rel, dst_type].edge_index
        isolated = find_isolated_sources(ei, mp_data[src_type].num_nodes)
        if isolated.numel() > 0:
            any_isolated = True
            print(f"  [check] {src_type}-{rel}-{dst_type}: "
                  f"{isolated.numel()}/{mp_data[src_type].num_nodes} nodi isolati")
    if not any_isolated:
        print("  [check] nessun nodo isolato residuo sulle relazioni controllate.")


# ----------------------------------------------------------------------
# 3. Costruzione del grafo per MetaPath2Vec (solo edge di message passing)
# ----------------------------------------------------------------------
def build_metapath2vec_graph(data, metapath):
    """
    Costruisce un HeteroData "pulito" con edge_index standard per ogni
    relazione, usando SOLO edge_index_train per ('publication','cites','dataset')
    (niente edge di validation/test), poi aggiunge le relazioni inverse
    con ToUndirected in automatico (rev_<relazione>).

    Dopo aver assemblato il grafo, ripara i nodi sorgente con out-degree 0
    su TUTTE le relazioni effettivamente usate nel metapath (comprese le
    inverse rev_*), per evitare l'IndexError nel sampler di MetaPath2Vec.
    """
    mp_data = HeteroData()

    # nodi: copia num_nodes per ogni tipo presente nel dataset originale
    for node_type in data.node_types:
        mp_data[node_type].num_nodes = data[node_type].num_nodes

    # arco target: SOLO train, per evitare leakage nei random walk
    mp_data['publication', 'cites', 'dataset'].edge_index = \
        data['publication', 'cites', 'dataset'].edge_index_train

    # tutte le altre relazioni: copiate così come sono (non fanno parte
    # del task da predire, quindi nessun leakage)
    other_relations = [
        ('publication', 'cites', 'publication'),
        ('dataset', 'cites', 'dataset'),
        ('publication', 'hasauthor', 'author'),
        ('dataset', 'hasauthor', 'author'),
        ('publication', 'hastopic', 'topic'),
        ('dataset', 'hastopic', 'topic'),
        ('publication', 'hasentity', 'entity'),
        ('dataset', 'hasentity', 'entity'),
        ('publication', 'hasvenue', 'venue'),
        ('publication', 'haskeyword', 'keyword'),
        ('dataset', 'haskeyword', 'keyword'),
        ('publication', 'hasorganization', 'organization'),
        ('dataset', 'hasorganization', 'organization'),
    ]
    for rel in other_relations:
        if rel in data.edge_types:
            mp_data[rel].edge_index = data[rel].edge_index

    # aggiunge automaticamente le relazioni inverse (rev_cites, rev_hastopic, ...)
    mp_data = T.ToUndirected(merge=False)(mp_data)

    # ------------------------------------------------------------------
    # FIX: pad dei nodi sorgente isolati, su tutte le relazioni (dirette e
    # inverse) effettivamente attraversate dal metapath. Fatto DOPO
    # ToUndirected così copriamo anche i rev_* che il metapath usa.
    # ------------------------------------------------------------------
    print("Controllo/riparazione nodi con out-degree 0 sulle relazioni del metapath...")
    metapath_relations = list(dict.fromkeys(metapath))  # dedup mantenendo l'ordine
    for src_type, rel, dst_type in metapath_relations:
        if (src_type, rel, dst_type) not in mp_data.edge_types:
            continue
        ei = mp_data[src_type, rel, dst_type].edge_index
        mp_data[src_type, rel, dst_type].edge_index = pad_isolated_sources(
            ei, mp_data[src_type].num_nodes, src_type, dst_type,
            rel_name=f"{src_type}-{rel}-{dst_type}"
        )

    check_all_relations(mp_data, metapath_relations)

    return mp_data


# ----------------------------------------------------------------------
# 4. Training di MetaPath2Vec
# ----------------------------------------------------------------------
def train_metapath2vec(mp_data, device, metapath, epochs=50, embedding_dim=128,
                        num_workers=2):
    mp2v = MetaPath2Vec(
        mp_data.edge_index_dict,
        embedding_dim=embedding_dim,
        metapath=metapath,
        walk_length=50,
        context_size=7,
        walks_per_node=5,
        num_negative_samples=5,
        sparse=True,
    ).to(device)

    loader = mp2v.loader(batch_size=128, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.SparseAdam(list(mp2v.parameters()), lr=0.01)

    mp2v.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = mp2v.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch:03d} | Loss: {total_loss / len(loader):.4f}")

    return mp2v


# ----------------------------------------------------------------------
# 5. Utility: da edge_index a dict {pub_id: set(dataset_id)}
# ----------------------------------------------------------------------
def edge_index_to_pairs(edge_index):
    pairs = {}
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for s, d in zip(src, dst):
        pairs.setdefault(s, set()).add(d)
    return pairs


# ----------------------------------------------------------------------
# 6. Ranking
# ----------------------------------------------------------------------
def rank_datasets(pub_id, pub_emb, ds_emb, exclude_ids=None):
    q = pub_emb[pub_id].unsqueeze(0)
    sims = F.cosine_similarity(q, ds_emb)
    if exclude_ids:
        sims = sims.clone()
        sims[list(exclude_ids)] = -float('inf')
    ranking = torch.argsort(sims, descending=True)
    return ranking, sims


# ----------------------------------------------------------------------
# 7. Valutazione (MRR, Hits@k, nDCG@10)
# ----------------------------------------------------------------------
def evaluate(test_pairs, pub_emb, ds_emb, train_pairs=None, k_list=(5, 10)):
    hits = {k: [] for k in k_list}
    mrr_at_k = {k: [] for k in k_list}
    ndcg_at_k = {k: [] for k in k_list}

    for pub_id, true_ds in test_pairs.items():
        exclude = train_pairs.get(pub_id, set()) if train_pairs else set()
        ranking, _ = rank_datasets(pub_id, pub_emb, ds_emb, exclude_ids=exclude)
        ranking_list = ranking.tolist()

        # posizioni (1-based) di tutti i dataset rilevanti nel ranking completo
        rank_positions = sorted(
            ranking_list.index(d) + 1 for d in true_ds if d in ranking_list
        )
        if not rank_positions:
            continue

        first_rank = rank_positions[0]

        for k in k_list:
            # Hits@k: il primo item rilevante è entro la top-k?
            hits[k].append(int(first_rank <= k))

            # MRR@k: 0 se il primo rilevante è oltre la soglia k
            mrr_at_k[k].append(1.0 / first_rank if first_rank <= k else 0.0)

            # nDCG@k: usa solo i rilevanti entro la top-k per il DCG,
            # e min(#rilevanti totali, k) per l'IDCG (ranking ideale)
            positions_within_k = [r for r in rank_positions if r <= k]
            dcg = sum(1.0 / np.log2(r + 1) for r in positions_within_k)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_ds), k)))
            ndcg_at_k[k].append(dcg / idcg if idcg > 0 else 0.0)

    results = {}
    for k in k_list:
        results[f"Hits@{k}"] = float(np.mean(hits[k])) if hits[k] else 0.0
        results[f"MRR@{k}"] = float(np.mean(mrr_at_k[k])) if mrr_at_k[k] else 0.0
        results[f"nDCG@{k}"] = float(np.mean(ndcg_at_k[k])) if ndcg_at_k[k] else 0.0
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-dataset', type=str,
                    help='nome cartella dataset dentro ../datasets/<dataset>')
    p.add_argument('-num_workers', type=int, default=2,
                    help='numero worker per il DataLoader di MetaPath2Vec '
                         '(tienilo <= ai core disponibili sul nodo)')
    return p.parse_args()

# ----------------------------------------------------------------------
# 8. Main
# ----------------------------------------------------------------------
from args_list import get_args

def main():
    args = parse_args()

    for data in ['pubmed','mes']:
        args.dataset = data
        print(args)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
    
        print("Caricamento ScholarlyDataset...")
        data = load_data(args)
        print(args.dataset)
        # Metapath ciclico publication -> dataset -> publication -> topic -> publication
        # Modificalo/estendilo se vuoi passare anche da author/entity/keyword/venue.
        metapath = [
            ('publication', 'cites', 'dataset'),
            ('dataset', 'rev_cites', 'publication'),
            ('publication', 'hastopic', 'topic'),
            ('topic', 'rev_hastopic', 'publication'),
            ('publication', 'hasauthor', 'author'),
            ('author', 'rev_hasauthor', 'publication'),
            ('publication', 'hasentity', 'entity'),
            ('entity', 'rev_hasentity', 'publication'),
        ]
    
        print("Costruzione grafo per MetaPath2Vec (solo edge di training)...")
        mp_data = build_metapath2vec_graph(data, metapath)
        print(mp_data)
    
        print("Training MetaPath2Vec...")
        mp2v = train_metapath2vec(mp_data, device, metapath, epochs=300,
                                   embedding_dim=128, num_workers=args.num_workers)
    
        print("Estrazione embedding...")
        pub_emb = mp2v('publication').detach().cpu()
        ds_emb = mp2v('dataset').detach().cpu()
    
        pd_store = data['publication', 'cites', 'dataset']
        train_pairs = edge_index_to_pairs(pd_store.edge_index_train)
    
        eval_splits = {
            'validation_trans': pd_store.edge_label_index_validation_trans,
            'test_trans': pd_store.edge_label_index_test_trans,
            'test_semi': pd_store.edge_label_index_test_semi,
            'test_ind': pd_store.edge_label_index_test_ind,
        }
    
        print("\n=== Risultati per split ===")
        all_results = {}
        for split_name, edge_label_index in eval_splits.items():
            test_pairs = edge_index_to_pairs(edge_label_index)
            results = evaluate(test_pairs, pub_emb, ds_emb, train_pairs=train_pairs,
                                k_list=(1, 5, 10))
            all_results[split_name] = results
            print(f"\n[{split_name}]")
            for metric, value in results.items():
                print(f"  {metric}: {value:.4f}")
    
        save = False
        if save:
            torch.save({'pub_emb': pub_emb, 'ds_emb': ds_emb, 'results': all_results},
                       f'metapath2vec_{args.dataset}_embeddings.pt')
            print(f"\nEmbedding e risultati salvati in 'metapath2vec_{args.dataset}_embeddings.pt'")
    

if __name__ == '__main__':
    main()