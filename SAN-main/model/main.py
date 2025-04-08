import pickle

import torch
from args_list import get_args
import numpy as np
import random
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch.nn as nn
import loader
import sampler
from loader import *
import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from utils import EarlyStoppingClass
from sampler_sigir import RandomWalkWithRestart
import time
from model_sigir import ScHetGNN
from torch_geometric import seed_everything


import torch.nn.functional as F

# in transductive basta fare i path una volta, tanto i nodi li ho sempre visti in ogni set
# in inductive devo farne 3 separati, uno per training, uno per validation e uno per test


"""
Questo è il primo setup, quello originale usato sin dall'inizio



"""
# seed_everything(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed_all(42)
# random.seed(42)
# torch.backends.cudnn.deterministic=True
# torch.backends.cudnn.benchmark = False
args = get_args()


def seed_torch(seed=42):
    random.seed(seed)
    seed_everything(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.set_rng_state(torch.get_rng_state())




# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0


def ndcg_at_k(true_labels, predictions, k):
    relevance_scores = [1 if item in true_labels else 0 for item in predictions]
    dcg = dcg_at_k(relevance_scores, k)
    idcg = dcg_at_k([1] * len(true_labels), k)  # IDCG assuming all true labels are relevant
    if not idcg:
        return 0
    return dcg / idcg


def cosine_similarity(emb1, emb2):
    dot_product = np.dot(emb1, emb2)
    norm_vector1 = np.linalg.norm(emb1)
    norm_vector2 = np.linalg.norm(emb2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

def load_embeddings(filename):
    """Load embeddings from a file using pickle."""
    with open(filename, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"Embeddings loaded from {filename}")
    return embeddings

class Trainer:
    def __init__(self, args):
        # self.device = 'cpu'
        # if args.train:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        print(f'DEVICE {self.device}')

        train_dataset = loader.ScholarlyDataset(root=f'datasets/{args.dataset}/split_transductive/train/')
        self.train_root = train_dataset.root
        self.dataset = train_dataset[0]
        if args.no_metadata:
            self.dataset['dataset'].x = torch.ones(self.dataset['dataset'].x.shape[0], 384)


        if args.split > 0:
            # Sostituire i valori di .x per i nodi campionati
            sampled_indices = torch.randperm(self.dataset['dataset'].num_nodes)[:int(args.split/100 * self.dataset['dataset'].num_nodes)]

            repeated_embeddings = load_embeddings("model/empty_embedding.pkl")
            repeated_embeddings = np.tile(repeated_embeddings, (len(sampled_indices), 1))
            repeated_embeddings_tensor = torch.tensor(repeated_embeddings, dtype=torch.float)
            self.dataset['dataset'].x[sampled_indices] = repeated_embeddings_tensor
            self.dataset['dataset'].x[sampled_indices] = torch.ones(len(sampled_indices), 384)


        print(f'TEST {args.test}')
        self.walker = RandomWalkWithRestart(self.args, self.dataset, 'transductive train',test=args.test)
        self.model = ScHetGNN(args).to(self.device)


        self.model.init_weights()
        # print("Pesi inizializzati:")
        # for name, param in self.model.named_parameters():
        #     if 'weight' in name:
        #         print(f"{name}:\n{param.data}\n")
        #
        #     # Se vuoi stampare anche i bias
        #     if 'bias' in name:
        #         print(f"{name}:\n{param.data}\n")

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wd)

    def get_walks(self):
        if not os.path.exists(f'./model/data/{self.args.dataset}_transductive_paths.txt'):
            f = open(f'./model/data/{self.args.dataset}_transductive_paths.txt', 'w')

            all_walks = self.walker.create_random_walks(all=True)
            for walk in all_walks:
                f.write(' '.join(walk))
                f.write('\n')
            f.close()
        else:
            self.walker.set_seeds()
            f = open(f'./model/data/{self.args.dataset}_test_walks.txt', 'r')
            all_walks = f.readlines()
            all_walks = [w.split() for w in all_walks]
            print(f'walks: {len(all_walks)}')
        assert all_walks != []
        return all_walks

    def get_test_walks(self):
        if not os.path.exists(f'./model/data/{self.args.dataset}_test_walks.txt'):
            f = open(f'./model/data/{self.args.dataset}_test_walks.txt', 'w')

            all_walks = self.walker.create_random_walks(all=True)
            for walk in all_walks:
                f.write(' '.join(walk))
                f.write('\n')
            f.close()
        else:
            self.walker.set_seeds()
            f = open(f'./model/data/{self.args.dataset}_test_walks.txt', 'r')
            all_walks = f.readlines()
            all_walks = [w.split() for w in all_walks]
            print(f'walks: {len(all_walks)}')
        assert all_walks != []
        return all_walks

    def trivial_baselines(self):
        """ compute trivial performances"""
        results = json.load(open(f'baselines/trivial/data/{self.args.dataset}/results.json', 'r'))
        mapped_results = {}
        precision, recall, ndcg = 0, 0, 0
        c = 0
        for k, v in results.items():
            mapped_k = self.dataset['publication'].mapping[k]
            mapped_v = [self.dataset['dataset'].mapping[a] for a in v]
            mapped_results[mapped_k] = mapped_v[0:20]
            c += 1
            pred = mapped_v[0:self.args.topk]
            true = self.y_test_true_labels[mapped_k]
            precision += len(list(set(pred).intersection(set(true)))) / self.args.topk
            recall += len(list(set(pred).intersection(set(true)))) / len(true)
            ndcg += ndcg_at_k(true, pred, self.args.topk)
        precision, recall, ndcg = precision / c, recall / c, ndcg / c

        reranking_line = 'GOAL precision = {}'.format(precision) + ' recall = {}'.format(
            recall) + ' ndcg = {}'.format(ndcg) + '\n'
        print(reranking_line)
        return mapped_results

    def test(self, test_positive_pd_edges, test_negative_pd_edges, epoch, stringa, best=False, type='trans'):
        print(f'TYPE: {type}')
        self.model.eval()
        with torch.no_grad():

            output_file_path = f"model/checkpoint/transductive/results/{self.args.dataset}/{self.args.enriched}/UNIQUE_RESULTS_unique_{epoch}_{stringa}_{type}.txt"
            print(output_file_path)
            if best:
                output_file_path = f"model/checkpoint/transductive/results/{self.args.dataset}/{self.args.enriched}/BEST_unique_{epoch}_{stringa}.txt"
            if isinstance(epoch, int) and epoch + 1 == self.args.epochs:
                output_file_path = f"model/checkpoint/transductive/results/{self.args.dataset}/{self.args.enriched}/LAST_unique_{epoch}_{stringa}.txt"

            if self.args.eval_lr:
                folder_path = f"model/checkpoint/transductive/results/{self.args.dataset}/ablation/lr_{self.args.lr}"
                os.makedirs(folder_path, exist_ok=True)
                output_file_path = f"{folder_path}/{epoch}_{stringa}.txt"

            elif self.args.eval_batch:
                folder_path = f"model/checkpoint/transductive/results/{self.args.dataset}/ablation/batch_{self.args.batch_size}"
                os.makedirs(folder_path, exist_ok=True)
                output_file_path = f"model/checkpoint/transductive/results/{self.args.dataset}/ablation/batch_{self.args.batch_size}/{epoch}_{stringa}.txt"

            elif self.args.eval_combine_aggregation:
                folder_path = f"model/checkpoint/transductive/results/{self.args.dataset}/ablation/aggr_all_{self.args.all_aggregation}"
                os.makedirs(folder_path, exist_ok=True)
                output_file_path = f"model/checkpoint/transductive/results/{self.args.dataset}/ablation/aggr_all_{self.args.all_aggregation}/{epoch}_{stringa}.txt"

            elif self.args.eval_aggregation:
                folder_path = f"model/checkpoint/transductive/results/{self.args.dataset}/ablation/aggr_{self.args.core_aggregation}"
                os.makedirs(folder_path, exist_ok=True)
                output_file_path = f"model/checkpoint/transductive/results/{self.args.dataset}/ablation/aggr_{self.args.core_aggregation}/{epoch}_{stringa}.txt"

            elif self.args.eval_neigh:
                folder_path = f"model/checkpoint/transductive/results/{self.args.dataset}/ablation/neigh_{self.args.n_cores}_{self.args.n_keys_hubs}_{self.args.n_top_hubs}"
                os.makedirs(folder_path, exist_ok=True)
                output_file_path = f"model/checkpoint/transductive/results/{self.args.dataset}/ablation/neigh_{self.args.n_cores}_{self.args.n_keys_hubs}_{self.args.n_top_hubs}/{epoch}_{stringa}.txt"

            elif self.args.eval_heads:
                folder_path = f"model/checkpoint/transductive/results/{self.args.dataset}/ablation/heads_{self.args.heads}"
                os.makedirs(folder_path, exist_ok=True)
                output_file_path = f"model/checkpoint/transductive/results/{self.args.dataset}/ablation/heads_{self.args.heads}/{epoch}_{stringa}.txt"
            f = open(output_file_path, 'w')
            f.write(stringa + '\n')
            self.model.eval()
            auc_tot = 0
            ap_tot = 0
            f1_tot = 0
            #if not os.path.exists(f'./model/data/test_walks/{self.args.dataset}_best_test_paths.txt'):
            range_s = 40
            if self.args.dataset == 'pubmed':
                range_s = 30
            # if self.args.dataset == 'mes' and self.args.inductive_type == 'trans':
            #     range_s = 1
            # mes: 27,28
            # pubmed: 6, 29
            selected_seeds_walks_1,selected_seeds_cores_1,selected_seeds_hubs_top_1,selected_seeds_hubs_key_1 = [],[],[],[]

            for iter in range(1):
                print('eval round: ', str(iter))
                removed,targets_0 = [],[]
                sources = list(self.y_test_true_labels.keys())
                neg_sources = list(test_negative_pd_edges[0].tolist())
                datasets = list(self.dataset['dataset'].mapping.keys())
                pos_source = [self.dataset['publication'].rev_mapping[j] for j in sources]
                neg_source = [self.dataset['publication'].rev_mapping[j] for j in neg_sources]
                all_seeds = pos_source + neg_source + datasets
                print(len(datasets), len(pos_source), len(neg_source), len(all_seeds))


                print(len(datasets), len(pos_source), len(neg_source), len(all_seeds))
                self.walker.set_seeds(all_seeds)
                if self.args.dataset == 'mes' :
                    if self.args.inductive_type == 'trans':
                        iter = 3
                    elif self.args.inductive_type == 'light':
                        iter = 13
                    else:
                        iter = 16
                else:
                    if self.args.inductive_type == 'trans':
                        iter = 8
                    elif self.args.inductive_type == 'light':
                        iter = 4
                    else:
                        iter = 14
                if not os.path.exists(f'./model/data/test_walks/{self.args.dataset}_{iter}_{type}_unique_test_paths_0.txt') :
                    all_walks = self.walker.create_random_walks(seeds_in=all_seeds)
                    print(f'created {len(all_walks)} random walks')
                    #if not args.eval_aggregation:
                    ff = open(f'./model/data/test_walks/{self.args.dataset}_{iter}_{type}_unique_test_paths_0.txt', 'w')
                    for walk in all_walks:
                        ff.write(' '.join(walk))
                        ff.write('\n')
                    print('written')
                    ff.close()
                else:
                    print('open')

                    ff = open(f'./model/data/test_walks/{self.args.dataset}_{iter}_{type}_unique_test_paths_0.txt', 'r')
                    all_walks = ff.readlines()
                    all_walks = [w.split() for w in all_walks]
                    ff.close()
                #
                walks = [w for w in all_walks]
                all_walks = {seed: [] for seed in self.walker.G.nodes if
                             self.walker.is_publication(seed) or self.walker.is_dataset(seed)}

                for walk in walks:
                    all_walks[walk[0]].append(walk)


                all_walks = {k: v for k, v in all_walks.items() if len(v) > 0}
                all_walks = [v for k, v in all_walks.items() if k in all_seeds]
                all_walks = [inner for outer in all_walks for inner in outer]
                #print(all_walks[0:10])
                selected_seeds_walks, selected_seeds_cores, selected_seeds_hubs_key, selected_seeds_hubs_top = self.walker.select_walks_mp(
                    all_walks)

                # selected_seeds_walks_0 = [elemento for sottolista in selected_seeds_walks for elemento in sottolista]
                # selected_seeds_walks_0 = [elemento for sottolista in selected_seeds_walks_0 for elemento in sottolista]
                # selected_seeds_cores_0 = [elemento for sottolista in selected_seeds_cores for elemento in sottolista]
                # selected_seeds_hubs_top_0 = [elemento for sottolista in selected_seeds_hubs_top for elemento in sottolista]
                # selected_seeds_hubs_key_0 = [elemento for sottolista in selected_seeds_hubs_key for elemento in sottolista]
                # print(selected_seeds_walks_0[0:5])
                # print(selected_seeds_cores_0[0:5])
                # print(selected_seeds_hubs_top_0[0:5])
                # print(selected_seeds_hubs_key_0[0:5])
                # if iter == 0:
                #     selected_seeds_walks_1 = selected_seeds_walks_0
                #     selected_seeds_cores_1 = selected_seeds_cores_0
                #     selected_seeds_hubs_top_1 = selected_seeds_hubs_top_0
                #     selected_seeds_hubs_key_1 = selected_seeds_hubs_key_0
                # if iter == 1:
                #     for i,j in zip(selected_seeds_walks_0, selected_seeds_walks_1):
                #         if i != j:
                #             print('attenzione walk')
                #
                #     for i, j in zip(selected_seeds_cores_0, selected_seeds_cores_1):
                #         if i != j:
                #             print('attenzione core')
                #
                #     for i, j in zip(selected_seeds_hubs_top_0, selected_seeds_hubs_top_1):
                #         if i != j:
                #             print('attenzione top')
                #
                #     for i, j in zip(selected_seeds_hubs_key_0, selected_seeds_hubs_key_1):
                #         if i != j:
                #             print('attenzione key')

                seed_vectors, seed_vectors_net, net_cores, cores, net_keys, keys, net_hubs, hubs, _, _, _, _, _, _, _, _ = self.walker.get_neighbours_vector(
                    selected_seeds_cores,
                    selected_seeds_hubs_key,
                    selected_seeds_hubs_top)
                # print(seed_vectors[0:5])
                # print(seed_vectors_net[0:5])
                # print(net_cores[0:5])
                k = [tuple([sublist[0],sublist[1]]) for lista in keys for sublist in lista]
                k = sorted(k, key=lambda tup: (tup[0],tup[1]))
                k_path = "./model/data/test_walks/keys0.json"

                # Se i pesi non sono stati ancora salvati, salvali
                # if not os.path.exists(k_path):
                #     json.dump(k, open(k_path, 'w'))
                # else:
                #     kk = json.load(open(k_path, 'r'))
                #     for i,j in zip(k,kk):
                #         if tuple(i) != tuple(j):
                #             print(f'different: {i}, {j}')
                # print(keys[0:1])
                # print(hubs[0:1])
                pos_source_indices = [all_seeds.index(j) for i, j in enumerate(pos_source)]

                pos_target_indices = [all_seeds.index(j) for i, j in enumerate(datasets)]
                if self.model.training:
                    print("Il modello è in modalità TRAIN")
                else:
                    print("Il modello è in modalità EVAL")
                torch.cuda.synchronize()
                self.model = self.model.float()
                final_embeddings = self.model(seed_vectors, seed_vectors_net, net_cores, cores, net_keys, keys,
                                              net_hubs,hubs, core_agg=self.args.core_aggregation,
                                              key_agg=self.args.key_aggregation, top_agg=self.args.top_aggregation,
                                              all_agg=self.args.all_aggregation)

                weights_path = "./model/data/test_walks/initial_weights.pth"

                # Se i pesi non sono stati ancora salvati, salvali
                if not os.path.exists(weights_path):
                    torch.save(self.model.state_dict(), weights_path)
                    print("Pesi iniziali salvati.")
                else:
                    saved_weights = torch.load(weights_path)

                    equal = True
                    # for (name, param), (saved_name, saved_param) in zip(self.model.named_parameters(),
                    #                                                     saved_weights.items()):
                    #     if not torch.equal(param.detach().cpu(), saved_param.cpu()):
                    #         equal = False
                    #         print(f"Differenza trovata nel parametro: {name}")
                    #         # Confronto vettore per vettore nel tensore
                    #         for i in range(param.numel()):  # numel() restituisce il numero totale di elementi nel tensore
                    #             if param.view(-1)[i] != saved_param.view(-1)[i]:
                    #                 print(
                    #                     f"Valore differente in posizione {i}: {param.view(-1)[i]} vs {saved_param.view(-1)[i]}")
                    #                 break  # Esce dal loop se trova una differenza

                    if equal:
                        print("Tutti i pesi sono identici!")
                    else:
                        print("Ci sono differenze nei pesi!")


                file_path = "./model/data/test_walks/final_embeddings.pkl"

                # Save the embeddings
                if not os.path.exists(file_path):
                    with open(file_path, "wb") as f:
                        pickle.dump(final_embeddings.detach().cpu(), f)
                else:
                    embs = pickle.load(open(file_path, "rb"))
                    equal = torch.equal(final_embeddings.detach().cpu(), embs.detach().cpu())
                    # print("Sono identici:", equal)
                    # for i in range(final_embeddings.shape[0]):
                        # equal = torch.allclose(final_embeddings.detach().cpu()[i], embs.detach().cpu()[i], atol=1e-3)
                        # torch.equal(final_embeddings.detach().cpu()[i], embs.detach().cpu()[i])
                        #if not equal:

                            # print(f"Vettore {i} diverso!")
                            #
                            # print("Final_embeddings:", final_embeddings[i])
                            # print("Embs:", embs[i])
                            # indices_diff = (final_embeddings.detach().cpu()[i] != embs.detach().cpu()[i]).nonzero(as_tuple=True)[0]
                            #
                            # # Visualizza gli indici e i valori diversi
                            # for idx in indices_diff:
                            #     print(
                            #         f"Indice {idx.item()}: Final_embeddings = {final_embeddings.detach().cpu()[i][idx].item()}, Embs = {embs.detach().cpu()[i][idx].item()}")
                            # break
                #print(final_embeddings[0:3])
                # Calcola il prodotto scalare tra gli embeddings normalizzati
                pub_embeddings = F.normalize(final_embeddings[pos_source_indices], p=2, dim=1)
                data_embeddings = F.normalize(final_embeddings[pos_target_indices], p=2, dim=1)
                #print(pub_embeddings[10:30])
                #print(data_embeddings[10:30])
                final_matrix = torch.mm(pub_embeddings, data_embeddings.t())
                #final_matrix = torch.sigmoid(final_matrix)
                #print(final_matrix)
                top_values, top_indices = torch.topk(final_matrix, k=10, dim=1)
                #print(top_values)
                y_test_predicted_labels_norerank = {source: [] for source in sources}
                for i, lista in enumerate(top_indices.tolist()):
                    y_test_predicted_labels_norerank[sources[i]] = lista[0:10]



                precisions,recalls,ndcgs = [],[],[]

                # LINK PREDICTION
                y_true_test = np.array([1] * test_positive_pd_edges.size(1) + [0] * test_negative_pd_edges.size(1))
                all_walks = {seed: [] for seed in self.walker.G.nodes if
                             self.walker.is_publication(seed) or self.walker.is_dataset(seed)}

                for walk in walks:
                    all_walks[walk[0]].append(walk)

                all_walks = {k: v for k, v in all_walks.items() if len(v) > 0}
                if self.args.lp:
                    loss, final_embeddings, pos_source, pos_target, neg_source, neg_target, all_seeds = self.run_minibatch_transductive(
                        self.dataset, 0,
                        test_positive_pd_edges, test_negative_pd_edges,
                        test=True, all_walks=all_walks)

                    pos_embeddings_source_ori, neg_embeddings_source_ori = final_embeddings[pos_source], \
                    final_embeddings[
                        neg_source]
                    pos_embeddings_target_ori, neg_embeddings_target_ori = final_embeddings[pos_target], \
                    final_embeddings[
                        neg_target]

                    pos_embeddings_source = pos_embeddings_source_ori.view(pos_embeddings_source_ori.size(0), 1,
                                                                           pos_embeddings_source_ori.size(
                                                                               1))  # [batch_size, 1, embed_d]
                    neg_embeddings_source = neg_embeddings_source_ori.view(neg_embeddings_source_ori.size(0), 1,
                                                                           neg_embeddings_source_ori.size(
                                                                               1))  # [batch_size, 1, embed_d]
                    pos_embeddings_target = pos_embeddings_target_ori.view(pos_embeddings_target_ori.size(0),
                                                                           pos_embeddings_target_ori.size(1),
                                                                           1)  # [batch_size, embed_d, 1]
                    neg_embeddings_target = neg_embeddings_target_ori.view(neg_embeddings_target_ori.size(0),
                                                                           neg_embeddings_target_ori.size(1),
                                                                           1)  # [batch_size, embed_d, 1]
                    if args.core_aggregation == 'mh-attention':
                        pos_embeddings_source = F.normalize(pos_embeddings_source, p=2, dim=-1)
                        pos_embeddings_target = F.normalize(pos_embeddings_target, p=2, dim=-1)

                        neg_embeddings_source = F.normalize(neg_embeddings_source, p=2, dim=-1)
                        neg_embeddings_target = F.normalize(neg_embeddings_target, p=2, dim=-1)

                    result_positive_matrix = torch.bmm(pos_embeddings_source, pos_embeddings_target)
                    result_positive_matrix = torch.sigmoid(result_positive_matrix)



                    result_negative_matrix = torch.bmm(neg_embeddings_source, neg_embeddings_target)
                    result_negative_matrix = torch.sigmoid(result_negative_matrix)

                    y_predicted_test = torch.cat([result_positive_matrix.squeeze(), result_negative_matrix.squeeze()])
                    y_predicted_test = y_predicted_test.detach().cpu().numpy()




                    auc = roc_auc_score(y_true_test, y_predicted_test)
                    auc_tot += auc
                    ap = average_precision_score(y_true_test, y_predicted_test)
                    ap_tot += ap
                    threshold = args.f1_threshold
                    y_pred = [1 if score >= threshold else 0 for score in y_predicted_test]
                    f1 = f1_score(y_true_test, y_pred)
                    f1_tot += f1
                    print('Link Prediction Test\n')
                    print('AUC = {}'.format(auc))
                    print('AP = {}'.format(ap))
                    print('F1 = {}\n\n'.format(f1))

                if self.args.rec:
                    for topk in [1, 5, 10]:
                        precision, no_rer_precision = 0, 0
                        recall, no_rer_recall = 0, 0
                        ndcg, no_rer_ndcg = 0, 0
                        for source in sources:
                            #print(source)
                            true = self.y_test_true_labels[source]
                            #print(true)
                            pred_nonrer = y_test_predicted_labels_norerank[source][:topk]
                            # print(pred_nonrer)
                            # print('\n\n')
                            no_rer_precision += len(list(set(pred_nonrer).intersection(true))) / topk
                            no_rer_recall += len(list(set(pred_nonrer).intersection(true))) / len(true)
                            no_rer_ndcg += ndcg_at_k(true, pred_nonrer, topk)

                        print(f'results {topk}')
                        print(no_rer_precision / len(sources))
                        print(no_rer_recall / len(sources))
                        print(no_rer_ndcg / len(sources))
                        precisions.append(no_rer_precision / len(sources))
                        recalls.append(no_rer_recall / len(sources))
                        ndcgs.append(no_rer_ndcg / len(sources))



            # f.write(f'\ntype\n {type}')
            # f.write('LINK PREDICTION\n')
            # f.write(f"AUC {auc}\n")
            # f.write(f"AP {ap}\n")
            # f.write(f"F1 {f1}\n\n")
            # f.write('RECOMMENDATION\n')
            # f.write('top 1\n')
            # f.write(f"precision {precisions[0]}\n")
            # f.write(f"recall {recalls[0]}\n")
            # f.write(f"nDCG {ndcgs[0]}\n")
            # f.write('top 5\n')
            # f.write(f"precision {precisions[1]}\n")
            # f.write(f"recall {recalls[1]}\n")
            # f.write(f"nDCG {ndcgs[1]}\n")
            # f.write('top 10\n')
            # f.write(f"precision {precisions[2]}\n")
            # f.write(f"recall {recalls[2]}\n")
            # f.write(f"nDCG {ndcgs[2]}\n")
            # f.close()


    def run_minibatch_transductive(self, dataset, iteration, positive_edges, negative_edges, test=False,
                                   all_walks=None):

        # seleziono gli indici dell'edge index che mi interessano. Divido per due la batch: voglio ugual numero di archi positivi e negativi
        # i nodi satanno al più il doppio della batchsize perchè ogni arco ha due nodi
        if not test:
            batch_positive = positive_edges[:, iteration * self.args.batch_size: (iteration + 1) * self.args.batch_size]
            batch_negative = negative_edges[:, iteration * self.args.batch_size: (iteration + 1) * self.args.batch_size]

        else:
            batch_positive = positive_edges
            batch_negative = negative_edges

        positive_sources = batch_positive[0].tolist()
        positive_target = batch_positive[1].tolist()
        mapped_sources = [dataset['publication'].rev_mapping[s] for s in positive_sources]
        mapped_targets = [dataset['dataset'].rev_mapping[s] for s in positive_target]
        positive_seeds = sorted(list(set(mapped_sources + mapped_targets)))

        negative_sources = batch_negative[0].tolist()
        negative_target = batch_negative[1].tolist()
        mapped_neg_sources = [dataset['publication'].rev_mapping[s] for s in negative_sources]
        mapped_neg_targets = [dataset['dataset'].rev_mapping[s] for s in negative_target]
        negative_seeds = sorted(list(set(mapped_neg_sources + mapped_neg_targets)))

        all_seeds = sorted(list(set(positive_seeds + negative_seeds)))

        if self.args.verbose:
            print(f'positive seeds {len(positive_seeds)}')
            print(f'negative seeds {len(negative_seeds)}')
            print(f'all seeds {len(all_seeds)}')

        if all_walks is None:
            all_walks = self.walker.create_random_walks(seeds_in=all_seeds)
        else:
            self.walker.set_seeds(all_seeds)
            all_walks = [v for k, v in all_walks.items() if k in all_seeds]
            all_walks = [inner for outer in all_walks for inner in outer]

        pos_source_indices = [all_seeds.index(j) for i, j in enumerate(mapped_sources)]
        pos_target_indices = [all_seeds.index(j) for i, j in enumerate(mapped_targets)]
        neg_source_indices = [all_seeds.index(j) for i, j in enumerate(mapped_neg_sources)]
        neg_target_indices = [all_seeds.index(j) for i, j in enumerate(mapped_neg_targets)]

        # # # seed_vectors: dim = 3: seed, mapped seed, vector
        # # # remaining: dim = 5: seed, id, score, mapped_id, vector
        if self.args.verbose:
            print('start model')

        selected_seeds_walks, selected_seeds_cores, selected_seeds_hubs_key, selected_seeds_hubs_top = self.walker.select_walks_mp(
            all_walks)
        seed_vectors, seed_vectors_net, net_cores, cores, net_keys, keys, net_hubs,hubs, _, _, _, _, _, _, _, _ = self.walker.get_neighbours_vector(
            selected_seeds_cores, selected_seeds_hubs_key, selected_seeds_hubs_top)
        final_embeddings = self.model(seed_vectors, seed_vectors_net, net_cores, cores, net_keys, keys, net_hubs,hubs,
                                      core_agg=self.args.core_aggregation, key_agg=self.args.key_aggregation,
                                      top_agg=self.args.top_aggregation, all_agg=self.args.all_aggregation)
        # print(final_embeddings.shape)
        # print(pos_source_indices)
        # print(pos_target_indices)
        # print(neg_source_indices)
        # print(neg_target_indices)

        if args.loss == 'bpr':
            loss = self.model.BPRloss(final_embeddings, pos_source_indices, pos_target_indices,
                                             neg_source_indices, neg_target_indices)
        elif args.loss == 'kl':
            print('loss kl')
            loss = self.model.kl_divergence_loss(final_embeddings, pos_source_indices, pos_target_indices,
                                         neg_source_indices, neg_target_indices)
        elif args.loss == 'binaryloss':
            loss = self.model.binary_cross_entropy_loss(final_embeddings, pos_source_indices, pos_target_indices,
                                         neg_source_indices, neg_target_indices)
        else:
            loss = self.model.cross_entropy_loss(final_embeddings, pos_source_indices, pos_target_indices,
                                             neg_source_indices, neg_target_indices)
        if test:
            return loss, final_embeddings, pos_source_indices, pos_target_indices, neg_source_indices, neg_target_indices, all_seeds
        else:
            return loss, final_embeddings

    def run_transductive(self,type='trans'):

        """
        CASE 1: validation and test sets connect nodes already seen in training set
        CASE 2: same but with enriched training set with new edges between nodes not present in validation and test
        CASE 3: original trnsductive split
        """

        # early_stopping = EarlyStopping(patience=patience, verbose=True, save_path='checkpoint/checkpoint_{}.pt'.format(self.args.dataset))

        # first learn node embeddings then use them to the downstream tasks

        edge_label_index_train = self.dataset['publication', 'cites', 'dataset'].edge_label_index_train
        edge_label_index_validation = self.dataset['publication', 'cites', 'dataset'].edge_label_index_validation_trans
        print(f'train {edge_label_index_train.shape}')
        print(f'validation {edge_label_index_validation.shape}')
        edge_label_index_test = self.dataset['publication', 'cites', 'dataset'].edge_label_index_test_trans
        test_negative_pd_edges = self.dataset['publication', 'cites', 'dataset'].negative_edge_label_index_test_trans

        if type == 'light':
            edge_label_index_test = self.dataset['publication', 'cites', 'dataset'].edge_label_index_test_semi
            test_negative_pd_edges = self.dataset['publication', 'cites', 'dataset'].negative_edge_label_index_test_semi

        elif type == 'full':
            edge_label_index_test = self.dataset['publication', 'cites', 'dataset'].edge_label_index_test_ind
            test_negative_pd_edges = self.dataset['publication', 'cites', 'dataset'].negative_edge_label_index_test_ind

        print('Generating negative edges')
        training_positive_pd_edges = edge_label_index_train
        validation_positive_pd_edges = edge_label_index_validation
        test_positive_pd_edges = edge_label_index_test

        training_negative_pd_edges = self.dataset['publication', 'cites', 'dataset'].negative_edge_label_index_train_trans
        validation_negative_pd_edges = self.dataset[
            'publication', 'cites', 'dataset'].negative_edge_label_index_validation_trans

        self.y_true_test = np.array([1] * test_positive_pd_edges.size(1) + [0] * test_negative_pd_edges.size(1))
        sources = list(test_positive_pd_edges[0].tolist())
        targets = list(test_positive_pd_edges[1].tolist())
        y_test_true_labels = {source: [] for source in sources}

        for source, target in zip(sources, targets):
            y_test_true_labels[source].append(target)

        self.y_test_true_labels = {k: y_test_true_labels[k] for k in sorted(list(y_test_true_labels.keys()))}
        print(f'true labels: {self.y_test_true_labels}')

        stringa = f'lr-{self.args.lr}_heads-{self.args.heads}_batch-{self.args.batch_size}_cores-{self.args.n_cores}_key-{self.args.n_keys_hubs}_top-{self.args.n_top_hubs}_aggrcore-{self.args.core_aggregation}_aggrkeys-{self.args.key_aggregation}_aggrtop-{self.args.top_aggregation}_allagg-{self.args.all_aggregation}_loss-{args.loss}'

        stringa = 'unique_' + self.args.dataset + '_' + stringa
        if self.args.no_aug:
            stringa = 'no_aug_unique_' + stringa

        save_epoch_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/{self.args.enriched}/unique_last_checkpoint_{stringa}_last_epoch.pt'
        save_early_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/{self.args.enriched}/unique_best_checkpoint_{stringa}.pt'
        if self.args.eval_lr :
            folder_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/lr_{self.args.lr}'
            os.makedirs(folder_path, exist_ok=True)

            save_epoch_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/lr_{self.args.lr}/checkpoint_{stringa}_last_epoch.pt'
            save_early_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/lr_{self.args.lr}/checkpoint_{stringa}.pt'
        elif self.args.eval_batch :
            folder_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/batch_{self.args.batch_size}'
            os.makedirs(folder_path, exist_ok=True)
            save_epoch_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/batch_{self.args.batch_size}/checkpoint_{stringa}_last_epoch.pt'
            save_early_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/batch_{self.args.batch_size}/checkpoint_{stringa}.pt'
        elif self.args.eval_combine_aggregation :
            folder_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/aggr_all_{self.args.all_aggregation}'
            os.makedirs(folder_path, exist_ok=True)
            save_early_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/aggr_all_{self.args.all_aggregation}/checkpoint_{stringa}.pt'
            save_epoch_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/aggr_all_{self.args.all_aggregation}/checkpoint_{stringa}_last_epoch.pt'
        elif self.args.eval_aggregation :
            folder_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/aggr_{self.args.core_aggregation}'
            os.makedirs(folder_path, exist_ok=True)
            save_epoch_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/aggr_{self.args.core_aggregation}/checkpoint_{stringa}_last_epoch.pt'
            save_early_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/aggr_{self.args.core_aggregation}/checkpoint_{stringa}.pt'
        elif self.args.eval_neigh :
            folder_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/neigh_{self.args.n_cores}_{self.args.n_keys_hubs}_{self.args.n_top_hubs}'
            os.makedirs(folder_path, exist_ok=True)
            save_epoch_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/neigh_{self.args.n_cores}_{self.args.n_keys_hubs}_{self.args.n_top_hubs}/checkpoint_{stringa}_last_epoch.pt'
            save_early_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/neigh_{self.args.n_cores}_{self.args.n_keys_hubs}_{self.args.n_top_hubs}/checkpoint_{stringa}.pt'
        elif self.args.eval_heads:
            folder_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/heads_{self.args.heads}'
            os.makedirs(folder_path, exist_ok=True)
            save_epoch_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/heads_{self.args.heads}/checkpoint_{stringa}_last_epoch.pt'
            save_early_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/heads_{self.args.heads}/checkpoint_{stringa}.pt'

        early_stopping = EarlyStoppingClass(patience=args.patience, verbose=True, save_epoch_path=save_epoch_path,
                                            save_early_path=save_epoch_path)
        epochs = -1
        print(os.path.exists(save_epoch_path))

        #save_epoch_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/enriched_all/unique_last_checkpoint_unique_mes_lr-1e-05_heads-8_batch-1024_cores-5_key-5_top-5_aggrcore-mh-attention_aggrkeys-mh-attention_aggrtop-mh-attention_allagg-concat_last_epoch_epoch_99.pt'
        if os.path.exists(save_epoch_path) and (not self.args.restart or self.args.test):
            print('LOADING')

            # path = save_epoch_path
            # checkpoint = torch.load(path)
            # self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # epochs = checkpoint['epoch']
            # best_score = checkpoint['best_score']
            # patience_reached = self.args.patience
            # early_stopping = EarlyStoppingClass(patience=patience_reached, verbose=True, save_epoch_path=save_epoch_path,
            #                                     save_early_path=save_early_path,best_score=best_score)
            # print(f'STARTING FROM: epoch {epochs}')
            print(f'path: {save_epoch_path}')
            checkpoint = torch.load(save_epoch_path)

            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epochs = checkpoint['epoch']
                best_score = checkpoint['best_score']
                patience_reached = self.args.patience
                early_stopping = EarlyStoppingClass(patience=patience_reached, verbose=True,
                                                    save_epoch_path=save_epoch_path,
                                                    save_early_path=save_early_path, best_score=best_score)
            except:
                self.model.load_state_dict(checkpoint)
        else:
            print('NEW TRAINING STARTED')


        if self.args.train and not self.args.test:
            for epoch in tqdm.tqdm(range(epochs + 1, self.args.epochs), desc="Epoch"):
                # random shuffle before mini-batch training
                t_start = time.time()
                self.model.train()

                # nel training shuffle così ho sempre minibatch diverse
                num_edges_train = edge_label_index_train.size(1)
                perm_pos_train = torch.randperm(num_edges_train)
                training_positive_pd_edges = training_positive_pd_edges[:, perm_pos_train]
                perm_neg_train = torch.randperm(num_edges_train)
                training_negative_pd_edges = training_negative_pd_edges[:, perm_neg_train]

                num_edges_validation = edge_label_index_validation.size(1)
                perm_pos_validation = torch.randperm(num_edges_validation)
                validation_positive_pd_edges = validation_positive_pd_edges[:, perm_pos_validation]
                perm_neg_validation = torch.randperm(num_edges_validation)
                validation_negative_pd_edges = validation_negative_pd_edges[:, perm_neg_validation]

                train_losses = []
                val_losses = []
                for iteration in tqdm.tqdm(
                        range(int(np.ceil(training_positive_pd_edges.size(1) / self.args.batch_size))),
                        desc="Mini-batch"):
                    loss_train, embeddings = self.run_minibatch_transductive(self.dataset, iteration,
                                                                             training_positive_pd_edges,
                                                                             training_negative_pd_edges, all_walks=None)
                    train_losses.append(loss_train)
                    self.optimizer.zero_grad()
                    loss_train.backward()
                    self.optimizer.step()
                train_loss_final = torch.mean(torch.tensor(train_losses))

                self.model.eval()
                with torch.no_grad():
                    for iteration in tqdm.tqdm(
                            range(int(np.ceil(validation_positive_pd_edges.size(1) / self.args.batch_size))),
                            desc="Mini-batch"):
                        loss_validation, _ = self.run_minibatch_transductive(self.dataset, iteration,
                                                                             validation_positive_pd_edges,
                                                                             validation_negative_pd_edges,
                                                                             all_walks=None)

                        val_losses.append(loss_validation)
                    t_end = time.time()
                    val_loss_final = torch.mean(torch.tensor(val_losses))
                    print('Epoch {:05d} |Train loss {:.4f} | Val Loss {:.4f} | Time(s) {:.4f}'.format(
                        epoch, train_loss_final.item(), val_loss_final.item(), t_end - t_start))
                    # early stopping

                    early_stopping(val_loss_final, self.model, self.optimizer, epoch)
                    if early_stopping.early_stop:
                        print('Early stopping!')
                        break

        if self.args.test:
            self.model.eval()
            with torch.no_grad():
                stringa = f'lr-{self.args.lr}_heads-{self.args.heads}_batch-{self.args.batch_size}_cores-{self.args.n_cores}_key-{self.args.n_keys_hubs}_top-{self.args.n_top_hubs}_aggrcore-{self.args.core_aggregation}_aggrkeys-{self.args.key_aggregation}_aggrtop-{self.args.top_aggregation}_allagg-{self.args.all_aggregation}_loss-{args.loss}'
                stringa = 'unique_' + self.args.dataset + '_' + stringa
                if args.no_aug:
                    stringa = 'no_aug_unique_' + stringa

                self.test(test_positive_pd_edges, test_negative_pd_edges, 'testepoch', stringa,type=type)




if __name__ == '__main__':
    args = get_args()
    seed_torch()
    # print("------arguments-------")
    # for k, v in vars(args).items():
    #     print(k + ': ' + str(v))


    # model - ABLATION
    args = get_args()
    if args.eval_neigh or args.eval_all:
        args.eval_neigh = True
        for lr in [(12, 12, 10), (3, 3, 3), (8, 8, 5)]:
            args.n_cores = lr[0]
            args.n_keys_hubs = lr[1]
            args.n_top_hubs = lr[2]
            trainer = Trainer(args)
            trainer.run_transductive(type='trans')
            # args.test = True
            # trainer = Trainer(args)
            # trainer.run_transductive(type='trans')

    args = get_args()
    if args.eval_heads or args.eval_all:
        args.eval_heads = True
        for lr in [1,2,4,8,16]:
            args.heads = lr
            trainer = Trainer(args)
            trainer.run_transductive(type='trans')
            # args.test = True
            # trainer = Trainer(args)
            # trainer.run_transductive(type='trans')


    args = get_args()
    seed_torch(args.random_seed)
    if args.eval_lr or args.eval_all:
        args.eval_lr = True
        for lr in [0.0001,0.00001,0.00005,0.001]:
            args.lr = lr
            trainer = Trainer(args)
            trainer.run_transductive(type='trans')
            # args.test = True
            # trainer = Trainer(args)
            # trainer.run_transductive(type='trans')


    args = get_args()
    if args.eval_batch or args.eval_all:
        lrs = [64, 256, 4096]
        args.eval_batch = True
        for lr in lrs:
            args.batch_size = lr
            trainer = Trainer(args)
            trainer.run_transductive(type='trans')
            # args.test = True
            # trainer = Trainer(args)
            # trainer.run_transductive(type='trans')

    args = get_args()
    if args.eval_aggregation or args.eval_all:

        args.eval_aggregation = True
        for lr in ['gru', 'lstm','linear','mh-attention']:
            print('AGGREGATION: ',lr)
            args.core_aggregation = lr
            args.key_aggregation = lr
            args.top_aggregation = lr
            trainer = Trainer(args)
            trainer.run_transductive(type='trans')
            # args.test = True
            # trainer = Trainer(args)
            # trainer.run_transductive(type='trans')


    args = get_args()
    if args.eval_combine_aggregation or args.eval_all:
        args.eval_combine_aggregation = True
        for lr in ['mh-attention','gru', 'lstm', 'mean']:
            args.all_aggregation = lr
            trainer = Trainer(args)
            trainer.run_transductive(type='trans')
            # args.test = True
            # trainer = Trainer(args)
            # trainer.run_transductive(type='trans')





    # if args.hetgnn and args.train:
    #     # hetgnn
    #     args = get_args()
    #     args.epochs = 100
    #     args.batch_size = 1024
    #     args.core_aggregation = 'lstm'
    #     args.key_aggregation = 'lstm'
    #     args.top_aggregation = 'lstm'
    #     args.all_aggregation = 'mh-attention'
    #     args.heads = 1
    #     trainer = Trainer(args)
    #     trainer.run_transductive(type='trans')
    #     if args.dataset =='mes' and args.no_metadata == False:
    #         args.no_metadata = True
    #         trainer = Trainer(args)
    #         trainer.run_transductive(type='trans')

    if args.train:
        # losses = ["binarycross","crossentropy"]
        # if args.dataset == 'pubmed':
        #     losses = ['crossentropy']
        # for loss in ['crossentropy']:
        for dataset in ['pubmed']:
            args = get_args()
            args.dataset = dataset
            trainer = Trainer(args)
            trainer.run_transductive(type='trans')



    elif args.test:
        args = get_args()
        trainer = Trainer(args)
        type = args.inductive_type
        trainer.run_transductive(type=type)
        print('\n\n')


