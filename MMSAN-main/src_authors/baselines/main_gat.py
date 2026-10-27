import os.path as osp
import time
import numpy
import torch
import torch.nn.functional as F
from torch.nn import BCELoss
import tqdm
from torch_geometric.nn import SAGEConv, to_hetero, GATConv

import torch.nn as nn
from torch_geometric.loader import NeighborLoader
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from torch_geometric.data import Batch
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader, LinkNeighborLoader, NeighborLoader
from torch_geometric.nn import GAT,GraphSAGE
from torch_geometric.utils import negative_sampling
import os.path as osp
from collections import Counter
import pandas as pd
from loader import ScholarlyDataset
import torch
from sentence_transformers import SentenceTransformer
from torch_geometric import seed_everything
from torch_geometric.data import HeteroData, download_url, extract_zip, InMemoryDataset
from torch_geometric.data import Data, InMemoryDataset
import utils
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
import argparse
from torch_geometric.nn import FastRGCNConv, RGCNConv,SAGEConv
import torch_geometric.transforms as T
from torch_geometric.nn import GAE, VGAE, GCNConv
parser = argparse.ArgumentParser()
import pandas as pd
import multiprocessing as mp
import json
import random
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import loader
from torch_geometric import seed_everything


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



parser.add_argument("-model", default='gat')
parser.add_argument("-dataset",default='mes', choices=['mes', 'pubmed', 'pubmed_kcore'],
                    type=str)

parser.add_argument("-seed", default=42, type=int)
parser.add_argument("-iteration", default=0, type=int)
parser.add_argument("-lr", default=0.00001, type=float) # gat = 0.001,200 #sage = 0.0001, 100
parser.add_argument("-epochs", default=100)
parser.add_argument("-ind_light",action="store_true")
parser.add_argument("-ind_full",action="store_true")
parser.add_argument("-no_metadata",type=int,default=0)
parser.add_argument("-test",action='store_true')



def main(args,indices = []):
    # dataset
    print(args)
    dataset = args.dataset
    lr = args.lr
    seed = args.seed
    ind_light = args.ind_light
    ind_full = args.ind_full
    seed_everything(seed)
    epochs = args.epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        

    stringa = f'unique_{args.iteration}_{dataset}_{lr}_{epochs}_{args.no_metadata}'
    f = open(f'baselines/sage/results/{args.model}/{stringa}.txt', 'w')
    root = f'./datasets/{dataset}/split_transductive/train'
    data = ScholarlyDataset(root=root)
    train_data = data[0]
    dataset_num = train_data['dataset'].x.shape[0]
    print(f'indices: {indices[0:10]}')
    train_data, validation_data, test_data_trans, test_data_semi, test_data_ind = loader.load_data(root, indices)


    num_publications = train_data['publication'].num_nodes
    num_dataset = train_data['dataset'].num_nodes
    print('eccomi')
    train_data = train_data.to_homogeneous()
    validation_data = validation_data.to_homogeneous()
    if args.model == 'sage':


        class GraphSAGE(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels):
                super(GraphSAGE, self).__init__()
                self.conv1 = SAGEConv(in_channels, hidden_channels)
                self.conv2 = SAGEConv(hidden_channels, out_channels)
                self.dropout = torch.nn.Dropout(p=0.5)  # Dropout layer for regularization

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = self.dropout(x)
                x = self.conv2(x, edge_index)
                return x

        # Creazione del modello
        model = GraphSAGE(in_channels=-1, hidden_channels=128, out_channels=128).to(device)


    elif args.model == 'gat':
        print('gat')

        class GAT(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels):
                super(GAT, self).__init__()
                self.conv1 = GATConv(in_channels, hidden_channels)
                self.conv2 = GATConv(hidden_channels, out_channels)
                self.dropout = torch.nn.Dropout(p=0.5)  # Dropout layer for regularization

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = self.dropout(x)
                x = self.conv2(x, edge_index)
                return x
        model = GAT(in_channels=-1, hidden_channels=128, out_channels=128).to(device)



    print(args.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def train(train_data):
        model.train()


        data = train_data.to(device)

        h = model(data.x, data.edge_index)
        h_src = h[data.edge_label_index[0]]
        h_dst = h[data.edge_label_index[1]]
        link_pred = (h_src * h_dst).sum(dim=-1).to(device)
        y_true_test = np.array(
            [1] * int(data.edge_label_index.size(1) / 2) + [0] * int(data.edge_label_index.size(1) / 2))
        y_true_test = torch.tensor(y_true_test).to(device)

        loss = F.binary_cross_entropy_with_logits(link_pred, y_true_test.float())
        loss.backward()
        optimizer.step()

        total_loss = float(loss)
        return total_loss

    @torch.no_grad()
    def validation(validation_data):
        model.train()
        data = validation_data.to(device)
        h = model(data.x, data.edge_index)
        h_src = h[data.edge_label_index[0]]
        h_dst = h[data.edge_label_index[1]]
        link_pred = (h_src * h_dst).sum(dim=-1).to(device)
        y_true_test = np.array(
            [1] * int(data.edge_label_index.size(1) / 2) + [0] * int(data.edge_label_index.size(1) / 2))
        y_true_test = torch.tensor(y_true_test).to(device)
        loss = F.binary_cross_entropy_with_logits(link_pred, y_true_test.float())

        total_loss = float(loss)
        return total_loss

    @torch.no_grad()
    def test(test_data,type='trans'):
        print('TYPE ',type)
        model.eval()
        test_data_homo = test_data.to_homogeneous()
        data = test_data_homo.to(device)

        h = model(data.x, data.edge_index)

        edge_label_index = data.edge_label_index
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        y_true_test = np.array(
            [1] * int(data.edge_label_index.size(1) / 2) + [0] * int(data.edge_label_index.size(1) / 2))
        y_true_test = torch.tensor(y_true_test).to(device)

        link_pred = (h_src * h_dst).sum(dim=-1)  # Inner product.
        roc_auc = roc_auc_score(y_true_test.cpu().numpy().astype(numpy.float32),
                                link_pred.cpu().numpy().astype(numpy.float32))
        ap = average_precision_score(y_true_test.cpu().numpy().astype(numpy.float32),
                                     link_pred.cpu().numpy().astype(numpy.float32))
        line = f'AUC {roc_auc} AP {ap}'
        f.write(line)
        threshold = 0.5

        # Converte i valori continui in predizioni binarie
        predicted_probs = link_pred.cpu().numpy().astype(numpy.float32).tolist()
        predicted_labels = [1 if prob >= threshold else 0 for prob in predicted_probs]
        f1 = f1_score(y_true_test.cpu().numpy().astype(numpy.float32), predicted_labels)

        if type == 'trans':
            edge_label_index_positive = test_data['publication', 'cites', 'dataset'].edge_label_index_test_trans
        elif type == 'ind':
            edge_label_index_positive = test_data['publication', 'cites', 'dataset'].edge_label_index_test_ind
        elif type == 'semi':
            edge_label_index_positive = test_data['publication', 'cites', 'dataset'].edge_label_index_test_semi

        publications_embeddings = h[:num_publications]
        datasets_embeddings = h[num_publications:num_publications + num_dataset]
        h_src = publications_embeddings[sorted(list(set(edge_label_index_positive[0].tolist()))), :]
        h_dst = datasets_embeddings
        final_matrix = torch.matmul(h_src, h_dst.t())
        sources = edge_label_index_positive[0].tolist()
        targets = edge_label_index_positive[1].tolist()

        y_test_true_labels = {source: [] for source in sources}

        for source, target in zip(sources, targets):
            y_test_true_labels[source].append(target)

        y_test_true_labels = {k: y_test_true_labels[k] for k in sorted(list(y_test_true_labels.keys()))}

        top_values, top_indices = torch.topk(final_matrix, k=num_dataset, dim=1)
        sources = sorted(list(y_test_true_labels.keys()))
        y_test_predicted_labels = {source: [] for source in sources}
        y_test_predicted_values = {source: [] for source in sources}
        for i, lista in enumerate(top_indices.tolist()):
            y_test_predicted_labels[sources[i]] = lista
            y_test_predicted_values[sources[i]] = top_values[i]

        for topk in [1,5, 10]:
            precision = 0
            recall_0 = 0
            ndcg = 0
            print('NO RERANK')
            for source in sources:
                true = y_test_true_labels[source]
                pred = y_test_predicted_labels[source][:topk]
                precision += len(list(set(pred).intersection(true))) / topk
                recall_0 += len(list(set(pred).intersection(true))) / len(true)
                ndcg += ndcg_at_k(true, pred, topk)
            f.write(f'TYPE: {type}\n')
            line = f'\nTOP {topk} - P {precision/len(sources)} R {recall_0/len(sources)} NDCG {ndcg/len(sources)}'
            f.write(line)

            print(f'AUC {roc_auc} AP {ap}')
            print(f'type: {type}')
            print(f'{str(precision / len(sources))} & {str(recall_0 / len(sources))} & {str(ndcg / len(sources))}')

        return roc_auc, ap,  precision / len(sources), recall_0 / len(sources), ndcg / len(sources)


    # in this setup we consider completely new publications

    max_epochs = 0
    best_val = float('inf')
    print(epochs)
    for epoch in range(1, (int(epochs)+1)):
        loss = train(train_data)
        loss_val = validation(validation_data)
        if loss_val < best_val:
            best_val = loss_val
            max_epochs = 0
        else:
            max_epochs += 1
        if max_epochs == 100:
            break

        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, val loss: {loss_val:.4f}')
        # print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

    # print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
    auc_trans, ap_trans, precision_trans, recall_trans, ndcg_trans = test(test_data_trans,'trans')
    auc_semi, ap_semi, precision_semi, recall_semi, ndcg_semi = test(test_data_semi,'semi')
    auc_ind, ap_ind, precision_ind, recall_ind, ndcg_ind = test(test_data_ind,'ind')
    f.close()
    
    auc = [auc_trans,auc_semi,auc_ind]
    ap = [ap_trans,ap_semi,ap_ind]
    precision = [precision_trans,precision_semi,precision_ind]
    recall = [recall_trans,recall_semi,recall_ind]
    ndcg = [ndcg_trans,ndcg_semi,ndcg_ind]

    if args.no_metadata in [25,50,75]:
        return auc, ap, precision, recall, ndcg


if __name__ == '__main__':
        args = parser.parse_args()

        datasets = ['pubmed','mes']
        for dataset in datasets:
            print(f'dataset: {dataset}')
            for model in ['gat','sage']:
                args.model = model
                args.dataset = dataset
                if args.test:
                    for epochs in [100,150,200]:
                        args.epochs = epochs
                        for lr in [0.00001,0.0001]:
                            for me in [0,100]:
                                 args.no_metadata = me
                                 root = f'./datasets/{args.dataset}/split_transductive/train'
                                 data = ScholarlyDataset(root=root)
                                 train_data = data[0]
                                 dataset_num = train_data['dataset'].x.shape[0]
                                 indices = [i for i in range(dataset_num)]
                                 if me == 0:
                                     indices = []
                                 args.lr = lr
                                 main(args,indices)


                            if dataset in ['mes']:
                                args.epochs = epochs
                                args.dataset = 'mes'
                                args.lr = lr
                                stringa = f'unique_{dataset}_{epochs}_{lr}'
                                root = f'./datasets/{args.dataset}/split_transductive/train'
                                data = ScholarlyDataset(root=root)
                                train_data = data[0]
                                dataset_num = train_data['dataset'].x.shape[0]
                                for me in [25,50,75]:
                                    gg = open(f'baselines/sage/results/{args.model}/bootstrapped_unique_{me}_{stringa}.txt',
                                              'w')
                                    auc_trans, ap_trans, prec_trans, rec_trans, ndcg_trans = [], [], [], [], []
                                    auc_semi, ap_semi, prec_semi, rec_semi, ndcg_semi = [], [], [], [], []
                                    auc_ind, ap_ind, prec_ind, rec_ind, ndcg_ind = [], [], [], [], []
                                    
                                    for j in range(1,11):
                                        random.seed(j)
                                        stringa = f'unique_{dataset}_{epochs}_{lr}'
                                        print(f'iteration {j} nometa {me} dataset {args.dataset}')
                                        args.iteration = j
                                        args.no_metadata = me
                                        number_perm = int((args.no_metadata / 100) * dataset_num)
                                        indices = random.sample(range(dataset_num), number_perm)
                                        print(len(indices))
                                        auc_tmp, ap_tmp, precision_tmp, recall_tmp, ndcg_tmp = main(args, indices)
                                        auc_trans.append(auc_tmp[0])
                                        ap_trans.append(ap_tmp[0])
                                        prec_trans.append(precision_tmp[0])
                                        rec_trans.append(recall_tmp[0])
                                        ndcg_trans.append(ndcg_tmp[0])

                                        auc_semi.append(auc_tmp[1])
                                        ap_semi.append(ap_tmp[1])
                                        prec_semi.append(precision_tmp[1])
                                        rec_semi.append(recall_tmp[1])
                                        ndcg_semi.append(ndcg_tmp[1])

                                        auc_ind.append(auc_tmp[2])
                                        ap_ind.append(ap_tmp[2])
                                        prec_ind.append(precision_tmp[2])
                                        rec_ind.append(recall_tmp[2])
                                        ndcg_ind.append(ndcg_tmp[2])
                                 
                                    gg.write('trans')
                                    gg.write(f"AP {sum(ap_trans) / len(ap_trans)} AUC {sum(auc_trans) / len(auc_trans)}\n P {sum(prec_trans)/len(prec_trans)} R {sum(rec_trans) / len(rec_trans)} NDCG {sum(ndcg_trans) / len(ndcg_trans)}\n")
                                    gg.write(f"AP {np.std(ap_trans) / np.sqrt(len(ap_trans))} AUC {np.std(auc_trans) / np.sqrt(len(auc_trans))}\n P {np.std(prec_trans)/np.sqrt(len(prec_trans))} R {np.std(rec_trans) / np.sqrt(len(rec_trans))} NDCG {np.std(ndcg_trans) / np.sqrt(len(ndcg_trans))}")

                                    gg.write('semi')
                                    gg.write(
                                        f"AP {sum(ap_semi) / len(ap_semi)} AUC {sum(auc_semi) / len(auc_semi)}\n P {sum(prec_semi) / len(prec_semi)} R {sum(rec_semi) / len(rec_semi)} NDCG {sum(ndcg_semi) / len(ndcg_semi)}\n")
                                    gg.write(
                                        f"AP {np.std(ap_semi) / np.sqrt(len(ap_semi))} AUC {np.std(auc_semi) / np.sqrt(len(auc_semi))}\n P {np.std(prec_semi) / np.sqrt(len(prec_semi))} R {np.std(rec_semi) / np.sqrt(len(rec_semi))} NDCG {np.std(ndcg_semi) / np.sqrt(len(ndcg_semi))}")

                                    gg.write('ind')
                                    gg.write(
                                        f"AP {sum(ap_ind) / len(ap_ind)} AUC {sum(auc_ind) / len(auc_ind)}\n P {sum(prec_ind) / len(prec_ind)} R {sum(rec_ind) / len(rec_ind)} NDCG {sum(ndcg_trans) / len(ndcg_ind)}\n")
                                    gg.write(
                                        f"AP {np.std(ap_ind) / np.sqrt(len(ap_ind))} AUC {np.std(auc_ind) / np.sqrt(len(auc_ind))}\n P {np.std(prec_ind) / np.sqrt(len(prec_ind))} R {np.std(rec_ind) / np.sqrt(len(rec_ind))} NDCG {np.std(ndcg_ind) / np.sqrt(len(ndcg_ind))}")

                                    gg.close()

                                 
                                    



   