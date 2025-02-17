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
from loader_old import ScholarlyDataset
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
import loader_old
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
    inductive = 'transductive'
    if ind_light:
        inductive = 'light'
    elif ind_full:
        inductive = 'full'

    stringa = f'no_aug_{args.iteration}_{inductive}_{dataset}_{lr}_{epochs}_{args.no_metadata}'
    f = open(f'baselines/sage/results/{args.model}/{stringa}.txt', 'w')
    root = f'./datasets/{dataset}/split_transductive/train'
    data = ScholarlyDataset(root=root)
    train_data = data[0]
    dataset_num = train_data['dataset'].x.shape[0]


    train_data, validation_data, test_data = loader_old.load_transductive_data(root, indices)
    if ind_full:
        train_data, validation_data, test_data = loader_old.load_inductive_data(root, 'full',indices)
    elif ind_light:
        train_data, validation_data, test_data = loader_old.load_inductive_data(root, 'light',indices)

    num_publications = train_data['publication'].num_nodes
    num_dataset = train_data['dataset'].num_nodes
    print('eccomi')
    if args.model == 'sage':
        train_data = train_data.to_homogeneous()
        validation_data = validation_data.to_homogeneous()
        test_data_homo = test_data.to_homogeneous()

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



        # model = GraphSAGE(
        #     in_channels=-1,
        #     hidden_channels=128,
        #     num_layers=2,
        #     out_channels=128
        # ).to(device)


    elif args.model == 'gat':
        print('gat')
        train_data = train_data.to_homogeneous()
        validation_data = validation_data.to_homogeneous()
        test_data_homo = test_data.to_homogeneous()
        print(train_data)
        # model = GAT(
        #     in_channels=-1,
        #     hidden_channels=128,
        #     num_layers=2,
        #     out_channels=128,dropout=0.7
        # ).to(device)
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

        # class GAT(torch.nn.Module):
        #     def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.6):
        #         super(GAT, self).__init__()
        #         self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        #         self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        #         self.dropout = dropout
        #
        #     def forward(self, x, edge_index):
        #         x = F.dropout(x, p=self.dropout, training=self.training)
        #         x = self.conv1(x, edge_index)
        #         x = F.elu(x)
        #         x = F.dropout(x, p=self.dropout, training=self.training)
        #         x = self.conv2(x, edge_index)
        #         return F.log_softmax(x, dim=1)
        # model = GAT(in_channels=-1,hidden_channels=128,out_channels=128,heads=8).to(device)
        # model = to_hetero(model, train_data.metadata())

    elif args.model == 'vgae':
        train_data = train_data.to_homogeneous()
        validation_data = validation_data.to_homogeneous()
        test_data_homo = test_data.to_homogeneous()

        class VariationalGCNEncoder(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.conv1 = GCNConv(in_channels, 2 * out_channels)
                self.conv_mu = GCNConv(2 * out_channels, out_channels)
                self.conv_logstd = GCNConv(2 * out_channels, out_channels)

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index).relu()
                return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

        model = VGAE(VariationalGCNEncoder(in_channels=-1, out_channels=128)).to(device)

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
    def test(test_data):
        model.eval()

        data = test_data_homo.to(device)
        h = model(data.x, data.edge_index)

        # train_data = train_dataset.to(device)
        #
        # link_pred1 = (h[train_data.edge_label_index[0]] * h[train_data.edge_label_index[1]])
        # nodes = link_pred1.detach().cpu()
        # labels = train_data.edge_label
        # labels = labels.detach().clone().cpu()
        # # print(nodes)
        # # print(labels)
        # clf.fit(nodes.numpy().astype(numpy.float32), labels.numpy().astype(numpy.float32))

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
        # print(y_true_test.cpu().numpy().astype(numpy.float32))
        # print(predicted_labels)
        f1 = f1_score(y_true_test.cpu().numpy().astype(numpy.float32), predicted_labels)

        edge_label_index_positive = test_data['publication', 'cites', 'dataset'].edge_label_index_test
        publications_embeddings = h[:num_publications]
        datasets_embeddings = h[num_publications:num_publications + num_dataset]
        h_src = publications_embeddings[sorted(list(set(edge_label_index_positive[0].tolist()))), :]
        h_dst = datasets_embeddings
        final_matrix = torch.matmul(h_src, h_dst.t())
        print(final_matrix.shape)

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

        for topk in [5]:
            precision = 0
            recall_0 = 0
            ndcg = 0
            print('NO RERANK')
            for source in sources:
                true = y_test_true_labels[source]
                pred = y_test_predicted_labels[source][:topk]
                # print(true)
                # print(pred)
                # print(len(list(set(pred).intersection(true))) / topk)
                # print(len(list(set(pred).intersection(true))) / len(true))
                # print('\n\n')
                precision += len(list(set(pred).intersection(true))) / topk
                recall_0 += len(list(set(pred).intersection(true))) / len(true)
                ndcg += ndcg_at_k(true, pred, topk)
            line = f'\nTOP {topk} - P {precision/len(sources)} R {recall_0/len(sources)} NDCG {ndcg/len(sources)}'
            f.write(line)
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

        # print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, val loss: {loss_val:.4f}')
        # print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

    # print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
    auc, ap, precision, recall, ndcg = test(test_data)
    # test_rec()

    print(f'auc: {auc:4f}, ap: {ap:4f}')
    if args.no_metadata in [25,50,75]:
        return auc, ap, precision, recall, ndcg


if __name__ == '__main__':
        args = parser.parse_args()



        datasets = ['mes']
        for dataset in datasets:
            for model in ['sage']:
                args.model = model
                args.dataset = dataset
                if args.test:
                    for epochs in [100]:
                        args.epochs = epochs
                        for lr in [0.00001]:
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
                                args.ind_full = False
                                args.ind_light = False
                                main(args, indices)
                                args.ind_light = True
                                args.ind_full = False
                                main(args, indices)
                                args.ind_full = True
                                args.ind_light = False
                                main(args, indices)


                            if dataset == 'mess':
                                args.epochs = epochs
                                args.dataset = 'mes'
                                args.lr = lr
                                root = f'./datasets/{args.dataset}/split_transductive/train'
                                data = ScholarlyDataset(root=root)
                                train_data = data[0]
                                dataset_num = train_data['dataset'].x.shape[0]
                                for me in [25,50,75]:
                                    auc, ap, prec, rec, ndcg = [], [], [], [], []
                                    for j in range(1,11):
                                        random.seed(j)
                                        print(f'iteration {j} nometa {me} dataset {args.dataset}')
                                        args.iteration = j
                                        args.no_metadata = me
                                        number_perm = int((args.no_metadata / 100) * dataset_num)
                                        indices = random.sample(range(dataset_num), number_perm)
                                        print(len(indices))
                                        args.ind_full = False
                                        args.ind_light = False
                                        auc_tmp, ap_tmp, precision_tmp, recall_tmp, ndcg_tmp = main(args, indices)
                                        auc.append(auc_tmp)
                                        ap.append(ap_tmp)
                                        prec.append(precision_tmp)
                                        rec.append(recall_tmp)
                                        ndcg.append(ndcg_tmp)
                                    inductive = 'transductive'
                                    if args.ind_light:
                                        inductive = 'light'
                                    elif args.ind_full:
                                        inductive = 'full'
                                    stringa = f'no_aug_{inductive}_{dataset}_{epochs}_{lr}'
                                    gg = open(f'baselines/sage/results/{args.model}/bootstrapped_{me}_{stringa}.txt', 'w')
                                    gg.write(f"AP {sum(ap) / len(ap)} AUC {sum(auc) / len(ap)}\n P {sum(prec)/len(ap)} R {sum(rec) / len(ap)} NDCG {sum(ndcg) / len(ap)}\n")
                                    gg.write(f"AP {np.std(ap) / np.sqrt(len(ap))} AUC {np.std(auc) / np.sqrt(len(ap))}\n P {np.std(prec)/np.sqrt(len(ap))} R {np.std(rec) / np.sqrt(len(ap))} NDCG {np.std(ndcg) / np.sqrt(len(ap))}")
                                    gg.close()

                                    auc, ap, prec, rec, ndcg = [], [], [], [], []
                                    for j in range(1,11):
                                        random.seed(j)
                                        print(f'iteration {j} nometa {me} dataset {args.dataset}')
                                        args.iteration = j
                                        args.no_metadata = me
                                        number_perm = int((args.no_metadata / 100) * dataset_num)
                                        indices = random.sample(range(dataset_num), number_perm)
                                        print(len(indices))
                                        args.ind_light = True
                                        args.ind_full = False
                                        auc_tmp, ap_tmp, precision_tmp, recall_tmp, ndcg_tmp = main(args, indices)
                                        auc.append(auc_tmp)
                                        ap.append(ap_tmp)
                                        prec.append(precision_tmp)
                                        rec.append(recall_tmp)
                                        ndcg.append(ndcg_tmp)
                                    inductive = 'transductive'
                                    if args.ind_light:
                                        inductive = 'light'
                                    elif args.ind_full:
                                        inductive = 'full'
                                    stringa = f'no_aug_{inductive}_{dataset}_{epochs}_{lr}'
                                    gg = open(f'baselines/sage/results/{args.model}/bootstrapped_{me}_{stringa}.txt', 'w')
                                    gg.write(
                                        f"AP {sum(ap) / len(ap)} AUC {sum(auc) / len(ap)}\n P {sum(prec) / len(ap)} R {sum(rec) / len(ap)} NDCG {sum(ndcg) / len(ap)}\n")
                                    gg.write(
                                        f"AP {np.std(ap) / np.sqrt(len(ap))} AUC {np.std(auc) / np.sqrt(len(ap))}\n P {np.std(prec) / np.sqrt(len(ap))} R {np.std(rec) / np.sqrt(len(ap))} NDCG {np.std(ndcg) / np.sqrt(len(ap))}")
                                    gg.close()


                                    auc, ap, prec, rec, ndcg = [], [], [], [], []
                                    for j in range(1,11):
                                        random.seed(j)
                                        print(f'iteration {j} nometa {me} dataset {args.dataset}')
                                        args.iteration = j
                                        args.no_metadata = me
                                        number_perm = int((args.no_metadata / 100) * dataset_num)
                                        indices = random.sample(range(dataset_num), number_perm)
                                        print(len(indices))

                                        args.ind_full = True
                                        args.ind_light = False
                                        auc_tmp, ap_tmp, precision_tmp, recall_tmp, ndcg_tmp = main(args, indices)
                                        auc.append(auc_tmp)
                                        ap.append(ap_tmp)
                                        prec.append(precision_tmp)
                                        rec.append(recall_tmp)
                                        ndcg.append(ndcg_tmp)
                                    inductive = 'transductive'
                                    if args.ind_light:
                                        inductive = 'light'
                                    elif args.ind_full:
                                        inductive = 'full'
                                    stringa = f'no_aug_{inductive}_{dataset}_{epochs}_{lr}'
                                    gg = open(f'baselines/sage/results/{args.model}/bootstrapped_{me}_{stringa}.txt', 'w')
                                    gg.write(
                                        f"AP {sum(ap) / len(ap)} AUC {sum(auc) / len(ap)}\n P {sum(prec) / len(ap)} R {sum(rec) / len(ap)} NDCG {sum(ndcg) / len(ap)}\n")
                                    gg.write(
                                        f"AP {np.std(ap) / np.sqrt(len(ap))} AUC {np.std(auc) / np.sqrt(len(ap))}\n P {np.std(prec) / np.sqrt(len(ap))} R {np.std(rec) / np.sqrt(len(ap))} NDCG {np.std(ndcg) / np.sqrt(len(ap))}")
                                    gg.close()




                # no metadata trans
                # args.ind_full = False
                # args.ind_light = False
                # args.no_metadata = 100
                # main(args)





    # args = parser.parse_args()
    # main(args)