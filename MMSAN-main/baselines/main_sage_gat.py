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
from torch_geometric.utils import to_networkx,from_networkx
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
parser.add_argument("-inductive_type",default='trans', choices=['trans', 'light', 'full'],
                    type=str)
parser.add_argument("-split",type=float,default=0)
parser.add_argument("-test",action='store_true')



def main(args,indices = []):
    print(args)
    # dataset
    dataset = args.dataset
    seed = args.seed
    ind_light = args.inductive_type == 'light'
    ind_full = args.inductive_type == 'full'
    seed_everything(seed)
    epochs = args.epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    root = f'./datasets/{dataset}/split_transductive/train'
    train_data, validation_data, test_data = loader.load_transductive_data(root, indices)
    if ind_full:
        train_data, validation_data, test_data = loader.load_inductive_data(root, 'full',indices)
    elif ind_light:
        train_data, validation_data, test_data = loader.load_inductive_data(root, 'light',indices)


    num_publications = train_data['publication'].num_nodes
    num_dataset = train_data['dataset'].num_nodes

    if args.split:
        number_perm = int((args.split / 100) * dataset_num)
        indices = random.sample(range(dataset_num), number_perm)
        test_data['dataset'].x[indices, :] = 1.0

    if args.model == 'sage':
        train_data = train_data.to_homogeneous().to('cpu')
        validation_data = validation_data.to_homogeneous().to('cpu')
        test_data_homo = test_data.to_homogeneous().to('cpu')


        class GraphSAGE(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels):
                super(GraphSAGE, self).__init__()
                self.conv1 = SAGEConv(in_channels, hidden_channels)
                self.conv2 = SAGEConv(hidden_channels, out_channels)
                self.dropout = torch.nn.Dropout(p=0.5)  # Dropout layer for regularization

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                #x = self.dropout(x)
                x = self.conv2(x, edge_index)
                return x

        # Creazione del modello
        model = GraphSAGE(in_channels=-1, hidden_channels=128, out_channels=128).to(device)

    elif args.model == 'gat':
        print('gat')
        print('elaboro train')
        train_data = train_data.to_homogeneous()
        print('elaboro val')
        validation_data = validation_data.to_homogeneous()
        print('elaboro test')
        test_data_homo = test_data.to_homogeneous()
        # else:
        #     tdx = to_networkx(train_data,node_attrs=["x"])
        #     train_data1 = from_networkx(tdx)
        #     print(train_data1)
        #     train_data = train_data.to_homogeneous()
        #     print(train_data)
        #     #print(train_data.edge_index)
        #     tdx = to_networkx(train_data,node_attrs=["x"])
        #     validation_data = from_networkx(tdx)
        #     tdx = to_networkx(test_data,node_attrs=["x"])
        #     test_data_homo = from_networkx(tdx)


        class GAT(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels):
                super(GAT, self).__init__()
                self.conv1 = GATConv(in_channels, hidden_channels)
                self.conv2 = GATConv(hidden_channels, out_channels)
                self.dropout = torch.nn.Dropout(p=0.5)  # Dropout layer for regularization

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                #x = self.dropout(x)
                x = self.conv2(x, edge_index)
                return x

        # Creazione del modello
        model = GAT(in_channels=-1, hidden_channels=128, out_channels=128).to(device)


    elif args.model == 'vgae':
        if args.dataset == 'mes':
            train_data = train_data.to_homogeneous()
            validation_data = validation_data.to_homogeneous()
            test_data_homo = test_data.to_homogeneous()
        else:
            tdx = to_networkx(train_data).to_undirected()
            train_data = from_networkx(tdx)
            print(train_data)
            tdx = to_networkx(train_data).to_undirected()
            validation_data = from_networkx(tdx)
            print(train_data)
            tdx = to_networkx(test_data).to_undirected()
            test_data_homo = from_networkx(tdx)
            print(train_data)

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

        edge_label_index = data.edge_label_index
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        y_true_test = np.array(
            [1] * int(data.edge_label_index.size(1) / 2) + [0] * int(data.edge_label_index.size(1) / 2))
        y_true_test = torch.tensor(y_true_test).to(device)

        link_pred = torch.sigmoid((h_src * h_dst).sum(dim=-1))  # Inner product.
        roc_auc = roc_auc_score(y_true_test.cpu().numpy().astype(numpy.float32),
                                link_pred.cpu().numpy().astype(numpy.float32))
        threshold = 0.5
        # Converte i valori continui in predizioni binarie
        predicted_probs = link_pred.cpu().numpy().astype(numpy.float32).tolist()

        predicted_labels = [1 if prob >= threshold else 0 for prob in predicted_probs]


        f1 = f1_score(y_true_test.cpu().numpy().astype(numpy.float32), predicted_labels)
        line = f'AUC {roc_auc} F1 {f1}'
        print(line)
        #return roc_auc,f1
        if args.inductive_type == 'trans':
            edge_label_index_positive = test_data['publication', 'cites', 'dataset'].edge_label_index_test_trans
        if args.inductive_type == 'light':
            edge_label_index_positive = test_data['publication', 'cites', 'dataset'].edge_label_index_test_semi
        if args.inductive_type == 'full':
            edge_label_index_positive = test_data['publication', 'cites', 'dataset'].edge_label_index_test_ind
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


        for topk in [5,10]:
            print(f'topk: {topk}')
            precision = 0
            recall_0 = 0
            ndcg = 0
            for source in sources:
                true = y_test_true_labels[source]
                pred = y_test_predicted_labels[source][:topk]
                precision += len(list(set(pred).intersection(true))) / topk
                recall_0 += len(list(set(pred).intersection(true))) / len(true)
                ndcg += ndcg_at_k(true, pred, topk)
            #line = f'\nTOP {topk} - P {precision/len(sources)} R {recall_0/len(sources)} NDCG {ndcg/len(sources)}'
            #f.write(line)
            print(f'{str(precision / len(sources))} & {str(recall_0 / len(sources))} & {str(ndcg / len(sources))}')

        return roc_auc, f1,  precision / len(sources), recall_0 / len(sources), ndcg / len(sources)


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
    test(test_data)
    # test_rec()

    # print(f'auc: {auc:4f}, ap: {ap:4f}')
    # if args.no_metadata in [25,50,75]:
    #     return auc, ap, precision, recall, ndcg


if __name__ == '__main__':
        args = parser.parse_args()


        datasets = ['pubmed']
        for dataset in datasets:
            print('\n\n\n')
            print(f'dataset: {dataset}')
            for model in ['sage']:
                print(f'model: {model}')
                print('\n\n\n')
                args.model = model
                args.dataset = dataset
                if args.test:

                        for lr in [0.00001]:
                            for epochs in [200]:
                                args.epochs = epochs
                                print(lr, epochs)
                                for me in [50,75]:
                                    print(f'split: {me}')
                                    args.split = me

                                    root = f'./datasets/{args.dataset}/split_transductive/train'
                                    data = ScholarlyDataset(root=root)
                                    train_data = data[0]
                                    dataset_num = train_data['dataset'].x.shape[0]


                                    number_perm = int((args.split / 100) * dataset_num)
                                    indices = random.sample(range(dataset_num), number_perm)
                                    print('indices',len(indices))
                                    print(args)
                                    args.lr = lr
                                    print('TRANS')
                                    args.inductive_type = 'trans'
                                    main(args, indices)
                                    # print('SEMI')
                                    # args.inductive_type = 'light'
                                    # main(args, indices)
                                    # print('IND')
                                    # args.inductive_type = 'full'
                                    # main(args, indices)

                                    print('\n\n')


