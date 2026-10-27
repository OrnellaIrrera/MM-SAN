import os.path
import networkx as nx
import argparse
parser = argparse.ArgumentParser()
from utils import create_graph_csv
import pandas as pd
import multiprocessing as mp
import json
import random
import torch
import numpy as np
from sklearn.model_selection import train_test_split


parser.add_argument("-dataset", default='mes',choices=['mes','pubmed','pubmed_kcore'],
                    type=str)
parser.add_argument("-seed",default=42,type=int)
parser.add_argument("-test_split",type=float)
parser.add_argument("-val_split",type=float)
parser.add_argument("-light",type=str)

args = parser.parse_args()
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# creare una partizione di archi facendo in modo di perdere il numero minore possibile di nodi

def get_count1():
    """overview of nodes"""

    dataset = args.dataset
    pubdata_test = pd.read_csv(f'./datasets/{args.dataset}/split_transductive/test/pubdataedges.csv')
    pubdata_vali = pd.read_csv(f'./datasets/{args.dataset}/split_transductive/validation/pubdataedges.csv')
    pubdata_train = pd.read_csv(f'./datasets/{args.dataset}/split_transductive/train/pubdataedges.csv')

    print(pubdata_train.shape)
    print(pubdata_vali.shape)
    print(pubdata_test.shape)

    tup_tr = [(row['source'],row['target']) for i,row in pubdata_train.iterrows()]
    tup_v = [(row['source'],row['target']) for i,row in pubdata_vali.iterrows()]
    tup_t = [(row['source'],row['target']) for i,row in pubdata_test.iterrows()]
    train_in_test = [t not in tup_tr for t in tup_t]
    train_in_va = [t not  in tup_tr for t in tup_v]
    test_in_va = [t not in tup_t for t in tup_v]
    print(len(train_in_test))
    print(len(train_in_va))
    print(len(test_in_va))



def get_count():

    """overview of nodes"""

    dataset = args.dataset
    pubdata_test = pd.read_csv(f'./datasets/{args.dataset}/split_transductive/test/pubdataedges.csv')
    pubdata_vali = pd.read_csv(f'./datasets/{args.dataset}/split_transductive/validation/pubdataedges.csv')
    pubdata_train = pd.read_csv(f'./datasets/{args.dataset}/split_transductive/train/pubdataedges.csv')

    print(pubdata_train.shape)
    print(pubdata_vali.shape)
    print(pubdata_test.shape)
    # publications_train = [ p for p in pubdata_train['source'].unique().tolist() if p not in pubdata_vali['source'].unique().tolist() and p not in pubdata_test['source'].unique().tolist()]
    # publications_validation = [ p for p in pubdata_vali['source'].unique().tolist() if p not in pubdata_test['source'].unique().tolist()]
    # publications_test = pubdata_test['source'].unique().tolist()
    # print(len([p for p in publications_validation if p not in publications_test and p not in publications_train]))
    # print(len([p for p in publications_test if p not in publications_validation and p not in publications_train]))
    # print(len([p for p in publications_train if p not in publications_validation and p not in publications_test]))
    # datasets_train = [ p for p in pubdata_train['target'].unique().tolist() if p not in pubdata_vali['target'].unique().tolist() and p not in pubdata_test['target'].unique().tolist()]
    # datasets_validation = [ p for p in pubdata_vali['target'].unique().tolist() if p not in pubdata_test['target'].unique().tolist()]
    # datasets_test = pubdata_test['target'].unique().tolist()
    # unique_data_test = [p for p in datasets_test if p not in datasets_validation and p not in datasets_train]
    # print(len([p for p in datasets_validation if p not in datasets_test and p not in datasets_train]))
    # print(len([p for p in datasets_test if p not in datasets_validation and p not in datasets_train]))
    # print(len([p for p in datasets_train if p not in datasets_validation and p not in datasets_test]))
    #
    # print('\n\n\n')

    # pubdata_test = pd.read_csv(f'./datasets/{args.dataset}/split_transductive/test/pubdataedges.csv')
    # pubdata_vali = pd.read_csv(f'./datasets/{args.dataset}/split_transductive/validation/pubdataedges.csv')
    # pubdata_train = pd.read_csv(f'./datasets/{args.dataset}/split_transductive/train/pubdataedges.csv')
    # test_data = pubdata_test['target'].unique().tolist()
    # vali_data = pubdata_vali['target'].unique().tolist()
    # train_data = pubdata_train['target'].unique().tolist()
    # # print(len([p for p in test_data if p not in vali_data and p not in train_data]))
    # # print(len([p for p in vali_data if p not in test_data and p not in train_data]))
    # test_data = pubdata_test['source'].unique().tolist()
    # vali_data = pubdata_vali['source'].unique().tolist()
    # train_data = pubdata_train['source'].unique().tolist()
    # print(len([p for p in test_data if p not in vali_data and p not in train_data]))
    # print(len([p for p in vali_data if p not in test_data and p not in train_data]))


    path_0_t = f'./datasets/{dataset}/split_inductive_light/train'
    path_0_v = f'./datasets/{dataset}/split_inductive_light/validation'
    path_0_te = f'./datasets/{dataset}/split_inductive_light/test'
    publications_train = pd.read_csv(path_0_t+'/pubdataedges.csv')
    publications_vali = pd.read_csv(path_0_v+'/pubdataedges.csv')
    publications_test = pd.read_csv(path_0_te+'/pubdataedges.csv')

    tup_tr = [(row['source'],row['target']) for i,row in publications_train.iterrows()]
    tup_v = [(row['source'],row['target']) for i,row in publications_vali.iterrows()]
    tup_t = [(row['source'],row['target']) for i,row in publications_test.iterrows()]
    # print(len([p for p in tup_t if p  in tup_tr or p in tup_v]))
    # print(len([p for p in tup_v if p  in tup_tr  or p  in tup_t]))
    print(len([p for p in tup_tr if not (p in tup_v or p in tup_t)]))
    print(len([p for p in tup_tr if p in tup_v  or p in tup_t]))
    print(len(tup_tr))
    print(len([p for p in tup_t if not (p in tup_tr  or p in tup_v)]))
    print(len([p for p in tup_t if p in tup_v  or p in tup_tr]))
    print(len(tup_t))
    print(len([p for p in tup_v if not (p in tup_tr)]))
    print(len([p for p in tup_v if p in tup_t]))
    print(len(tup_v))





    # publications_train = pd.read_csv(path_0_t+'/publications.csv')['id'].tolist()
    # publications_vali = pd.read_csv(path_0_v+'/publications.csv')['id'].tolist()
    # publications_test = pd.read_csv(path_0_te+'/publications.csv')['id'].tolist()
    # print(len([p for p in publications_vali if p not in publications_train ]))
    # print(len([p for p in publications_test if p not in publications_train and p not in publications_vali]))
    #
    # # print(publications_train.shape[0],publications_vali.shape[0],publications_test.shape[0])
    # publications_train = pd.read_csv(path_0_t+'/datasets.csv')['id'].tolist()
    # publications_vali = pd.read_csv(path_0_v+'/datasets.csv')['id'].tolist()
    # publications_test = pd.read_csv(path_0_te+'/datasets.csv')['id'].tolist()
    # print(len([p for p in publications_vali if p not in publications_train ]))
    # print(len([p for p in publications_test if p not in publications_train and p not in publications_vali]))
    # print(publications_train.shape[0],publications_vali.shape[0],publications_test.shape[0])
    # publications_train = pd.read_csv(path_0_t+'/pubdataedges.csv')
    # publications_vali = pd.read_csv(path_0_v+'/pubdataedges.csv')
    # publications_test = pd.read_csv(path_0_te+'/pubdataedges.csv')
    #
    # ptr = publications_train['source'].unique().tolist()
    # pv = publications_vali['source'].unique().tolist()
    # pt = publications_test['source'].unique().tolist()
    #
    # print(len(ptr),len(pv),len(pt))
    #
    # print(len(list(set(pv).intersection(set(pt)))))
    # print(len(list(set(pv).union(set(pt)))))
    #
    # ptr = publications_train['target'].unique().tolist()
    # pv = publications_vali['target'].unique().tolist()
    # pt = publications_test['target'].unique().tolist()
    #
    # print(len(ptr),len(pv),len(pt))
    # print(len(list(set(pv).intersection(set(pt)))))
    # print(len(list(set(pv).union(set(pt)))))
    #
    #
    # publications_train = pd.read_csv(path_0_t+'/pubdataedges.csv')
    # publications_vali = pd.read_csv(path_0_v+'/pubdataedges.csv')
    # publications_test = pd.read_csv(path_0_te+'/pubdataedges.csv')
    # tup_tr = [(row['source'],row['target']) for i,row in publications_train.iterrows()]
    # tup_v = [(row['source'],row['target']) for i,row in publications_vali.iterrows()]
    # tup_t = [(row['source'],row['target']) for i,row in publications_test.iterrows()]
    # print(any(t in tup_v for t in tup_tr))
    # print(any(t in tup_t for t in tup_tr))
    # print(any(t in tup_t for t in tup_v))


    # path_1 = f'./datasets/{dataset}/split_inductive_light/train/pubdataedges.csv'
    # G_train = create_graph_csv('split_inductive_light/train',dataset,['all'])
    # nodes = list(G_train.nodes())
    # path_2 = f'./datasets/{dataset}/split_inductive_full/train/pubdataedges.csv'
    # csv_0 = pd.read_csv(path_0_t)
    # pubs_vali = pd.read_csv(path_0_v)['source'].unique().tolist()
    # pubs_test = pd.read_csv(path_0_te)['source'].unique().tolist()
    #
    # pubs_train = pd.read_csv(path_1)['source'].unique().tolist()
    # print(len(list(set(pubs_train).intersection(set(pubs_vali+pubs_test))))) # deve essere 0
    # pubs_train = pd.read_csv(path_2)['source'].unique().tolist()
    # print(len(list(set(pubs_train).intersection(set(pubs_vali+pubs_test))))) # deve essere 0
    #
    #
    # pubs_vali = pd.read_csv(path_0_v)['target'].unique().tolist()
    # pubs_test = pd.read_csv(path_0_te)['target'].unique().tolist()
    #
    # pubs_train = pd.read_csv(path_1)['target'].unique().tolist()
    # print(len(set(pubs_vali+pubs_test)))
    # print(len(list(set(nodes).intersection(set(pubs_vali+pubs_test))))) # deve essere 0
    # print(len(list(set(pubs_train).intersection(set(pubs_vali+pubs_test))))) # deve essere 0
    # pubs_train = pd.read_csv(path_2)['target'].unique().tolist()
    # print(len(list(set(nodes).intersection(set(pubs_vali+pubs_test))))) # deve essere 0
    # print(len(list(set(pubs_train).intersection(set(pubs_vali+pubs_test))))) # deve essere 0


    # csv_1 = pd.read_csv(path_1)
    # csv_2 = pd.read_csv(path_2)
    # print(csv_0.shape[0],csv_1.shape[0],csv_2.shape[0])
    #
    #
    #
    # path_train = f'./datasets/{dataset}/split_transductive/train'
    # path_train_light = f'./datasets/{dataset}/split_inductive_light/train'
    # path_train_full = f'./datasets/{dataset}/split_inductive_full/train'
    #
    # path_vali = f'./datasets/{dataset}/split_transductive/validation/pubdataedges.csv'
    # path_vali_light = f'./datasets/{dataset}/split_inductive_light/validation'
    # path_vali_full = f'./datasets/{dataset}/split_inductive_full/validation'
    #
    # path_test = f'./datasets/{dataset}/split_transductive/test/pubdataedges.csv'
    # path_test_light = f'./datasets/{dataset}/split_inductive_light/validation'
    # path_test_full = f'./datasets/{dataset}/split_inductive_full/validation'
    #
    # pubs = pd.read_csv(path_train + '/publications.csv')
    # pubs_light = pd.read_csv(path_train_light + '/publications.csv')
    # pubs_full = pd.read_csv(path_train_full + '/publications.csv')
    # print(pubs.shape[0],pubs_full.shape[0],pubs_light.shape[0])
    #
    # pubs = pd.read_csv(path_vali)['source'].unique().tolist()
    # pubs_light = pd.read_csv(path_vali_light + '/publications.csv')
    # pubs_full = pd.read_csv(path_vali_full + '/publications.csv')
    # print(len(pubs),pubs_full.shape[0],pubs_light.shape[0])
    #
    # pubs = pd.read_csv(path_test)['source'].unique().tolist()
    # pubs_light = pd.read_csv(path_test_light + '/publications.csv')
    # pubs_full = pd.read_csv(path_test_full + '/publications.csv')
    # print(len(pubs),pubs_full.shape[0],pubs_light.shape[0])


    # for path in [path_train]:
    #     for file in os.listdir(path):
    #         print(file)
    #         if 'csv' in file and 'target' not in file:
    #             csv = pd.read_csv(path+'/'+file)
    #             csv_light = pd.read_csv(path_train_light+'/'+file)
    #             csv_full = pd.read_csv(path_train_full+'/'+file)
    #             print(f'filename: {file}')
    #             print(f'row count: {csv.shape[0]},{csv_light.shape[0]},{csv_full.shape[0]}')
import shutil
def copy_trans():
    dataset = args.dataset
    path = f'./datasets/{dataset}/split_transductive/train'
    pathv = f'./datasets/{dataset}/split_transductive/validation'
    patht = f'./datasets/{dataset}/split_transductive/test'
    for file in os.listdir(path):
        print(file)
        if not file.startswith('pubdataedges') and file.endswith('.csv'):
            shutil.copy(path+'/'+file, pathv+'/'+file)
            shutil.copy(path+'/'+file, patht+'/'+file)


def check_subgraph_trans():
    dataset = args.dataset
    G_train = create_graph_csv('split_transductive/train',dataset, ['all'])
    G_validation = create_graph_csv('split_transductive/validation',dataset, ['all'])
    G_test = create_graph_csv('split_transductive/test',dataset, ['all'])
    print(len(list(nx.connected_components(G_train))))
    print(len(max(list(nx.connected_components(G_train)),key=len)))
    print(len(list(nx.connected_components(G_validation))))
    print(len(max(list(nx.connected_components(G_validation)),key=len)))
    print(len(list(nx.connected_components(G_test))))
    print(len(max(list(nx.connected_components(G_test)),key=len)))

    training_nodes, validation_nodes, test_nodes = set(list(G_train.nodes())), set(list(G_validation.nodes())), set(
        list(G_test.nodes()))
    print(len(training_nodes),len(validation_nodes),len(test_nodes))
    val_test = test_nodes.union(validation_nodes)
    is_subset = val_test.issubset(training_nodes)
    diff = training_nodes.difference(validation_nodes)
    if is_subset:
        print('ECCOMI')
    else:
        diff = validation_nodes.difference(training_nodes)
        print(diff)
        diff = test_nodes.difference(training_nodes)
        print(diff)

    # for ty in ['train']:
    #     pd_edges = pd.read_csv(f'./datasets/{dataset}/split_transductive/{ty}/pubdataedges.csv')
    #     pd_edges = [tuple([row['source'],row['target']]) for i,row in pd_edges.iterrows()]
    #     pd_edges_0 = pd.read_csv(f'./datasets/{dataset}/split_transductive_old/{ty}/pubdataedges.csv')
    #     pd_edges_0 = [tuple([row['source'],row['target']]) for i,row in pd_edges_0.iterrows()]
    #     pd_edges_ads = pd.read_csv(f'./datasets/{dataset}/split_transductive_old/{ty}/pubdataedges_ads.csv')
    #     pd_edges_ads = [tuple([row['source'],row['target']]) for i,row in pd_edges_ads.iterrows()]
    #     # pd_edges_kcore = pd.read_csv(f'./datasets/{dataset}/split_transductive_old/{ty}/pubdataedges_kcore.csv')
    #     # pd_edges_kcore = [tuple([row['source'],row['target']]) for i,row in pd_edges_kcore.iterrows()]
    #     for x in pd_edges_ads:
    #         if x not in pd_edges_0:
    #             print(x)

        # print('\n\n')
        # for x in pd_edges_kcore:
        #     if x not in pd_edges:
        #         print(x)

        # print(pd_edges_kcore.isin(pd_edges).all().all())
        # print(pd_edges_ads.isin(pd_edges).all().all())
        # print(pd_edges_0.isin(pd_edges).all().all())


def split(val_split,test_split):

    """Create transductive split: partition p->d edges keeping as many nodes as possible --> VALIDATION AND TEST are a subset of training! """

    def create_corresponding_graph_split(G,edges_to_remove):

        G_sub = G.copy()
        G_sub.remove_edges_from(edges_to_remove)
        components = list(nx.connected_components(G_sub))
        component = max(components,key=len)

        return G.subgraph(list(component))


    dataset = args.dataset
    G_all =  create_graph_csv('all/final',dataset, ['all'])
    components = list(nx.connected_components(G_all))
    component = max(components, key=len)
    G = G_all.subgraph(list(component))

    G_train = G.copy()
    G_validation = G.copy()
    G_test = G.copy()

    edges_pd = [(u, v) for u, v in G.edges() if
                     (u.startswith('p_') and v.startswith('d_')) or (
                             u.startswith('d_') and v.startswith('p_'))]

    max_iterations = 50
    found_best = False
    nodes_lost = float('inf')

    for i in range(max_iterations):
        if found_best:
            break
        print(f'iteration: {i}')
        train_data, test_val_data = train_test_split(edges_pd, test_size=test_split + val_split)
        validation_data, test_data = train_test_split(test_val_data, test_size=0.5)

        G_train = create_corresponding_graph_split(G_train,validation_data+test_data)
        G_validation = create_corresponding_graph_split(G_validation,train_data+test_data)
        G_test = create_corresponding_graph_split(G_test,validation_data+train_data)

        training_nodes, validation_nodes, test_nodes = set(list(G_train.nodes())),set(list(G_validation.nodes())),set(list(G_test.nodes()))
        val_test = test_nodes.union(validation_nodes)
        is_subset = validation_nodes.issubset(training_nodes)
        is_subset_0 = test_nodes.issubset(training_nodes)
        if is_subset and is_subset_0:
            print('is subset',len(list(training_nodes)),len(list(test_nodes)),len(list(validation_nodes)))
            cur_train = list(training_nodes)
            print(len(cur_train))
            cur_val = list(training_nodes.intersection(validation_nodes))
            print(len(cur_val))

            cur_test = list(training_nodes.intersection(test_nodes))
            print(len(cur_test))
            create_split_partitions(dataset, list(cur_train), list(cur_val), list(cur_test), train_data,validation_data,test_data)
            found_best = True
        else:
            nodes_val_test_lost = training_nodes.difference(val_test)
            pub_lost = [p for p in list(nodes_val_test_lost) if p.startswith("p")]
            data_lost = [p for p in list(nodes_val_test_lost) if p.startswith("d_")]
            print(f'current count of nodes lost: {len(list(nodes_val_test_lost))}')
            print(f'current count of publications lost: {len(pub_lost)}')
            print(f'current count of datasets lost: {len(data_lost)}')
            nodes_val_test_lost = len(list(pub_lost+data_lost))

            # if nodes_val_test_lost < 10:
            #     create_split_partitions(dataset, list(training_nodes), list(validation_nodes), list(test_nodes))
            #     found_best = True

            if nodes_lost >= nodes_val_test_lost :
                nodes_lost = nodes_val_test_lost
                cur_train = list(training_nodes)
                cur_val = list(training_nodes.intersection(validation_nodes))
                cur_test = list(training_nodes.intersection(test_nodes))

                create_split_partitions(dataset,cur_train,cur_val,cur_test, train_data,validation_data,test_data)
                # found_best = True


def create_split_partitions(dataset,training,validation,test, train_data,validation_data,test_data):

    """This script create the partitions corresponding to the selected communities"""
    """full transductive! All nodes in vali and test seen in training"""

    path = f'./datasets/{dataset}/all/final/'
    path_train = f'./datasets/{dataset}/split_transductive/train'
    path_test = f'./datasets/{dataset}/split_transductive/test'
    path_validation = f'./datasets/{dataset}/split_transductive/validation'

    train = training
    test = test
    for file in os.listdir(path):
        # print(file)
        if 'csv' in file:
            if 'edges' in file:
                csv = pd.read_csv(path+'/'+file)
                csv_train = csv[csv['source'].isin(train) & csv['target'].isin(train)]
                csv_train.to_csv(path_train+'/'+file,index=False)
                csv_test = csv[csv['source'].isin(test) & csv['target'].isin(test)]
                csv_test.to_csv(path_test+'/'+file,index=False)
                csv_vali = csv[csv['source'].isin(validation) & csv['target'].isin(validation)]
                csv_vali.to_csv(path_validation+'/'+file,index=False)
            else:
                csv = pd.read_csv(path+'/'+file)
                if 'id' in csv:
                    # csv_train = csv[csv['id'].isin(train)]
                    csv.to_csv(path_train+'/'+file,index=False)
                    # csv_test = csv[csv['id'].isin(test)]
                    csv.to_csv(path_test+'/'+file,index=False)
                    # csv_vali = csv[csv['id'].isin(validation)]
                    csv.to_csv(path_validation+'/'+file,index=False)
                    # csv_train = csv[csv['id'].isin(train)]
                    # csv_train.to_csv(path_train+'/'+file,index=False)
                    # csv_test = csv[csv['id'].isin(test)]
                    # csv_test.to_csv(path_test+'/'+file,index=False)
                    # csv_vali = csv[csv['id'].isin(validation)]
                    # csv_vali.to_csv(path_validation+'/'+file,index=False)


    train_data = pd.DataFrame(train_data, columns=['source', 'target'])
    train_data = train_data[train_data['source'].isin(train) & train_data['target'].isin(train)]
    train_data.to_csv(path_train+'/pubdataedges.csv',index=False)
    validation_data = pd.DataFrame(validation_data, columns=['source', 'target'])
    validation_data = validation_data[validation_data['source'].isin(validation) & validation_data['target'].isin(validation)]
    validation_data.to_csv(path_validation+'/pubdataedges.csv',index=False)
    test_data = pd.DataFrame(test_data, columns=['source', 'target'])
    test_data = test_data[test_data['source'].isin(test) & test_data['target'].isin(test)]
    test_data.to_csv(path_test+'/pubdataedges.csv',index=False)
    G_train = create_graph_csv('split_transductive/train',dataset, ['all'])
    diff = set(list(G_train.nodes())).difference(set(train))
    print(len(list(G_train.nodes())))
    print(diff)

def inductive_split():

    """In this split publications and datasets of val.and test. are not in training, however, their neighbourhood is.

    Mantengo lo stesso grafo ed elimino pub e dataset del val e del test dal training

    inductive_full = pub and data removed
    inductive_light = pub removed
    train set contains the entire graph, while vali and test the nodes to add
    """
    # check that test and val are subsets of training
    print()
    dataset = args.dataset
    # tolgo da training i nodi visti in validation e test.
    pubdata_test = pd.read_csv(f'./datasets/{args.dataset}/split_transductive/test/pubdataedges.csv')
    pubdata_vali = pd.read_csv(f'./datasets/{args.dataset}/split_transductive/validation/pubdataedges.csv')
    pubdata_train = pd.read_csv(f'./datasets/{args.dataset}/split_transductive/train/pubdataedges.csv')
    publications_train = [ p for p in pubdata_train['source'].unique().tolist() if p not in pubdata_vali['source'].unique().tolist() and p not in pubdata_test['source'].unique().tolist()]
    publications_validation = publications_train + [ p for p in pubdata_vali['source'].unique().tolist() if p not in pubdata_test['source'].unique().tolist()]
    publications_test = publications_train + publications_validation + pubdata_test['source'].unique().tolist()
    print(len([p for p in publications_validation if p not in publications_test and p]))
    print(len([p for p in publications_test if p not in publications_validation and p not in publications_train]))
    datasets_train = [ p for p in pubdata_train['target'].unique().tolist() if p not in pubdata_vali['target'].unique().tolist() and p not in pubdata_test['target'].unique().tolist()]
    datasets_validation = datasets_train + [ p for p in pubdata_vali['target'].unique().tolist() if p not in pubdata_test['target'].unique().tolist()]
    datasets_test = datasets_train + datasets_validation + pubdata_test['target'].unique().tolist()
    print(len([p for p in datasets_validation if p not in datasets_test and p]))
    print(len([p for p in datasets_test if p not in datasets_validation and p not in datasets_train]))
    data_test_unique = [p for p in datasets_test if p not in datasets_validation and p not in datasets_train]

    def switch_values(row):
        if row['source'].startswith('d_'):
            return pd.Series({'source': row['target'], 'target': row['source']})
        else:
            return row

    pubdata_test = pubdata_test.apply(switch_values, axis=1)
    pubdata_vali = pubdata_vali.apply(switch_values, axis=1)
    pubdata_train = pubdata_train.apply(switch_values, axis=1)


    print(f'publications starting: {len(pubdata_train["source"].unique().tolist())}, {len(pubdata_test["source"].unique().tolist())},{len(pubdata_vali["source"].unique().tolist())}')
    print(f'datasets starting: {len(pubdata_train["target"].unique().tolist())}, {len(pubdata_test["target"].unique().tolist())},{len(pubdata_vali["target"].unique().tolist())}')
    pubdata_train_tups = [(row['source'],row['target']) for i,row in pubdata_train.iterrows()]
    pubdata_test_tups = [(row['source'],row['target']) for i,row in pubdata_test.iterrows()]
    pubdata_vali_tups = [(row['source'],row['target']) for i,row in pubdata_vali.iterrows()]

    train_forbidden_pubs = list(set(pubdata_test['source'].unique().tolist() + pubdata_vali['source'].unique().tolist()))
    train_forbidden_data = list(set(pubdata_test['target'].unique().tolist() + pubdata_vali['target'].unique().tolist()))
    print(f'publications to remove from train: {len(train_forbidden_pubs)}')
    print(f'datasets to remove from train: {len(train_forbidden_data)}')
    vali_forbidden_pubs = list(
        set(pubdata_test['source'].unique().tolist()))
    vali_forbidden_data = list(
        set(pubdata_test['target'].unique().tolist()))
    print(f'publications to remove from validation: {len(train_forbidden_pubs)}')
    print(f'datasets to remove from validation: {len(train_forbidden_data)}')

    path = f'./datasets/{dataset}/all/final/'
    path_train_light = f'./datasets/{dataset}/split_inductive_light/train'
    path_train_full = f'./datasets/{dataset}/split_inductive_full/train'
    G = create_graph_csv('split_transductive/train', dataset, ['all'])
    G_light = G.copy()
    G_full = G.copy()
    # print(len([a for a in G_light.nodes() if a.startswith('p_')]))
    # print(len([a for a in G_light.nodes() if a.startswith('d_')]))

    G_light.remove_nodes_from(train_forbidden_pubs)
    # for n in train_forbidden_pubs:
    #     G_light.remove_node(n)
    train_nodes_light = max(list(nx.connected_components(G_light)),key=len)

    print('TRAIN LIGHT')
    pubs_light = [a for a in list(train_nodes_light) if a.startswith("p_")]
    data_light = [a for a in list(train_nodes_light) if a.startswith("d_")]
    print(f'publications {len(pubs_light)}')
    print(f'datasets {len(data_light)}')
    # print(len([a for a in G_light.nodes() if a.startswith('p_')]))
    # print(len([a for a in G_light.nodes() if a.startswith('d_')]))
    # print('\n\n')
    # print(len(list(nx.connected_components(G_light))))
    print(len(max(list(nx.connected_components(G_light)),key=len)))
    # print(len(list(G_light.nodes())))


    # print(len([a for a in G_full.nodes() if a.startswith('p_')]))
    # print(len([a for a in G_full.nodes() if a.startswith('d_')]))
    G_full.remove_nodes_from(train_forbidden_pubs)
    G_full.remove_nodes_from(train_forbidden_data)

    # print('\n\n')
    # print(len(list(nx.connected_components(G_full))))
    print(len(max(list(nx.connected_components(G_full)),key=len)))
    train_nodes_full = max(list(nx.connected_components(G_full)),key=len)
    print('TRAIN FULL')
    pubs_full = [a for a in list(train_nodes_full) if a.startswith("p_")]
    data_full = [a for a in list(train_nodes_full) if a.startswith("d_")]
    print(any(f in data_test_unique for f in data_full))
    print(f'publications {len(pubs_full)}')
    print(f'datasets {len(data_full)}')

    for file in os.listdir(path):
        if 'csv' in file:
            if 'edges' in file:
                csv = pd.read_csv(path+'/'+file)
                csv_train_light = csv[csv['source'].isin(train_nodes_light) & csv['target'].isin(train_nodes_light)]
                csv_train_full = csv[csv['source'].isin(train_nodes_full) & csv['target'].isin(train_nodes_full)]
                csv_train_light.to_csv(path_train_light+'/'+file,index=False)
                csv_train_full.to_csv(path_train_full+'/'+file,index=False)
            else:
                csv = pd.read_csv(path+'/'+file)
                if 'id' in csv:
                    csv_train_light = csv[csv['id'].isin(train_nodes_light)]
                    csv_train_full = csv[csv['id'].isin(train_nodes_full)]
                    csv_train_light.to_csv(path_train_light + '/' + file, index=False)
                    csv_train_full.to_csv(path_train_full + '/' + file, index=False)

    G = create_graph_csv('split_transductive/train', dataset, ['all'])
    # aggiungo gli edges tra p e d nel validation
    G.add_edges_from(pubdata_vali_tups)

    G_light_vali = G.copy()
    G_light_vali.remove_nodes_from(list(set(pubdata_test['source'].unique().tolist())))
    # print('\n\n')
    # print(len(list(nx.connected_components(G_light_vali))))
    print(len(max(list(nx.connected_components(G_light_vali)),key=len)))
    G_full_vali = G.copy()
    G_full_vali.remove_nodes_from(list(set(pubdata_test['source'].unique().tolist())))
    G_full_vali.remove_nodes_from(list(set(pubdata_test['target'].unique().tolist())))
    vali_nodes_light =  max(list(nx.connected_components(G_light_vali)),key=len)

    print('VALIDATION LIGHT')
    pubs_light_vali = [a for a in list(vali_nodes_light) if a.startswith("p_") and a not in pubs_light]
    data_light_vali = [a for a in list(vali_nodes_light) if a.startswith("d_") and a not in data_light]
    print(f'publications {len(pubs_light_vali)}')
    print(f'datasets {len(data_light_vali)}')

    vali_nodes_full =  max(list(nx.connected_components(G_full_vali)),key=len)

    print(len(max(list(nx.connected_components(G_full_vali)),key=len)))
    print('VALIDATION FULL')
    pubs_full_vali = [a for a in list(vali_nodes_full) if a.startswith("p_") and a not in pubs_full]
    data_full_vali = [a for a in list(vali_nodes_full) if a.startswith("d_") and a not in data_full]
    print(any(f in data_test_unique for f in data_full_vali))

    print(f'publications {len(pubs_full_vali)}')
    print(f'datasets {len(data_full_vali)}')

    path_vali_light = f'./datasets/{dataset}/split_inductive_light/validation'
    path_vali_full = f'./datasets/{dataset}/split_inductive_full/validation'
    for file in os.listdir(path):
        if 'csv' in file:
            if 'edges' in file:
                csv = pd.read_csv(path+'/'+file)
                csv_vali_light = csv[csv['source'].isin(vali_nodes_light) & csv['target'].isin(vali_nodes_light)]
                csv_vali_full = csv[csv['source'].isin(vali_nodes_full) & csv['target'].isin(vali_nodes_full)]
                csv_vali_light.to_csv(path_vali_light+'/'+file,index=False)
                csv_vali_full.to_csv(path_vali_full+'/'+file,index=False)
            else:
                csv = pd.read_csv(path+'/'+file)
                if 'id' in csv:
                    csv_vali_light = csv[csv['id'].isin(vali_nodes_light)]
                    csv_vali_full = csv[csv['id'].isin(vali_nodes_full)]
                    csv_vali_light.to_csv(path_vali_light + '/' + file, index=False)
                    csv_vali_full.to_csv(path_vali_full + '/' + file, index=False)

    #
    G = create_graph_csv('split_transductive/train', dataset, ['all'])
    G.add_edges_from(pubdata_test_tups)
    G.add_edges_from(pubdata_vali_tups)
    G_light_test = G.copy()
    G_full_test = G.copy()
    test_nodes_light = max(list(nx.connected_components(G_light_test)),key=len)
    test_nodes_full = max(list(nx.connected_components(G_full_test)),key=len)
    # print('\n\n')
    # print(len(list(nx.connected_components(G_light_test))))
    print(len(max(list(nx.connected_components(G_light_test)),key=len)))
    print(len(max(list(nx.connected_components(G_full_test)),key=len)))
    print('TEST FULL')
    pubs_full_TEST = [a for a in list(test_nodes_light) if a.startswith("p_") and a not in pubs_light and a not in pubs_light_vali]
    data_full_TEST = [a for a in list(test_nodes_light) if a.startswith("d_") and a not in data_light and a not in data_light_vali]
    print(f'publications {len(pubs_full_TEST)}')
    print(f'datasets {len(data_full_TEST)}')
    print('TEST FULL')
    pubs_full_TEST = [a for a in list(test_nodes_full) if a.startswith("p_") and a not in pubs_full and a not in pubs_full_vali]
    data_full_TEST = [a for a in list(test_nodes_full) if a.startswith("d_") and a not in data_full and a not in data_full_vali]
    print(f'publications {len(pubs_full_TEST)}')
    print(f'datasets {len(data_full_TEST)}')


    path_test_light = f'./datasets/{dataset}/split_inductive_light/test'
    path_test_full = f'./datasets/{dataset}/split_inductive_full/test'
    for file in os.listdir(path):
        if 'csv' in file:
            if 'edges' in file:
                csv = pd.read_csv(path+'/'+file)
                csv_test_light = csv[csv['source'].isin(test_nodes_light) & csv['target'].isin(test_nodes_light)]
                csv_test_full = csv[csv['source'].isin(test_nodes_full) & csv['target'].isin(test_nodes_full)]
                csv_test_light.to_csv(path_test_light+'/'+file,index=False)
                csv_test_full.to_csv(path_test_full+'/'+file,index=False)
            else:
                csv = pd.read_csv(path+'/'+file)
                if 'id' in csv:
                    csv_test_light = csv[csv['id'].isin(test_nodes_light)]
                    csv_test_full = csv[csv['id'].isin(test_nodes_full)]
                    csv_test_light.to_csv(path_test_light + '/' + file, index=False)
                    csv_test_full.to_csv(path_test_full + '/' + file, index=False)



def split_same_pool():

    """In this kind of split, publications and datasets are seen in training,test and validation"""

    path = f'./datasets/{args.dataset}/all/final/pubdataedges.csv' # kcore è pub e data con entrambi quindi kcore 2. ads è con dataset con almeno 2 pub connesse
    path_train = f'./datasets/{args.dataset}/split_transductive/train/pubdataedges_ads.csv'
    path_test = f'./datasets/{args.dataset}/split_transductive/test/pubdataedges_ads.csv'
    path_validation = f'./datasets/{args.dataset}/split_transductive/validation/pubdataedges_ads.csv'
    df = pd.read_csv(path)
    df['count'] = df.groupby('target')['target'].transform('count')
    result = df[(df['count'] > 1)]
    training = df[df['count'] == 1]

    training_rows,validation_rows,test_rows = [],[],[]

    for target, group_df in result.groupby('target'):
        i = 0
        for index, row in group_df.iterrows():
            # print(i)

            if i == 0 or i % 3 == 0:
                training_rows.append(tuple([row['source'], row['target']]))
            elif len(validation_rows) < len(test_rows):
                validation_rows.append(tuple([row['source'], row['target']]))
            elif len(test_rows) <= len(validation_rows):
                test_rows.append(tuple([row['source'], row['target']]))
            i += 1

        # print()  # Separate groups with an empty line
    print(len(training_rows))
    print(len(validation_rows))
    print(len(test_rows))
    random.shuffle(test_rows)
    random.shuffle(validation_rows)

    max_sample = int(0.1*df.shape[0]) if int(0.1*df.shape[0]) < len(validation_rows) else len(validation_rows)
    validation_rows_selected = random.sample(validation_rows, max_sample)
    max_sample = int(0.1*df.shape[0]) if int(0.1*df.shape[0]) < len(test_rows) else len(test_rows)

    test_rows_selected = random.sample(test_rows,max_sample)
    for t in test_rows+validation_rows:
        if t not in test_rows_selected and t not in validation_rows_selected:
            training_rows.append(t)
        # else:
        #     print('found')



    df_train = pd.DataFrame(training_rows, columns=['source', 'target'])
    df_validation = pd.DataFrame(validation_rows_selected, columns=['source', 'target'])
    df_test = pd.DataFrame(test_rows_selected, columns=['source', 'target'])

    df_train = pd.concat([df_train,training],ignore_index=True)
    df_test.to_csv(path_test,index=False)
    df_validation.to_csv(path_validation,index=False)
    df_train.to_csv(path_train,index=False)

    print(f'all edges: {df.shape[0]}')
    print(f'df_validation edges: {df_validation.shape[0]}')
    print(f'df_train edges: {df_train.shape[0]}')
    print(f'df_test edges: {df_test.shape[0]}')
    print(f'total: { df_validation.shape[0] + df_test.shape[0] + df_train.shape[0]}')


def split_kcore_transductive():
    def split_dataset(dataset):
        # Mescoliamo il dataset per garantire la casualità
        random.shuffle(dataset)

        train_set = []
        test_set = []

        # Aggiungiamo la prima tupla al training set
        train_set.append(dataset[0])

        for t in dataset[1:]:
            if all(v in [item for sublist in train_set for item in sublist] for v in t):
                if len(test_set) < len(dataset) // 2 and len(test_set) <= 0.2*len(dataset):
                    test_set.append(t)
                else:
                    train_set.append(t)
            else:
                train_set.append(t)

        return train_set, test_set[0:int(len(test_set)/2)],test_set[int(len(test_set)/2):]
    
    path = f'./datasets/{args.dataset}/all/final/pubdataedges.csv'  # kcore è pub e data con entrambi quindi kcore 2. ads è con dataset con almeno 2 pub connesse
    path_train = f'./datasets/{args.dataset}/split_transductive/train/pubdataedges_train_kcore_1.csv'
    path_test = f'./datasets/{args.dataset}/split_transductive/train/pubdataedges_test_kcore_1.csv'
    path_validation = f'./datasets/{args.dataset}/split_transductive/train/pubdataedges_validation_kcore_1.csv'
    df = pd.read_csv(path)
    tuples = [tuple([r['source'],r['target']]) for i,r in df.iterrows()]
    training_rows,validation_rows,test_rows = split_dataset(tuples)
    print(len(training_rows))
    print(len(validation_rows))
    print(len(test_rows))
    df_train = pd.DataFrame(training_rows, columns=['source', 'target'])
    df_validation = pd.DataFrame(validation_rows, columns=['source', 'target'])
    df_test = pd.DataFrame(test_rows, columns=['source', 'target'])
    #
    df_test.to_csv(path_test, index=False)
    df_validation.to_csv(path_validation, index=False)
    df_train.to_csv(path_train, index=False)
    #
    print(f'all edges: {df.shape[0]}')
    print(f'df_validation edges: {df_validation.shape[0]}')
    print(f'df_train edges: {df_train.shape[0]}')
    print(f'df_test edges: {df_test.shape[0]}')
    print(f'total: {df_validation.shape[0] + df_test.shape[0] + df_train.shape[0]}')

from collections import Counter

def split_kcore_inductive():
    # split_kcore_transductive()
    path_train = f'./datasets/{args.dataset}/split_transductive/train/pubdataedges_train_kcore_1.csv'
    df = pd.read_csv(path_train)
    tuples_train = [tuple([r['source'], r['target']]) for i, r in df.iterrows()]
    path_test = f'./datasets/{args.dataset}/split_transductive/train/pubdataedges_test_kcore_1.csv'
    df = pd.read_csv(path_test)
    tuples_test = [tuple([r['source'], r['target']]) for i, r in df.iterrows()]
    path_validation = f'./datasets/{args.dataset}/split_transductive/train/pubdataedges_validation_kcore_1.csv'
    df = pd.read_csv(path_validation)
    tuples_vali = [tuple([r['source'], r['target']]) for i, r in df.iterrows()]
    tuples_to_exclude = tuples_vali + tuples_test + tuples_train
    print(len(tuples_vali),len(tuples_test))

    def split_dataset(dataset,train,test,vali):
        random.shuffle(dataset)

        train_set = []
        test_set_semi = []
        test_set_ind = []
        test_set_trans = test + vali

        # per il semi induttivo seleziono una serie di coppie dal dataset transduttivo che ho già in cui i dataset hanno almeno due archi
        test_set_semi = random.sample(dataset,int(0.05*len(dataset)))

        train_set = [t for t in train if t[0] not in [v[0] for v in test_set_semi]]
        test = [t for t in test if t[0] not in [v[0] for v in test_set_semi]]
        vali = [t for t in vali if t[0] not in [v[0] for v in test_set_semi]]

        # per l'induttivo seleziono una serie di coppie pub-data dal training set e le rimuovo, includendo anche coppie con più di un dataset rilevante
        test_set_ind = random.sample(dataset,int(0.05*len(dataset)))
        train_set = [t for t in train_set if t[0] not in [v[0] for v in test_set_ind] and t[1] not in [v[1] for v in test_set_ind]]
        test = [t for t in test if t[0] not in [v[0] for v in test_set_ind] and t[1] not in [v[1] for v in test_set_ind]]
        vali = [t for t in vali if t[0] not in [v[0] for v in test_set_ind] and t[1] not in [v[1] for v in test_set_ind]]
        test_set_semi = [t for t in test_set_semi if t[0] not in [v[0] for v in test_set_ind] and t[1] not in [v[1] for v in test_set_ind]]
        print('ind')
        print(len(test_set_ind))
        print(len(train_set))
        print('semi')
        print(len(test_set_semi))


        test_set_trans = test + vali
        # controllo che tutto sia ok
        # check semi
        pub_removed = [t[0] for t in test_set_semi]
        tup_all = train_set + test_set_ind + test_set_trans
        print(len([t for t in tup_all if t[0] in pub_removed]))

        # check inductive
        pub_removed = [t[0] for t in test_set_ind]
        data_removed = [t[1] for t in test_set_ind]
        tup_all = train_set + test_set_semi + test_set_trans
        print(len([t for t in tup_all if t[0] in pub_removed or t[1] in data_removed]))

        validation = random.sample(train_set,int(0.05*len(dataset)))
        train_set = [t for t in train_set if t not in validation]
        print(len(train_set))
        test_set_trans = random.sample(train_set,int(0.05*len(dataset)))
        train_set = [t for t in train_set if t not in test_set_trans]
        print(len(train_set))
        print([set(train_set).intersection(set(test_set_trans))])
        print([set(train_set).intersection(set(test_set_semi))])
        print([set(train_set).intersection(set(test_set_ind))])
        print([set(validation).intersection(set(test_set_trans))])
        print([set(validation).intersection(set(test_set_semi))])
        print([set(validation).intersection(set(test_set_ind))])
        print([set(validation).intersection(set(train_set))])
        print('\n\n')
        print(len(validation))
        print(len(train_set))
        print('trans')
        print(len(test_set_trans))
        print(len(validation))
        return train_set, test_set_trans, test_set_semi, test_set_ind, validation


    path = f'./datasets/{args.dataset}/all/final/pubdataedges.csv'  # kcore è pub e data con entrambi quindi kcore 2. ads è con dataset con almeno 2 pub connesse
    path_train_0 = f'./datasets/{args.dataset}/split_transductive/train/pubdataedges_train_kcore_2.csv'
    path_test_trans = f'./datasets/{args.dataset}/split_transductive/train/pubdataedges_test_trans_kcore_2.csv'
    path_vali_trans = f'./datasets/{args.dataset}/split_transductive/train/pubdataedges_validation_trans_kcore_2.csv'
    path_test_semi = f'./datasets/{args.dataset}/split_transductive/train/pubdataedges_test_semi_kcore_2.csv'
    #path_vali_semi = f'./datasets/{args.dataset}/split_transductive/train/pubdataedges_validation_semi_kcore_2.csv'
    path_test_ind = f'./datasets/{args.dataset}/split_transductive/train/pubdataedges_test_ind_kcore_2.csv'
    #path_vali_ind = f'./datasets/{args.dataset}/split_transductive/train/pubdataedges_validation_ind_kcore_2.csv'


    df = pd.read_csv(path)
    tuples = [tuple([r['source'], r['target']]) for i, r in df.iterrows() if tuple([r['source'], r['target']])]
    print(len(tuples))
    train_set, test_set_trans, test_set_semi, test_set_ind, validation = split_dataset(tuples_to_exclude,tuples_train,tuples_vali,tuples_test)

    df_train = pd.DataFrame(train_set, columns=['source', 'target'])
    df_vali_trans = pd.DataFrame(validation, columns=['source', 'target'])
    #df_vali_semi = pd.DataFrame(vali_set_semi, columns=['source', 'target'])
    #df_vali_ind = pd.DataFrame(vali_set_ind, columns=['source', 'target'])
    df_test_trans = pd.DataFrame(test_set_trans, columns=['source', 'target'])
    df_test_semi = pd.DataFrame(test_set_semi, columns=['source', 'target'])
    df_test_ind = pd.DataFrame(test_set_ind, columns=['source', 'target'])
    #
    df_train.to_csv(path_train_0, index=False)
    df_vali_trans.to_csv(path_vali_trans, index=False)
    df_test_trans.to_csv(path_test_trans, index=False)
    df_test_semi.to_csv(path_test_semi, index=False)
    df_test_ind.to_csv(path_test_ind, index=False)
    #



def split_same_pool_kcore():

    """In this kind of split, datasets are seen in training,test and validation"""

    """Split having the same set of datasets in the three splits, this means that the datasets in validation and test have at least 2 publications connected"""

    # conto i dataset con almeno due pub
    path = f'./datasets/{args.dataset}/all/final/pubdataedges.csv' # kcore è pub e data con entrambi quindi kcore 2. ads è con dataset con almeno 2 pub connesse
    path_train_isolated = f'./datasets/{args.dataset}/split_transductive/train/pubdataedges_train_kcore_all_0.csv'
    path_train = f'./datasets/{args.dataset}/split_transductive/train/pubdataedges_train_kcore_0.csv'
    path_test = f'./datasets/{args.dataset}/split_transductive/train/pubdataedges_test_kcore_0.csv'
    path_validation = f'./datasets/{args.dataset}/split_transductive/train/pubdataedges_validation_kcore_0.csv'
    path_tmp = f'./datasets/{args.dataset}/split_transductive/train/tmp.csv'
    df = pd.read_csv(path)
    df['count'] = df.groupby('target')['target'].transform('count')
    df['count_source'] = df.groupby('source')['source'].transform('count')
    df.to_csv(path_tmp,index=False)
    result = df[(df['count'] > 1) & (df['count_source'] > 1)]
    print(result.shape)
    # result = df[(df['count'] > 1) ]
    training = df[df['count'] == 1]


    training_rows,validation_rows,test_rows = [],[],[]
    sources_trian,target_train = [],[]
    sources_vali,target_vali = [],[]
    for (source, target), group_df in result.groupby(['source','target']):
        if source in sources_trian or target in target_train:
            sources_vali.append(source)
            target_vali.append(source)
            validation_rows.append(tuple([source, target]))
        else:
            sources_trian.append(source)
            target_train.append(source)
            training_rows.append(tuple([source, target]))
    # print(len(training_rows))
    # print(len(validation_rows))
    # return 0
    random.shuffle(validation_rows)
    test_rows = validation_rows[:len(validation_rows)//2]
    validation_rows = validation_rows[len(validation_rows)//2:]

    # for (source, target), group_df in result.groupby(['source','target']):
    #     print(source,target)
    #     # print(f"Target: {target}")
    #     print(group_df)
    #     i = 1
    #     for index,row in group_df.iterrows():
    #         # print(i)
    #
    #         if i == 1 or i % 3 == 0:
    #             training_rows.append(tuple([row['source'],row['target']]))
    #         elif len(validation_rows) < len(test_rows):
    #             validation_rows.append(tuple([row['source'],row['target']]))
    #         elif len(test_rows) <= len(validation_rows):
    #             test_rows.append(tuple([row['source'],row['target']]))
    #         i += 1

        # print()  # Separate groups with an empty line
    print(len(training_rows))
    print(len(validation_rows))
    print(len(test_rows))
    random.shuffle(test_rows)
    random.shuffle(validation_rows)

    max_sample = int(0.1*df.shape[0]) if int(0.1*df.shape[0]) < len(validation_rows) else len(validation_rows)
    validation_rows = random.sample(validation_rows, max_sample)
    max_sample = int(0.1*df.shape[0]) if int(0.1*df.shape[0]) < len(test_rows) else len(test_rows)
    #
    test_rows = random.sample(test_rows,max_sample)
    # for t in test_rows+validation_rows:
    #     if t not in test_rows_selected and t not in validation_rows_selected:
    #         training_rows.append(t)
    #     # else:
    #     #     print('found')
    # #
    # #
    # #
    df_train = pd.DataFrame(training_rows, columns=['source', 'target'])
    df_validation = pd.DataFrame(validation_rows, columns=['source', 'target'])
    df_test = pd.DataFrame(test_rows, columns=['source', 'target'])
    #
    df_train_isolated = pd.concat([df_train,training],ignore_index=True)
    df_test.to_csv(path_test,index=False)
    df_validation.to_csv(path_validation,index=False)
    df_train.to_csv(path_train,index=False)
    df_train_isolated.to_csv(path_train_isolated,index=False)
    #
    print(f'all edges: {df.shape[0]}')
    print(f'df_validation edges: {df_validation.shape[0]}')
    print(f'df_train_isolated edges: {df_train_isolated.shape[0]}')
    print(f'df_train edges: {df_train.shape[0]}')
    print(f'df_test edges: {df_test.shape[0]}')
    print(f'total: { df_validation.shape[0] + df_test.shape[0] + df_train.shape[0]}')






def main():
    test_split = args.test_split
    val_split = args.val_split
    if val_split + test_split > 1.0:
        raise ValueError("The sum of test and validation split sizes cannot exceed 1.")
    split(val_split,test_split)
    # split_same_pool()

def inductive_split_new(full=False,type='train'):
    path = f'./datasets/{args.dataset}/split_transductive/train/' # kcore è pub e data con entrambi quindi kcore 2. ads è con dataset con almeno 2 pub connesse
    if type == 'test':
        path = f'./datasets/{args.dataset}/split_transductive/train/'

    path_test = f'./datasets/{args.dataset}/split_transductive/train/pubdataedges_test_kcore_1.csv'
    path_validation = f'./datasets/{args.dataset}/split_transductive/train/pubdataedges_validation_kcore_1.csv'
    pubdata_vali = pd.read_csv(path_validation)
    pubdata_test = pd.read_csv(path_test)
    # train

    pub_to_remove = []
    data_to_remove = []
    if type == 'train':
        pub_to_remove = list(set(pubdata_vali['source'].unique().tolist() + pubdata_test['source'].unique().tolist()))
        data_to_remove = list(set(pubdata_vali['target'].unique().tolist() + pubdata_test['target'].unique().tolist()))
    elif type == 'validation':
        pub_to_remove = list(set(pubdata_test['source'].unique().tolist()))
        data_to_remove = list(set(pubdata_test['target'].unique().tolist()))

    G = create_graph_csv('split_transductive/train', args.dataset, ['all'])
    G.remove_nodes_from(pub_to_remove)
    if full:
        G.remove_nodes_from(data_to_remove)

    path_to_save = f'./datasets/{dataset}/split_inductive_light/{type}'
    if full:
        path_to_save = f'./datasets/{dataset}/split_inductive_full/{type}'

    nodes_to_keep = [a for a in list(G.nodes()) if a not in pub_to_remove and a not in data_to_remove]
    for file in os.listdir(path):
        if 'csv' in file:
            if 'edges' in file:
                csv = pd.read_csv(path+'/'+file)
                csv_train_light = csv[csv['source'].isin(nodes_to_keep) & csv['target'].isin(nodes_to_keep)]
                csv_train_light.to_csv(path_to_save+'/'+file,index=False)
            else:
                csv = pd.read_csv(path+'/'+file)
                if 'id' in csv:
                    csv_train_light = csv[csv['id'].isin(nodes_to_keep)]
                    csv_train_light.to_csv(path_to_save + '/' + file, index=False)



if __name__ == '__main__':
    # main()
    dataset = args.dataset
    split_kcore_inductive()
    
    # path_train = f'./datasets/{dataset}/split_transductive/train/pubdataedges_train.csv'
    # source_train = pd.read_csv(path_train)['source'].unique().tolist()
    # target_train = pd.read_csv(path_train)['target'].unique().tolist()
    # path_train = f'./datasets/{dataset}/split_transductive/train/pubdataedges_validation.csv'
    # source_validation = pd.read_csv(path_train)['source'].unique().tolist()
    # target_validation = pd.read_csv(path_train)['target'].unique().tolist()
    # path_train = f'./datasets/{dataset}/split_transductive/train/pubdataedges_test.csv'
    # source_test = pd.read_csv(path_train)['source'].unique().tolist()
    # target_test = pd.read_csv(path_train)['target'].unique().tolist()
    # # copy_trans()
    # print(len(source_train))
    # print(len(source_test))
    # print(len(source_validation))
    # print(len(target_train))
    # print(len(target_test))
    # print(len(target_validation))
    # print(len(list(set(source_train).intersection(set(source_test)))))
    # print(len(list(set(source_train).intersection(set(source_validation)))))
    # print(len(list(set(target_train).intersection(set(target_test)))))
    # print(len(list(set(target_train).intersection(set(target_validation)))))
    # inductive_split()
    # inductive_split()
    # split(0.1,0.1)
    # split_kcore()
    # get_count1()

    # check_subgraph_trans()
    # copy_trans()
    # split_same_pool()
    # check_subgraph_trans()
    # for dataset in [args.dataset]:
    #     path_train_ads = f'./datasets/{dataset}/split_transductive/train/pubdataedges_ads.csv'
    #     path_train = f'./datasets/{dataset}/split_transductive/train/pubdataedges.csv'
    #     path_train_ind = f'./datasets/{dataset}/split_inductive/train/pubdataedges.csv'
    #     print(pd.read_csv(path_train_ads).shape[0],pd.read_csv(path_train).shape[0],pd.read_csv(path_train_ind).shape[0])
    #
    #     path_train_ads = f'./datasets/{dataset}/split_transductive/validation/pubdataedges_ads.csv'
    #     path_train = f'./datasets/{dataset}/split_transductive/validation/pubdataedges.csv'
    #     path_train_ind = f'./datasets/{dataset}/split_inductive/validation/pubdataedges.csv'
    #     print(pd.read_csv(path_train_ads).shape[0],pd.read_csv(path_train).shape[0],pd.read_csv(path_train_ind).shape[0])
    #
    #     path_train_ads = f'./datasets/{dataset}/split_transductive/test/pubdataedges_ads.csv'
    #     path_train = f'./datasets/{dataset}/split_transductive/test/pubdataedges.csv'
    #     path_train_ind = f'./datasets/{dataset}/split_inductive/test/pubdataedges.csv'
    #     print(pd.read_csv(path_train_ads).shape[0],pd.read_csv(path_train).shape[0],pd.read_csv(path_train_ind).shape[0])

