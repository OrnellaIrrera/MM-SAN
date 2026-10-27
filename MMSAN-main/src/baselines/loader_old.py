import os.path as osp
import pandas as pd
import torch_geometric.utils.convert
from sentence_transformers import SentenceTransformer
from torch_geometric import seed_everything
from torch_geometric.data import HeteroData
from torch_geometric.data import Data, InMemoryDataset
import pickle
import argparse
import json
import random
import torch
import numpy as np
import utils
seed_everything(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
seed_everything(42)
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True


def cosine_similarity(emb1, emb2):
    dot_product = np.dot(emb1, emb2)
    norm_vector1 = np.linalg.norm(emb1)
    norm_vector2 = np.linalg.norm(emb2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity


def load_node_csv(path, index_col, encoders=None, **kwargs):
    print(path)
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    print(path, df.shape[0])
    if 'content' in list(df.columns):
        df['content'] = df['content'].astype(str)
    if 'name' in list(df.columns):
        df['name'] = df['name'].astype(str)
    if 'fullname' in list(df.columns):
        df['fullname'] = df['fullname'].astype(str)
    if 'description' in list(df.columns):
        df['description'] = df['description'].astype(str)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)


    return x, mapping


def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,pubs_to_avoid=[],data_to_avoid=[],
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)
    print(path, df.shape[0])

    def switch_values(row):
        if row['source'].startswith('d_') and row['target'].startswith('p_'):
            return pd.Series({'source': row['target'], 'target': row['source']})
        else:
            return row

    df = df.apply(switch_values, axis=1)
    df = df.drop_duplicates(keep='first')


    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]

    src_0 = []
    dst_0 = []
    for s,t in zip(src,dst):
        if 'pubpub' in path:
            if s not in pubs_to_avoid and t not in pubs_to_avoid:
                src_0.append(s)
                dst_0.append(t)
        elif 'pubdata' in path:
            if s not in pubs_to_avoid and t not in data_to_avoid:
                src_0.append(s)
                dst_0.append(t)
        elif 'datadata' in path:
            if s not in data_to_avoid and t not in data_to_avoid:
                src_0.append(s)
                dst_0.append(t)
        elif 'pub' in path:
            if s not in pubs_to_avoid:
                src_0.append(s)
                dst_0.append(t)
        elif 'dat' in path:
            if s not in data_to_avoid:
                src_0.append(s)
                dst_0.append(t)
    edge_index = torch.tensor([src_0, dst_0])
    print(f'edge_index size {edge_index.shape}')

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


class ContentEncoder:
    # The 'SequenceEncoder' encodes raw column strings into embeddings.
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()


class KeywordEncoder:
    # The 'SequenceEncoder' encodes raw column strings into embeddings.
    def __init__(self, model_name='whaleloops/phrase-bert', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()


class ScholarlyDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        if 'transductive' in self.root:
            # return ['baseline_trans_1.pt']
            return ['graph_transductive_kcore_1.pt']

        elif 'inductive_full' in self.root:
            return ['graph_inductive_full_0.pt']

        elif 'inductive_light' in self.root:
            return ['graph_inductive_light_0.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        dataset = self.root.split('/')[1]
        print(f'DATASET: {dataset}')
        if 'inductive_light' in self.root:
            data_train = self.create_inductive_graph()
            data_list = [data_train]
        elif 'inductive_full' in self.root:
            data_train = self.create_inductive_graph(full=True)
            data_list = [data_train]
        elif 'transductive' in self.root:
            data_pre = self.create_transductive_graph()
            data_list = [data_pre]


        print(self.root)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        # print(self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])

    def create_transductive_graph(self):
        publication_path = self.root + '/publications.csv'
        dataset_path = self.root + '/datasets.csv'
        authors_path = self.root + '/authors.csv'

        publication_x, publication_mapping = load_node_csv(publication_path, index_col='id',
                                                                              encoders={'content': ContentEncoder()})
        dataset_x, dataset_mapping = load_node_csv(dataset_path, index_col='id',
                                                                  encoders={'content': ContentEncoder()})
        authors_x, author_mapping = load_node_csv(authors_path, index_col='id',
                                                                 encoders={'fullname': ContentEncoder()})

        if 'mes' not in self.root:
            keyword_path = self.root +  '/keywords.csv'
            orgs_path = self.root +  '/organizations.csv'
            venue_path = self.root  + '/venues.csv'
            orgs_x, orgs_mapping = load_node_csv(orgs_path, index_col='id',encoders={'name': ContentEncoder()})
            venue_x, venue_mapping = load_node_csv(venue_path, index_col='id',encoders={'name': ContentEncoder()})
            keyword_x, keyword_mapping = load_node_csv(keyword_path, index_col='id', encoders={'name': ContentEncoder()})


        pd_edges_path_train = self.root + '/pubdataedges_train_kcore_1.csv'
        pd_edges_path_vali = self.root + '/pubdataedges_validation_kcore_1.csv'
        pd_edges_path_test = self.root + '/pubdataedges_test_kcore_1.csv'
        pp_edges_path = self.root + '/pubpubedges.csv'
        dd_edges_path = self.root + '/datadataedges.csv'
        pa_edges_path = self.root + '/pubauthedges.csv'
        da_edges_path = self.root + '/dataauthedges.csv'
        if 'mes' not in self.root:
            pv_edges_path = self.root + '/pubvenuesedges.csv'
            po_edges_path = self.root + '/puborgedges.csv'
            do_edges_path = self.root + '/dataorgedges.csv'
            pk_edges_path = self.root + '/pubkeyedges.csv'
            dk_edges_path = self.root + '/datakeyedges.csv'

        pd_edge_index_train, pd_edge_label = load_edge_csv(
            pd_edges_path_train,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=dataset_mapping,
        )
        pd_edge_index_validation, pd_edge_label = load_edge_csv(
            pd_edges_path_vali,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=dataset_mapping,
        )
        pd_edge_index_test, pd_edge_label = load_edge_csv(
            pd_edges_path_test,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=dataset_mapping,
        )

        if 'mes' not in self.root:
            pv_edge_index, pv_edge_label = load_edge_csv(
                pv_edges_path,
                src_index_col='source',
                src_mapping=publication_mapping,
                dst_index_col='target',
                dst_mapping=venue_mapping,
            )
            po_edge_index, po_edge_label = load_edge_csv(
                po_edges_path,
                src_index_col='source',
                src_mapping=publication_mapping,
                dst_index_col='target',
                dst_mapping=orgs_mapping,
            )
            do_edge_index, do_edge_label = load_edge_csv(
                do_edges_path,
                src_index_col='source',
                src_mapping=dataset_mapping,
                dst_index_col='target',
                dst_mapping=orgs_mapping,
            )
            pk_edge_index, pk_edge_label = load_edge_csv(
                pk_edges_path,
                src_index_col='source',
                src_mapping=publication_mapping,
                dst_index_col='target',
                dst_mapping=keyword_mapping,
            )
            dk_edge_index, dk_edge_label = load_edge_csv(
                dk_edges_path,
                src_index_col='source',
                src_mapping=dataset_mapping,
                dst_index_col='target',
                dst_mapping=keyword_mapping,
            )

        # pd_edge_index_train_mp = pd_edge_index_train[:,:int(2*pd_edge_index_train.size(1)/3)]
        # pd_edge_index_label_train = pd_edge_index_train[:,int(2*pd_edge_index_train.size(1)/3):]

        # pd_edge_index_validation_mp = torch.cat([pd_edge_index_train,pd_edge_index_validation[:,:int(pd_edge_index_validation.size(1)/2)]],dim=1)
        # pd_edge_index_label_validation = pd_edge_index_validation[:,int(pd_edge_index_validation.size(1)/2):]
        # #
        # pd_edge_index_test_mp = torch.cat([pd_edge_index_train,pd_edge_index_validation,pd_edge_index_test[:,:int(pd_edge_index_test.size(1)/2)]],dim=1)
        # pd_edge_index_label_test = pd_edge_index_test[:,int(pd_edge_index_test.size(1)/2):]

        pp_edge_index, pp_edge_label = load_edge_csv(
            pp_edges_path,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=publication_mapping,
        )

        dd_edge_index, dd_edge_label = load_edge_csv(
            dd_edges_path,
            src_index_col='source',
            src_mapping=dataset_mapping,
            dst_index_col='target',
            dst_mapping=dataset_mapping,
        )

        pa_edge_index, pa_edge_label = load_edge_csv(
            pa_edges_path,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=author_mapping,
        )
        da_edge_index, da_edge_label = load_edge_csv(
            da_edges_path,
            src_index_col='source',
            src_mapping=dataset_mapping,
            dst_index_col='target',
            dst_mapping=author_mapping,
        )



        data = HeteroData()
        data['publication'].x = publication_x
        data['publication'].mapping = publication_mapping
        data['publication'].num_nodes = len(publication_mapping)  # Users do not have any features.

        data['dataset'].num_nodes = len(dataset_mapping)  # Users do not have any features.
        data['dataset'].x = dataset_x
        data['dataset'].mapping = dataset_mapping

        data['author'].num_nodes = len(author_mapping)  # Users do not have any features.
        data['author'].x = authors_x
        data['author'].mapping = author_mapping

        if 'mes' not in self.root:
            data['venue'].num_nodes = len(venue_mapping)  # Users do not have any features.
            data['venue'].x = venue_x
            data['venue'].mapping = venue_mapping

            data['keyword'].num_nodes = len(keyword_mapping)  # Users do not have any features.
            data['keyword'].x = keyword_x
            data['keyword'].mapping = keyword_mapping

            data['organization'].num_nodes = len(orgs_mapping)  # Users do not have any features.
            data['organization'].x = orgs_x
            data['organization'].mapping = orgs_mapping


        # leave part of edges for message passing
        data['publication', 'cites', 'dataset'].edge_index_train = pd_edge_index_train
        data['publication', 'cites', 'dataset'].edge_label_index_train = pd_edge_index_train
        cosine_label_ind = torch.cat([pd_edge_index_train,pd_edge_index_validation,pd_edge_index_test],dim=1)
        data['publication', 'cites', 'dataset'].negative_edge_label_index_train = cosine_based_negative_samples(
            cosine_label_ind, pd_edge_index_train, data['publication'].x, data['dataset'].x)

        data['publication', 'cites', 'dataset'].edge_index_validation = pd_edge_index_train
        data['publication', 'cites', 'dataset'].edge_label_index_validation = pd_edge_index_validation
        data['publication', 'cites', 'dataset'].negative_edge_label_index_validation = cosine_based_negative_samples(
            cosine_label_ind, pd_edge_index_validation, data['publication'].x, data['dataset'].x)

        data['publication', 'cites', 'dataset'].edge_index_test = torch.cat(
            [pd_edge_index_train, pd_edge_index_validation], dim=1)
        data['publication', 'cites', 'dataset'].edge_label_index_test = pd_edge_index_test
        data['publication', 'cites', 'dataset'].negative_edge_label_index_test = cosine_based_negative_samples(
            cosine_label_ind, pd_edge_index_test,
            data['publication'].x, data['dataset'].x)

        data['publication', 'cites', 'publication'].edge_index = pp_edge_index
        data['publication', 'cites', 'publication'].edge_label = pp_edge_label
        data['publication', 'hasauthor', 'author'].edge_index = pa_edge_index
        data['publication', 'hasauthor', 'author'].edge_label = pa_edge_label
        data['dataset', 'hasauthor', 'author'].edge_index = da_edge_index
        data['dataset', 'hasauthor', 'author'].edge_label = da_edge_label
        data['dataset', 'cites', 'dataset'].edge_index = dd_edge_index
        data['dataset', 'cites', 'dataset'].edge_label = dd_edge_label

        if 'mes' not in self.root:
            data['publication', 'hasvenue', 'venue'].edge_index = pv_edge_index
            data['publication', 'hasvenue', 'venue'].edge_label = pv_edge_label
            data['publication', 'haskeyword', 'keyword'].edge_index = pk_edge_index
            data['publication', 'haskeyword', 'keyword'].edge_label = pk_edge_label
            data['dataset', 'haskeyword', 'keyword'].edge_index = dk_edge_index
            data['dataset', 'haskeyword', 'keyword'].edge_label = dk_edge_label
            data['publication', 'hasorganization', 'organization'].edge_index = po_edge_index
            data['publication', 'hasorganization', 'organization'].edge_label = po_edge_label
            data['dataset', 'hasorganization', 'organization'].edge_index = do_edge_index
            data['dataset', 'hasorganization', 'organization'].edge_label = do_edge_label

        return data

    def create_inductive_graph(self,full=False):
        path = self.root.replace('_inductive_light','')
        path = path.replace('_inductive_full','')
        print(path)
        path = path.strip()
        publication_path = path + '/publications.csv'
        dataset_path = path + '/datasets.csv'
        authors_path = path + '/authors.csv'

        publication_x, publication_mapping = load_node_csv(publication_path, index_col='id',
                                                           encoders={'content': ContentEncoder()})
        dataset_x, dataset_mapping = load_node_csv(dataset_path, index_col='id',
                                                   encoders={'content': ContentEncoder()})
        authors_x, author_mapping = load_node_csv(authors_path, index_col='id',
                                                  encoders={'fullname': KeywordEncoder()})

        pd_edges_path_train = path + '/pubdataedges_train_kcore_1.csv'
        pd_edges_path_vali = path + '/pubdataedges_validation_kcore_1.csv'
        pd_edges_path_test = path + '/pubdataedges_test_kcore_1.csv'
        pp_edges_path = path + '/pubpubedges.csv'
        dd_edges_path = path + '/datadataedges.csv'
        pa_edges_path = path + '/pubauthedges.csv'
        da_edges_path = path + '/dataauthedges.csv'


        pd_edge_index_validation, pd_edge_label = load_edge_csv(
            pd_edges_path_vali,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=dataset_mapping,
        )

        pd_edge_index_test, pd_edge_label = load_edge_csv(
            pd_edges_path_test,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=dataset_mapping,
        )
        sources_to_exclude_from_training = list(set(pd_edge_index_validation[0].unique().tolist() + pd_edge_index_test[0].unique().tolist()))
        targets_to_exclude_from_training = list(set(pd_edge_index_validation[1].unique().tolist() + pd_edge_index_test[1].unique().tolist()))
        if full == False:
            targets_to_exclude_from_training = []
        print(sources_to_exclude_from_training)
        print(len(sources_to_exclude_from_training))
        print(targets_to_exclude_from_training)
        print(len(targets_to_exclude_from_training))
        pd_edge_index_train, pd_edge_label = load_edge_csv(
            pd_edges_path_train,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=dataset_mapping,

            pubs_to_avoid=sources_to_exclude_from_training,
            data_to_avoid=targets_to_exclude_from_training,
        )

        pp_edge_index_train, pp_edge_label_train = load_edge_csv(
            pp_edges_path,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=publication_mapping,
            pubs_to_avoid=sources_to_exclude_from_training,
            data_to_avoid=targets_to_exclude_from_training,
        )
        pp_edge_index_test, pp_edge_label_test = load_edge_csv(
            pp_edges_path,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=publication_mapping,
        )

        dd_edge_index_train, dd_edge_label_train = load_edge_csv(
            dd_edges_path,
            src_index_col='source',
            src_mapping=dataset_mapping,
            dst_index_col='target',
            dst_mapping=dataset_mapping,
            pubs_to_avoid=sources_to_exclude_from_training,
            data_to_avoid=targets_to_exclude_from_training,
        )
        dd_edge_index_test, dd_edge_label_test = load_edge_csv(
            dd_edges_path,
            src_index_col='source',
            src_mapping=dataset_mapping,
            dst_index_col='target',
            dst_mapping=dataset_mapping,
        )
        pa_edge_index_test, pa_edge_label_test = load_edge_csv(
            pa_edges_path,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=author_mapping,
        )
        pa_edge_index_train, pa_edge_label_train = load_edge_csv(
            pa_edges_path,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=author_mapping,
            pubs_to_avoid=sources_to_exclude_from_training,
            data_to_avoid=targets_to_exclude_from_training,
        )
        da_edge_index_train, da_edge_label_train = load_edge_csv(
            da_edges_path,
            src_index_col='source',
            src_mapping=dataset_mapping,
            dst_index_col='target',
            dst_mapping=author_mapping,
            pubs_to_avoid=sources_to_exclude_from_training,
            data_to_avoid=targets_to_exclude_from_training,
        )
        da_edge_index_test, da_edge_label_test = load_edge_csv(
            da_edges_path,
            src_index_col='source',
            src_mapping=dataset_mapping,
            dst_index_col='target',
            dst_mapping=author_mapping,
        )

        data = HeteroData()
        data['publication'].x = publication_x
        data['publication'].mapping = publication_mapping
        data['publication'].num_nodes = len(publication_mapping)  # Users do not have any features.

        data['dataset'].num_nodes = len(dataset_mapping)  # Users do not have any features.
        data['dataset'].x = dataset_x
        data['dataset'].mapping = dataset_mapping

        data['author'].num_nodes = len(author_mapping)  # Users do not have any features.
        data['author'].x = authors_x
        data['author'].mapping = author_mapping
        # leave part of edges for message passing
        data['publication', 'cites', 'dataset'].edge_index_train = pd_edge_index_train
        data['publication', 'cites', 'dataset'].edge_label_index_train = pd_edge_index_train
        cosine_label_ind = torch.cat([pd_edge_index_train, pd_edge_index_validation, pd_edge_index_test], dim=1)
        data['publication', 'cites', 'dataset'].negative_edge_label_index_train = cosine_based_negative_samples(
            cosine_label_ind, pd_edge_index_train, data['publication'].x, data['dataset'].x,pub_to_avoid=sources_to_exclude_from_training,data_to_avoid=targets_to_exclude_from_training)

        data['publication', 'cites', 'dataset'].edge_index_validation = pd_edge_index_train
        data['publication', 'cites', 'dataset'].edge_label_index_validation = pd_edge_index_validation
        data['publication', 'cites', 'dataset'].negative_edge_label_index_validation = cosine_based_negative_samples(
            cosine_label_ind, pd_edge_index_validation, data['publication'].x, data['dataset'].x)

        data['publication', 'cites', 'dataset'].edge_index_test = torch.cat(
            [pd_edge_index_train, pd_edge_index_validation], dim=1)
        data['publication', 'cites', 'dataset'].edge_label_index_test = pd_edge_index_test
        data['publication', 'cites', 'dataset'].negative_edge_label_index_test = cosine_based_negative_samples(
            cosine_label_ind, pd_edge_index_test,
            data['publication'].x, data['dataset'].x)

        data['publication', 'cites', 'publication'].edge_index_train = pp_edge_index_train
        data['publication', 'cites', 'publication'].edge_label_train = pp_edge_label_train
        data['publication', 'hasauthor', 'author'].edge_index_train = pa_edge_index_train
        data['publication', 'hasauthor', 'author'].edge_label_train = pa_edge_label_train
        data['dataset', 'hasauthor', 'author'].edge_index_train = da_edge_index
        data['dataset', 'hasauthor', 'author'].edge_label_train = da_edge_label
        data['dataset', 'cites', 'dataset'].edge_index_train = dd_edge_index_train
        data['dataset', 'cites', 'dataset'].edge_label_train = dd_edge_label_train
        data['publication', 'cites', 'publication'].edge_index_test = pp_edge_index_test
        data['publication', 'cites', 'publication'].edge_label_test = pp_edge_label_test
        data['publication', 'hasauthor', 'author'].edge_index_test = pa_edge_index_test
        data['publication', 'hasauthor', 'author'].edge_label_test = pa_edge_label_test
        data['dataset', 'hasauthor', 'author'].edge_index_test = da_edge_index_test
        data['dataset', 'hasauthor', 'author'].edge_label_test = da_edge_label_test
        data['dataset', 'cites', 'dataset'].edge_index_test = dd_edge_index_test
        data['dataset', 'cites', 'dataset'].edge_label_test = dd_edge_label_test
        return data


def cosine_based_negative_samples(edge_index, edge_label_index, source_vectors, target_vectors, similar=False,pub_to_avoid=[],data_to_avoid=[]):
    edges_all = torch.cat([edge_index, edge_label_index], dim=1)
    edges_t = edges_all.t().tolist()
    edges_t = list(set(tuple(e) for e in edges_t))

    # print(source_vectors[0])
    sources = list(set(edges_all[0].tolist()))
    print(len(sources))
    sources = [s for s in sources if s not in pub_to_avoid]
    print(len(sources))
    targets = list(set(edges_all[1].tolist()))
    print(len(targets))
    targets = [s for s in targets if s not in data_to_avoid]
    print(len(targets))

    negative_edges = []
    total_negative_samples = edge_label_index.size(1)
    # print(len(sources))
    # print(total_negative_samples)
    # if len(sources) < total_negative_samples:
    #     sources = sources + sources
    # random_sources = random.sample(sources,total_negative_samples)
    # print(f'total random negative edges {total_negative_samples}')
    while len(negative_edges) < total_negative_samples:
        s = random.sample(sources, 1)[0]
        random.shuffle(targets)
        targets_selected = random.sample(targets, 100)
        targets_cosine = []
        for j, t in enumerate(targets_selected):
            # print('working on target: ', t, j)
            if tuple([s, t]) not in edges_t:
                cosine = cosine_similarity(source_vectors[s], target_vectors[t])
                targets_cosine.append([t, cosine])
            # else:
            # print('FOUND',tuple([s, t]))
        targets_cosine = sorted(targets_cosine, key=lambda t: t[1], reverse=similar)
        # print(targets_cosine)
        negative_edges.append([s, targets_cosine[0][0]])
    negative_edges = torch.tensor(negative_edges).t()
    # print(negative_edges)
    # print(edges_t)
    return negative_edges


def random_negative_samples(edge_index, edge_label_index):
    edges_t = edge_index.t().tolist()
    edges_t = [tuple(t) for t in edges_t]

    sources = edge_index[0].tolist()
    targets = edge_index[1].tolist()
    new_edges_t = torch.tensor([[], []])
    found = False
    while not found:
        new_sources = random.sample(sources, edge_label_index.size(1))
        new_targets = random.sample(targets, edge_label_index.size(1))
        new_edges_t = torch.tensor([new_sources, new_targets]).t().tolist()
        new_edges_t = [tuple(t) for t in new_edges_t]
        if set(edges_t).isdisjoint(new_edges_t):
            found = True
    new_edges = torch.tensor([list(e) for e in new_edges_t])
    new_edges = new_edges.t()
    return new_edges




def get_results_to_rerank(dataset,train_data,deep_rerank=20,topk=10):
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

    results = json.load(open(f'baselines/trivial/data/{dataset}/results.json', 'r'))
    g = open(f'./baselines/trivial/data/{dataset}/groundtruth.json', 'r')
    ground_truth = json.load(g)
    precision, recall, ndcg = 0, 0, 0
    c = 0
    rerankers = {}
    for k, v in results.items():
        c += 1
        true = ground_truth[k]
        rerankers[train_data['publication'].mapping[k]] = [train_data['dataset'].mapping[x] for x in v][0:deep_rerank]
        pred = v[0:topk]
        precision += len(list(set(pred).intersection(true))) / topk
        recall += len(list(set(pred).intersection(true))) / len(true)
        ndcg += ndcg_at_k(true, pred, topk)

    print('PARTIAL RESULTS TRIVIAL')
    print(precision / c)
    print(recall / c)
    print(ndcg / c)

    return rerankers

def load_transductive_data(root,indices=False):
    print('SONO QUA!')
    def remove_nodes(train_data):
        # del train_data['publication', 'hasentity', 'entity']
        # del train_data['dataset', 'hasentity', 'entity']
        # del train_data['entity']
        # del train_data['publication', 'hastopic', 'topic']
        # del train_data['dataset', 'hastopic', 'topic']
        # del train_data['topic']

        if indices != [] and indices:
            train_data['dataset'].x[indices, :] = 1.0

        train_data['publication'].x = torch.cat([train_data['publication'].x, train_data['publication'].net_x], dim=1)
        train_data['dataset'].x = torch.cat([train_data['dataset'].x, train_data['dataset'].net_x], dim=1)
        # print(train_data['publication'].x.shape)
        # print(train_data['dataset'].x.shape)
        if 'pubmed' in root:
            # del train_data['publication', 'hasvenue', 'venue']
            del train_data['publication', 'hasauthor', 'author']
            del train_data['dataset', 'hasauthor', 'author']
            del train_data['author']
            del train_data['publication', 'hasorganization', 'organization']
            del train_data['dataset', 'hasorganization', 'organization']
            del train_data['organization']
            del train_data['publication', 'hasvenue', 'venue']
            del train_data['venue']
            del train_data['publication', 'haskeyword', 'keyword']
            del train_data['dataset', 'haskeyword', 'keyword']
            del train_data['keyword']

        return train_data

    data = ScholarlyDataset(root=root)
    train_data = data[0]
    train_data = remove_nodes(train_data)



    train_data['publication', 'cites', 'dataset'].edge_index = train_data[
        'publication', 'cites', 'dataset'].edge_index_train
    train_data['publication', 'cites', 'dataset'].edge_label_index = torch.cat(
        [train_data['publication', 'cites', 'dataset'].edge_label_index_train,
         train_data['publication', 'cites', 'dataset'].negative_edge_label_index_train], dim=1)

    data = ScholarlyDataset(root)
    validation_data = data[0]
    validation_data = remove_nodes(validation_data)

    validation_data['publication', 'cites', 'dataset'].edge_index = validation_data[
        'publication', 'cites', 'dataset'].edge_index_train
    validation_data['publication', 'cites', 'dataset'].edge_label_index = torch.cat(
        [validation_data['publication', 'cites', 'dataset'].edge_label_index_validation,
         validation_data['publication', 'cites', 'dataset'].negative_edge_label_index_validation], dim=1)

    data = ScholarlyDataset(root)
    test_data = data[0]
    test_data = remove_nodes(test_data)

    test_data['publication', 'cites', 'dataset'].edge_index = test_data['publication', 'cites', 'dataset'].edge_index_train
    test_data['publication', 'cites', 'dataset'].edge_label_index = torch.cat(
        [test_data['publication', 'cites', 'dataset'].edge_label_index_test,
         test_data['publication', 'cites', 'dataset'].negative_edge_label_index_test], dim=1)

    return train_data,validation_data,test_data

def load_inductive_data(root,inductive_type,indices=False):

    def remove_nodes(train_data):

        # del train_data['publication', 'hasentity', 'entity']
        # del train_data['dataset', 'hasentity', 'entity']
        # del train_data['entity']
        # del train_data['publication', 'hastopic', 'topic']
        # del train_data['dataset', 'hastopic', 'topic']
        # del train_data['topic']
        # del train_data['publication', 'hasauthor', 'author']
        # del train_data['dataset', 'hasauthor', 'author']
        # del train_data['author']
        if indices != [] and indices:
            train_data['dataset'].x[indices, :] = 1.0

        train_data['publication'].x = torch.cat([train_data['publication'].x, train_data['publication'].net_x], dim=1)
        train_data['dataset'].x = torch.cat([train_data['dataset'].x, train_data['dataset'].net_x], dim=1)

        if 'pubmed' in root:
            del train_data['publication', 'hasauthor', 'author']
            del train_data['dataset', 'hasauthor', 'author']
            del train_data['author']
            del train_data['publication', 'hasorganization', 'organization']
            del train_data['dataset', 'hasorganization', 'organization']
            del train_data['organization']
            del train_data['publication', 'hasvenue', 'venue']
            del train_data['venue']
            del train_data['publication', 'haskeyword', 'keyword']
            del train_data['dataset', 'haskeyword', 'keyword']
            del train_data['keyword']

        return train_data



    data = ScholarlyDataset(root=root)
    train_data = data[0]
    train_data = remove_nodes(train_data)



    edge_index_train = train_data['publication', 'cites', 'dataset'].edge_index_train
    negative_edge_index_train = train_data['publication', 'cites', 'dataset'].negative_edge_label_index_train
    edge_index_validation = train_data['publication', 'cites', 'dataset'].edge_label_index_validation
    negative_edge_index_validation = train_data['publication', 'cites', 'dataset'].negative_edge_label_index_validation
    edge_index_test = train_data['publication', 'cites', 'dataset'].edge_label_index_test
    # print(edge_index_train.shape)
    # print(edge_index_validation.shape)
    # print(edge_index_test.shape)

    to_avoid_sources_test = [x[0] for x in edge_index_test.t().tolist()]
    to_avoid_sources_vali = [x[0] for x in edge_index_validation.t().tolist()]
    to_avoid_dsts_test = [x[1] for x in edge_index_test.t().tolist()]
    to_avoid_dsts_vali = [x[1] for x in edge_index_validation.t().tolist()]
    filtered_edge_index_train = []
    filtered_edge_index_train_neg = []
    filtered_edge_index_vali = []
    filtered_edge_index_vali_neg = []
    if inductive_type == 'full':
        for x in edge_index_train.t().tolist():
            if x[0] not in to_avoid_sources_test + to_avoid_sources_vali and x[1] not in to_avoid_dsts_test + to_avoid_dsts_vali:
                filtered_edge_index_train.append(x)
        for x in edge_index_validation.t().tolist():
            if x[0] not in to_avoid_sources_test and x[1] not in to_avoid_dsts_test:
                filtered_edge_index_vali.append(x)
        for x in negative_edge_index_train.t().tolist():
            if x[0] not in to_avoid_sources_test + to_avoid_sources_vali and x[1] not in to_avoid_dsts_test + to_avoid_dsts_vali:
                filtered_edge_index_train_neg.append(x)
        for x in negative_edge_index_validation.t().tolist():
            if x[0] not in to_avoid_sources_test and x[1] not in to_avoid_dsts_test:
                filtered_edge_index_vali_neg.append(x)

    if inductive_type == 'light':
        for x in edge_index_train.t().tolist():
            if x[0] not in to_avoid_sources_test + to_avoid_sources_vali:
                filtered_edge_index_train.append(x)
        for x in edge_index_validation.t().tolist():
            if x[0] not in to_avoid_sources_test:
                filtered_edge_index_vali.append(x)
        for x in negative_edge_index_train.t().tolist():
            if x[0] not in to_avoid_sources_test + to_avoid_sources_vali:
                filtered_edge_index_train_neg.append(x)
        for x in negative_edge_index_validation.t().tolist():
            if x[0] not in to_avoid_sources_test:
                filtered_edge_index_vali_neg.append(x)


    negative_edge_index_validation = torch.tensor(filtered_edge_index_vali_neg).t()
    negative_edge_index_train = torch.tensor(filtered_edge_index_train_neg).t()
    edge_index_train = torch.tensor(filtered_edge_index_train).t()
    edge_index_validation = torch.tensor(filtered_edge_index_vali).t()

    # if inductive_type == 'full':
    #     train_mask = edge_index_validation[0].tolist() + edge_index_test[0].tolist() + edge_index_validation[1].tolist() + edge_index_test[1].tolist()
    #     validation_mask = edge_index_test[0].tolist() + edge_index_test[1].tolist()
    # elif inductive_type == 'light':
    #     train_mask = edge_index_validation[0].tolist() + edge_index_test[0].tolist()
    #     validation_mask = edge_index_test[0].tolist()
    # else:
    #     exit(1)
    # train_mask = ~np.isin(edge_index_train[0, :], list(set(train_mask)))
    # target_mask = ~np.isin(edge_index_train[1, :], list(set(train_mask)))
    # vali_mask = ~np.isin(edge_index_validation[0, :], list(set(validation_mask)))
    # target_mask_vali = ~np.isin(edge_index_validation[1, :], list(set(validation_mask)))
    # combined_mask_vali = vali_mask & target_mask_vali
    # combined_mask_train = train_mask & target_mask
    # edge_index_train = edge_index_train[:, combined_mask_train]
    # edge_index_validation = edge_index_validation[:, combined_mask_vali]
    #
    # neg_train_mask = ~np.isin(train_data['publication', 'cites', 'dataset'].negative_edge_label_index_train[0, :], list(set(train_mask)))
    # neg_target_mask = ~np.isin(train_data['publication', 'cites', 'dataset'].negative_edge_label_index_train[1, :], list(set(train_mask)))
    # neg_train_mask = neg_train_mask & neg_target_mask
    # neg_vali_mask = ~np.isin(train_data['publication', 'cites', 'dataset'].negative_edge_label_index_validation[0, :], list(set(validation_mask)))
    # neg_target_mask_vali = ~np.isin(train_data['publication', 'cites', 'dataset'].negative_edge_label_index_validation[1, :], list(set(validation_mask)))
    # neg_validation_mask = neg_vali_mask & neg_target_mask_vali
    # negative_edge_index_train = train_data['publication', 'cites', 'dataset'].negative_edge_label_index_train[:, neg_train_mask]
    # negative_edge_index_validation = train_data['publication', 'cites', 'dataset'].negative_edge_label_index_validation[:, neg_validation_mask]
    # print('\n\n')
    min_train = min([edge_index_train.size(1),negative_edge_index_train.size(1)])
    # print(edge_index_train.shape)
    edge_index_train = edge_index_train[:,:min_train]
    # print(edge_index_train.shape)
    negative_edge_index_train = negative_edge_index_train[:,:min_train]
    # print(negative_edge_index_train.shape)

    train_data['publication', 'cites', 'dataset'].edge_index_train = edge_index_train
    train_data['publication', 'cites', 'dataset'].edge_label_index_train = edge_index_train
    train_data['publication', 'cites', 'dataset'].negative_edge_label_index_train = negative_edge_index_train
    train_data['publication', 'cites', 'dataset'].edge_index = edge_index_train
    train_data['publication', 'cites', 'dataset'].edge_label_index = torch.cat([edge_index_train,negative_edge_index_train], dim=1)
    # print(train_data['publication', 'cites', 'dataset'].edge_label_index.shape)
    # print('\n\n')

    for edge_type, edge_index in train_data.edge_index_dict.items():
        edge_index = train_data[edge_type].edge_index
        edge_index = edge_index.t().tolist()
        edge_index = [x for x in edge_index if x[0] not in to_avoid_sources_test+to_avoid_sources_vali]
        if inductive_type == 'full' and 'author' not in edge_type:
            edge_index = [x for x in edge_index if x[1] not in to_avoid_dsts_test + to_avoid_dsts_vali]

        edge_index = torch.tensor(edge_index).t()
        train_data[edge_type].edge_index = edge_index
        # vali_mask = ~np.isin(train_data[edge_type].edge_index[0, :], list(set(train_mask)))
        # target_mask_vali = ~np.isin(train_data[edge_type].edge_index[1, :], list(set(train_mask)))
        # train_mask = vali_mask & target_mask_vali
        # train_data[edge_type].edge_index = train_data[edge_type].edge_index[:,train_mask]
    # print(train_data['publication', 'cites', 'dataset'].edge_label_index.shape)
    data = ScholarlyDataset(root=root)
    validation_data = data[0]
    validation_data = remove_nodes(validation_data)
    min_vali = min([edge_index_validation.size(1),negative_edge_index_validation.size(1)])
    negative_edge_index_validation = negative_edge_index_validation[:,:min_vali]
    edge_index_validation = edge_index_validation[:,:min_vali]

    validation_data['publication', 'cites', 'dataset'].negative_edge_label_index_validation = negative_edge_index_validation
    validation_data['publication', 'cites', 'dataset'].edge_label_index_validation = edge_index_validation
    validation_data['publication', 'cites', 'dataset'].edge_index = edge_index_train
    validation_data['publication', 'cites', 'dataset'].edge_label_index = torch.cat(
        [validation_data['publication', 'cites', 'dataset'].edge_label_index_validation,
         validation_data['publication', 'cites', 'dataset'].negative_edge_label_index_validation], dim=1)

    for edge_type, edge_index in validation_data.edge_index_dict.items():
        edge_index = validation_data[edge_type].edge_index
        edge_index = edge_index.t().tolist()
        edge_index = [x for x in edge_index if x[0] not in to_avoid_sources_vali]
        if inductive_type == 'full':
            edge_index = [x for x in edge_index if x[1] not in to_avoid_dsts_vali]

        edge_index = torch.tensor(edge_index).t()
        validation_data[edge_type].edge_index = edge_index
        # vali_mask = ~np.isin(train_data[edge_type].edge_index[0, :], list(set(validation_mask)))
        # target_mask_vali = ~np.isin(train_data[edge_type].edge_index[1, :], list(set(validation_mask)))
        # train_mask = vali_mask & target_mask_vali
        # validation_data[edge_type].edge_index = train_data[edge_type].edge_index[:,train_mask]


    data = ScholarlyDataset(root=root)
    test_data = data[0]
    test_data = remove_nodes(test_data)

    test_data['publication', 'cites', 'dataset'].edge_index = edge_index_train
    test_data['publication', 'cites', 'dataset'].edge_label_index = torch.cat(
        [test_data['publication', 'cites', 'dataset'].edge_label_index_test,
         test_data['publication', 'cites', 'dataset'].negative_edge_label_index_test], dim=1)
    # print(edge_index_train.shape)
    # print(edge_index_validation.shape)

    test_coupd = test_data['publication', 'cites', 'dataset'].edge_label_index_test.t().tolist()
    testsrc = [t[0] for t in test_coupd]
    testdst = [t[1] for t in test_coupd]
    train_coup = train_data['publication', 'cites', 'dataset'].edge_label_index_train.t().tolist()
    trainsrc = [t[0] for t in train_coup]
    traindst = [t[1] for t in train_coup]

    # print(set(testsrc).intersection(set(trainsrc)))
    # print(set(testdst).intersection(set(traindst)))


    return train_data,validation_data,test_data



# if __name__ == '__main__':
#     dataset = 'mes'
#     root = f'./datasets/{dataset}/split_transductive/train'
#     data = ScholarlyDataset(root=root)
#     train_data = data[0]
#     print(train_data)
#     print('\n')
#     root = f'./datasets/{dataset}/split_transductive/train_inductive_light'
#     data = ScholarlyDataset(root=root)
#     train_data = data[0]
#     print(train_data)
#     print('\n')
#     root = f'./datasets/{dataset}/split_transductive/train_inductive_full'
#     data = ScholarlyDataset(root=root)
#     train_data = data[0]
#     print(train_data)
#     print('\n')

    # dataset



