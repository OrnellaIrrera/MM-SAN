import pickle
import statistics
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-dataset", default='mes',choices=['mes','pubmed','pubmed_kcore'],
                    type=str)
parser.add_argument("-cutoff",default="1,5,10")
import time
import shutil



def create_embeddings_all(dataset):
    print(dataset)
    path_embedding = f'./baselines/trivial/data/{dataset}/'

    # trivial baseline: take the queries in the test set and get the ranking of all the datasets for each query
    publications_all = pd.read_csv(f'./datasets/{dataset}/all/final/publications.csv')
    datasets_all = pd.read_csv(f'./datasets/{dataset}/all/final/datasets.csv', low_memory=False)
    pdedges_all = pd.read_csv(f'./datasets/{dataset}/all/final/pubdataedges.csv', low_memory=False)
    datasets_all = datasets_all[datasets_all['id'].isin(pdedges_all['target'].unique().tolist())]
    publications_all = publications_all[publications_all['id'].isin(pdedges_all['source'].unique().tolist())]
    publications_ids = publications_all['id'].tolist()
    print(f'total publications: {len(publications_ids)}')
    datasets_ids = datasets_all['id'].tolist()
    print(f'total dataset: {len(datasets_ids)}')
    docs_ids = publications_ids + datasets_ids

    publications_all = publications_all['content'].tolist()
    datasets_all = datasets_all['content'].tolist()

    docs = publications_all + datasets_all
    print(len(docs))

    print('path does not exist')
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(docs, show_progress_bar=True)

    return docs_ids, publications_ids, datasets_ids, embeddings

def create_embeddings(dataset):
    print(dataset)
    path_embedding = f'./baselines/trivial/data/{dataset}/'

    if os.path.exists(path_embedding):

        shutil.rmtree(path_embedding)
    os.makedirs(path_embedding)

    # trivial baseline: take the queries in the test set and get the ranking of all the datasets for each query
    publications_test = pd.read_csv(f'./datasets/{dataset}/split_transductive/test/publications.csv')
    datasets_all = pd.read_csv(f'./datasets/{dataset}/all/final/datasets.csv', low_memory=False)
    pdedges_all = pd.read_csv(f'./datasets/{dataset}/all/final/pubdataedges.csv', low_memory=False)
    datasets_all = datasets_all[datasets_all['id'].isin(pdedges_all['target'].unique().tolist())]
    pd_edges = pd.read_csv(f'./datasets/{dataset}/split_transductive/train/pubdataedges_kcore_t.csv', low_memory=False)
    publications_test = publications_test[publications_test['id'].isin(pd_edges['source'].unique().tolist())]
    publications_ids = publications_test['id'].tolist()
    print(f'total publications: {len(publications_ids)}')
    datasets_ids = datasets_all['id'].tolist()
    print(f'total dataset: {len(datasets_ids)}')
    docs_ids = publications_ids + datasets_ids
    publications_test = publications_test['content'].tolist()
    datasets_all = datasets_all['content'].tolist()

    docs = publications_test + datasets_all
    print(len(docs))

    print('path does not exist')
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(docs, show_progress_bar=True)
    with open(path_embedding + 'content_embeddings.pickle', 'wb') as pkl:
        pickle.dump(embeddings, pkl)
    return docs_ids, publications_ids, datasets_ids, embeddings



def cosine_similarity(publication_emb, dataset_emb):
    dot_product = np.dot(publication_emb, dataset_emb)
    norm_vector1 = np.linalg.norm(publication_emb)
    norm_vector2 = np.linalg.norm(dataset_emb)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

def create_ranking(dataset):
    final_scores = {}
    docs_ids, publications_ids, datasets_ids, embeddings = create_embeddings(dataset)
    publications = embeddings[:len(publications_ids)]
    datasets = embeddings[len(publications_ids):]

    for i,publication in enumerate(publications):
        print(f'query: {i}')
        final_scores[publications_ids[i]] = []
        scores = []
        for j,data in enumerate(datasets):
            scores.append(tuple([datasets_ids[j],cosine_similarity(publication,data)]))
        scores = sorted(scores, key = lambda x: x[1],reverse=True)
        final_scores[publications_ids[i]] = [x[0] for x in scores][:100]
    f = open(f'./baselines/trivial/data/{dataset}/results.json','w')
    print('saving')
    json.dump(final_scores,f)


def evaluate(dataset,k=10):
    print(f'evaluation at K = {k}')
    precision, recall, ndcg = 0, 0, 0
    f = open(f'./baselines/trivial/data/{dataset}/results.json','r', encoding='utf-8')
    print('restoring')
    data = json.load(f)
    # print(data)
    def concatenate_targets(group):
        return ' '.join(group)
    #
    # # build ground truth
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


    if not os.path.exists(f'./baselines/trivial/data/{dataset}/groundtruth.json'):
        pd_edges = pd.read_csv(f'./datasets/{dataset}/split_transductive/train/pubdataedges_kcore_t.csv')
        concatenated_targets = pd_edges.groupby('source')['target'].agg(concatenate_targets).reset_index()
        ground_truth = {}
        for i,row in concatenated_targets.iterrows():
            ground_truth[row['source']] = row['target'].split()
        f = open(f'./baselines/trivial/data/{dataset}/groundtruth.json','w')
        json.dump(ground_truth,f)
    else:
        g = open(f'./baselines/trivial/data/{dataset}/groundtruth.json','r')
        ground_truth = json.load(g)
    #
    queries = len(list(data.keys()))
    for key,value in data.items():
        # print(key)
        prediction = value[:k]
        true_values = ground_truth[key]
        # print(prediction,true_values)
        correct = list(set(prediction) & set(true_values))
        precision_cur = len(correct) / k
        precision += precision_cur

        recall_cur = len(correct) / len(true_values)
        recall += recall_cur
        # print(precision_cur,recall_cur)


        # relevance_scores = [1 if pid in true_values else 0 for pid in prediction]
        # dcg = np.sum(relevance_scores / np.log2(np.arange(len(relevance_scores)) + 2))
        # ideal_relevance_scores = sorted(relevance_scores, reverse=True)
        # idcg = np.sum(ideal_relevance_scores / np.log2(np.arange(len(ideal_relevance_scores)) + 2))
        #
        # # Compute Normalized DCG (NDCG)
        # if idcg == 0:
        #     ndcg += 0  # Handle the case when there are no relevant items
        # else:
        #     ndcg += dcg / idcg

        ndcg += ndcg_at_k(true_values, prediction, k)
    return precision/queries, recall/queries, ndcg/queries



def main():
    args = parser.parse_args()
    dataset = args.dataset
    cutoffs = args.cutoff.split(',')
    cutoffs = [int(i) for i in cutoffs]

    create_ranking(dataset)
    for k in cutoffs:
        precision, recall, ndcg = evaluate(dataset,k = k)
        print(precision,recall,ndcg)
        print('\n\n')


def count_rels():
    args = parser.parse_args()

    dataset = args.dataset
    torem = 0
    pd_edges = pd.read_csv(f'./datasets/{dataset}/all/final/pubpubedges.csv')
    docs_ids, publications_ids, datasets_ids, embeddings = create_embeddings_all(dataset)
    sources,targets = pd_edges['source'].tolist(),pd_edges['target'].tolist()
    for i,source in enumerate(sources):
        print(i)
        s = embeddings[docs_ids.index(source)]
        target = targets[i]
        t = embeddings[docs_ids.index(target)]
        score = cosine_similarity(s,t)
        if score > 0.8:
            torem += 1
    print(f'all: {pd_edges.shape[0]}')
    print(f'remove: {torem}')

def count_relevance():
    args = parser.parse_args()
    dataset = args.dataset
    torem,notrem = 0,0
    pd_edges = pd.read_csv(f'./datasets/{dataset}/all/final/pubpubedges.csv')
    sources,targets = pd_edges['source'].tolist(),pd_edges['target'].tolist()
    new_df = pd.DataFrame()
    rels = []
    for source,target in zip(sources,targets):
        rels.append(tuple([source,target]))

    docs_ids, publications_ids, datasets_ids, embeddings = create_embeddings_all(dataset)
    for i in range(len(publications_ids)-1):
        print(i,torem, notrem)
        source = publications_ids[i]
        s = embeddings[docs_ids.index(source)]
        for j in range(i+1,len(publications_ids)):
            target = publications_ids[j]
            t = embeddings[docs_ids.index(target)]
            score = cosine_similarity(s, t)
            if score > 0.75:
                if tuple([source,target]) not in rels:
                    torem += 1
                    sources.append(source)
                    targets.append(target)
                else:
                    notrem += 1
    new_df['source'] = sources
    new_df['target'] = targets
    print(new_df.shape)
    new_df.to_csv(f'./datasets/{dataset}/all/final/pubpubedges_enriched.csv',index=False)
    print(f'remove: {torem} {notrem}')

if __name__ == '__main__':
    main()


