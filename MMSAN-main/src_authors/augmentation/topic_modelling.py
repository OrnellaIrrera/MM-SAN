import pickle
from nltk.stem import PorterStemmer,LancasterStemmer
import statistics
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, PartOfSpeech, MaximalMarginalRelevance
from sklearn.datasets import fetch_20newsgroups
import os
import spacy
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
# from cuml.cluster import HDBSCAN
# from cuml.manifold import UMAP
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-dataset", default='mes',choices=['mes','pubmed','pubmed_kcore'],
                    type=str)
parser.add_argument("-repr_model")
parser.add_argument("-save")
parser.add_argument("-cluster_size")
import time

# utils functions
def remove_plurals_with_stemming(nouns):
    stemmer = PorterStemmer()

    stemmed_nouns = [stemmer.stem(word) for word in nouns]
    singular_nouns = list(set(stemmed_nouns))  # Remove duplicates
    return singular_nouns

#------------------
def create_embdeddings(dataset):
    print(dataset)
    print('creating embeddings')
    path_embedding = f'./datasets/{dataset}/all/embeddings/'
    if not os.path.exists(path_embedding):
        os.makedirs(path_embedding)

    publications = pd.read_csv(f'./datasets/{dataset}/split_transductive/train/publications.csv')

    datasets = pd.read_csv(f'./datasets/{dataset}/split_transductive/train/datasets.csv',low_memory=False)

    publications = publications['content'].tolist()
    datasets = datasets['content'].tolist()
    docs = publications + datasets
    print(len(docs))

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(docs, show_progress_bar=True)
    with open(path_embedding+'content_embeddings.pickle', 'wb') as pkl:
        pickle.dump(embeddings, pkl)

def get_embeddings(dataset):
    print(dataset)
    path_embedding = f'./datasets/{dataset}/all/embeddings/'

    if not os.path.exists(path_embedding):
        os.makedirs(path_embedding)

    publications = pd.read_csv(f'./datasets/{dataset}/split_transductive/train/publications.csv')

    datasets = pd.read_csv(f'./datasets/{dataset}/split_transductive/train/datasets.csv',low_memory=False)
    publications_ids = publications['id'].tolist()
    datasets_ids = datasets['id'].tolist()
    docs_ids = publications_ids + datasets_ids

    publications = publications['content'].tolist()
    datasets = datasets['content'].tolist()

    docs = publications + datasets
    print(len(docs))
    if os.path.exists(path_embedding+'content_embeddings.pickle'):
        print('path exists')
        with open(path_embedding + 'content_embeddings.pickle', 'rb') as pkl:
            embeddings = pickle.load(pkl)
    else:
        print('path does not exist')
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedding_model.encode(docs, show_progress_bar=True)
        with open(path_embedding+'content_embeddings.pickle', 'wb') as pkl:
            pickle.dump(embeddings, pkl)
    return docs,docs_ids,publications_ids,datasets_ids,embeddings

def find_topics(dataset,repr_model = None,cluster_size=3):

    # embeddings
    docs,docs_ids,publications_ids,datasets_ids,embeddings = get_embeddings(dataset=dataset)
    # bertopic models definition
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=cluster_size, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    vectorizer_model = CountVectorizer(stop_words="english",ngram_range=(1,2))

    if not os.path.exists(f'./agumentation/topic_modelling/models_{str(cluster_size)}/'):
        os.makedirs(f'./agumentation/topic_modelling/models_{str(cluster_size)}/')
        os.makedirs(f'./agumentation/topic_modelling/models_{str(cluster_size)}/analyses')

    path = f'./agumentation/topic_modelling/models_{str(cluster_size)}/{dataset}_{str(repr_model).lower()}.json'

    # fine tune
    if repr_model == 'keybert':
        representation_model = KeyBERTInspired() # no embeddings in fit_transform
    elif repr_model == 'pos':
        representation_model = PartOfSpeech(spacy.load("en_core_web_sm"))
    elif repr_model == 'mmr':
        representation_model = MaximalMarginalRelevance(diversity=0.3)
    else:
        representation_model =None

    model_path = f'./agumentation/topic_modelling/models_{str(cluster_size)}/{dataset}_model_{str(repr_model).lower()}.pickle'

    # if not  os.path.exists(model_path):
    if repr_model == 'keybert':
        topic_model = BERTopic(vectorizer_model=vectorizer_model,embedding_model="all-MiniLM-L6-v2", hdbscan_model=hdbscan_model, n_gram_range=(1, 2),
                               umap_model=umap_model, representation_model=representation_model)
    else:
        topic_model = BERTopic(vectorizer_model=vectorizer_model,
                               hdbscan_model=hdbscan_model, n_gram_range=(1, 2),
                               umap_model=umap_model, representation_model=representation_model)
    #     topic_model.fit_transform(docs)
    # else:
    st = time.time()
    topic_model.fit_transform(docs, embeddings)
    end = time.time()
    print('training in: ',str(end-st))
    print("saving the model")
    topic_model.save(model_path)

    topics = topic_model.get_topics()
    t = {}
    for k,v in topics.items():
        t[str(k)] = [str(tup[0]) for tup in v]
    g = open(path,'w')
    json.dump(t,g,indent=4)
    g.close()

def save_topics_json(dataset,repr_model,cluster_size=3):

    # st = LancasterStemmer()
    # lemma_tags = {"NNS", "NNPS"}

    model_path = f'./agumentation/topic_modelling/models_{str(cluster_size)}/{dataset}_model_{str(repr_model).lower()}.pickle'
    path = f'./agumentation/topic_modelling/models_{str(cluster_size)}/analyses/{dataset}_{str(repr_model).lower()}.json'
    path_reduced = f'./agumentation/topic_modelling/models_{str(cluster_size)}/analyses/{dataset}_{str(repr_model).lower()}_reduced.json'

    topic_model = BERTopic.load(model_path)
    topics = topic_model.get_topics()
    t = {}
    for k, v in topics.items():
        t[str(k)] = [str(tup[0]) for tup in v]
    g = open(path, 'w')
    json.dump(t, g, indent=4)
    g.close()
    nlp = spacy.load("en_core_web_lg")

    new_json = {}
    for k, v in topics.items():
            # print(v)
            new_list = []
            vv = [str(tup[0]) for tup in v]
            for word in vv:
                word = nlp(word)
                lemmatized_word = ''
                for w in word:
                    lemmatized_word += w.lemma_ + ' '
                new_list.append(lemmatized_word.strip())
            # new_list = [st.stem(word) for word in vv]
            reduced = list(set(new_list))
            new_json[str(k)] = reduced
            # if len(new_json[str(k)]) != len(v):
            #     print(k)

    g = open(path_reduced, 'w')
    json.dump(new_json, g, indent=4)
    g.close()



def get_topic_info(dataset,repr_model,save=False,cluster_size=3):
    if save:
        f = open(f'topic_info_{dataset}_{repr_model}.txt', 'w')
        sys.stdout = f

    model_path = f'./agumentation/topic_modelling/models_{cluster_size}/{dataset}_model_{str(repr_model).lower()}.pickle'
    path_csv = f'./agumentation/topic_modelling/models_{cluster_size}/analyses/documents_{dataset}_{str(repr_model).lower()}_documents.csv'
    path_csv_t = f'./agumentation/topic_modelling/models_{cluster_size}/analyses/documents_{dataset}_{str(repr_model).lower()}_topics.csv'
    path_reduced = f'./agumentation/topic_modelling/models_{cluster_size}/analyses/{dataset}_{str(repr_model).lower()}_reduced.json'
    data = json.load(open(path_reduced, 'r'))
    # count shared topics
    all_keys = []
    restricted_topics = []

    for k, t in data.items():
        if any("publication" in word.lower() for word in t) or any("author" in word.lower() for word in t):
            restricted_topics.append(int(k))
            print(k)
        if int(k) not in restricted_topics and int(k) != -1:
            all_keys.extend(t)
    print(f'the total number of restricted topics is: {len(restricted_topics)}')

    docs,docs_ids,publications_ids,datasets_ids,embeddings = get_embeddings(dataset=dataset)
    print(len(docs),len(docs_ids))
    print(len(publications_ids),len(datasets_ids))
    print(len(embeddings))
    try:
        topic_model = BERTopic.load(model_path)

        d = topic_model.get_document_info(docs)
        d = d[['Topic','Name','Top_n_words']]
        d.to_csv(path_csv,index=False)
        # print(d)
        t = topic_model.get_topic_info()
        t = t[['Topic','Count','Name','Representation']]

        t.to_csv(path_csv_t,index=False)
        no_topic = t[t['Topic'] == -1]['Count'].values[0]

        for top in restricted_topics:
            print(f'restricted topic: {top} count: {t[t["Topic"] == top]["Count"].values[0]}')
        print(t.shape[0])
        restricted_topics.append(-1)
        t = t[~t['Topic'].isin(restricted_topics)]
        print(t.shape[0])

        t = t.sort_values(by=['Count'], ascending=False)



        # print(t)
        maxt = t['Count'].tolist()[:5]
        mint = t['Count'].tolist()[-5:]
        # less_20 = t[t['Count'] < 21]
        medt = statistics.median(t['Count'].tolist())

        print(f'the total number of topics is: {len(t["Count"]-1)}')
        print(f'the 5 max populated topics contain: {maxt} element')
        print(f'the 5 min populated topics contain: {mint} element')
        print(f'the median of documents per topic is {medt}')
        print(f'the number of documents without topic is {no_topic}')
        # print(f'the number of topics with less than 20 documents is {less_20.shape[0]}')


        pubs_top = d['Topic'].tolist()[0:(len(publications_ids))]
        # print(len(pubs_top),len(publications_ids))
        pubs_no_top = sum([1 for s in pubs_top if s == -1 or s in restricted_topics])
        pubs_w_top = sum([1 for s in pubs_top if s > -1 and s not in restricted_topics])
        print(f'the number of publications with topic is {pubs_w_top}')
        print(f'the number of publications without topic is {pubs_no_top}')

        # dats = [i for i,p in enumerate(docs_ids) if p.startswith('d')]
        dats_top = d['Topic'].tolist()[len(publications_ids):]
        # print(len(dats_top),len(datasets_ids))
        dats_no_top = sum([1 for s in dats_top if s == -1 or s in restricted_topics])
        dats_w_top = sum([1 for s in dats_top if s > -1 and s not in restricted_topics])
        print(f'the number of datasets with topic is {dats_w_top}')
        print(f'the number of datasets without topic is {dats_no_top}')

        doc_top = d['Topic'].tolist()
        doc_no_top = sum([1 for s in doc_top if s == -1 or s in restricted_topics])
        doc_w_top = sum([1 for s in doc_top if s > -1 and s not in restricted_topics])
        print(f'the number of documents with topic is {doc_w_top}')
        print(f'the number of documents without topic is {doc_no_top}')

        print(f'the number of keywords in all the topics is {len(list(set(all_keys)))}')
        unique_elements = set(all_keys)
        element_counts = {element: all_keys.count(element) for element in unique_elements}
        max_pair = max(element_counts.items(), key=lambda x: x[1])
        result_pairs = {k: v for k, v in element_counts.items() if v > 1}
        print(f'the most recurrent keyword is {max_pair[0]} which recurs {max_pair[1]}')
        print(f'the keywords occurring more than once are: {len(result_pairs)}')

    except Exception as e:
        print(e)


def create_topic_nodes(dataset,repr_model,cluster_size):

    """This method creates for each topic keyword a new node each node has id, keyword and topic_id attributes (keyword) and for each topic a node has id, description (with keywords) and topic_id attributes (attributed)"""

    path_reduced = f'./agumentation/topic_modelling/models_{cluster_size}/analyses/{dataset}_{str(repr_model).lower()}_reduced.json'
    path_keyword = f'./datasets/{dataset}/split_transductive/train/topics_keywords_{str(cluster_size)}.csv'
    path_keyword_full = f'./datasets/{dataset}/split_transductive/train/topics_keywords_full_{str(cluster_size)}.csv'
    path_attributed = f'./datasets/{dataset}/split_transductive/train/topics_attributed_{str(cluster_size)}.csv'
    path_pubtopic_keyword = f'./datasets/{dataset}/split_transductive/train/pubtopicedges_keywords_{str(cluster_size)}.csv'
    path_pubtopic_attributed = f'./datasets/{dataset}/split_transductive/train/pubtopicedges_attributed_{str(cluster_size)}.csv'
    path_datatopic_keyword = f'./datasets/{dataset}/split_transductive/train/datatopicedges_keywords_{str(cluster_size)}.csv'
    path_datatopic_attributed = f'./datasets/{dataset}/split_transductive/train/datatopicedges_attributed_{str(cluster_size)}.csv'

    docs,docs_ids,publications_ids,datasets_ids,embeddings = get_embeddings(dataset=dataset)

    data = json.load(open(path_reduced, 'r'))
    df_keywords = pd.DataFrame()
    df_keywords_filtered = pd.DataFrame()
    kids = []
    ktopics = []
    kwords = []
    df_attributed = pd.DataFrame()
    topic_ids = []
    descs = []
    tids = []
    c = 0
    print('check!!')

    for k, t in data.items():
        if not (any("publication" in word.lower() for word in t) or any("author" in word.lower() for word in t)
                or str(k) == "-1" or k == -1):
            topic_id = f't_{str(k)}'
            topic_ids.append(topic_id)
            descs.append(' ,'.join(t))
            tids.append(int(k))
            for i,word in enumerate(t):
                id = f'tk_{str(c)}'
                c+=1
                if word in kwords:
                    print(word,str(kids[kwords.index(word)]))
                    kids.append(kids[kwords.index(word)])
                else:
                    kids.append(id)
                ktopics.append(topic_id)
                kwords.append(word)

    df_keywords['id'] = kids
    print(len(topic_ids))
    df_keywords['topic_id'] = ktopics
    df_keywords['description'] = kwords
    keys = list(set(kwords))
    ids = [f'tk_{str(c)}' for c,k in enumerate(keys)]
    # df_keywords_filtered['id'] = ids
    # df_keywords_filtered['description'] = keys


    df_attributed['id'] = topic_ids
    df_attributed['description'] = descs

    path_csv = f'./agumentation/topic_modelling/models_{cluster_size}/analyses/documents_{dataset}_{str(repr_model).lower()}_documents.csv'
    df_docs = pd.read_csv(path_csv)
    df_docs['id'] = docs_ids
    df_docs = df_docs[df_docs['Topic'].isin(tids)]
    df_docs['Topic'] = 't_' + df_docs['Topic'].astype(str)

    df = df_docs.rename(columns={'Topic': 'target', 'id': 'source'})

    df_pubs = df[df['source'].str.startswith('p')]
    df_pubs = df_pubs[['source','target']]
    df_data = df[df['source'].str.startswith('d')]
    df_data = df_data[['source','target']]


    df_attributed.to_csv(path_attributed,index=False)
    print(f'single topics: {df_attributed.shape[0]}')

    # df_keywords_filtered.to_csv(path_keyword,index=False)
    # print(f'single topics keywords: {df_keywords_filtered.shape[0]}')
    df_pubs.to_csv(path_pubtopic_attributed,index=False)
    print(f'single topics pub-topic: {df_pubs.shape[0]}')
    df_data.to_csv(path_datatopic_attributed,index=False)
    print(f'single topics data-topic: {df_data.shape[0]}')

    df_k_pubs = pd.merge(df_keywords, df_pubs, left_on='topic_id', right_on='target', how='inner')
    df_k_pubs = df_k_pubs[['id','topic_id','source']]
    df_k_dats = pd.merge(df_keywords, df_data, left_on='topic_id', right_on='target', how='inner')
    df_k_dats = df_k_dats[['id','topic_id','source']]
    df_k_pubs = df_k_pubs.rename(columns={'id': 'target'})
    df_k_dats = df_k_dats.rename(columns={'id': 'target'})

    df_k_pubs.drop_duplicates().to_csv(path_pubtopic_keyword,index=False)
    print(f'single topickeys pub-keywords: {df_k_pubs.shape[0]}')
    df_k_dats.drop_duplicates().to_csv(path_datatopic_keyword,index=False)
    print(f'single topickeys data-keywords: {df_k_dats.shape[0]}')
    df_keywords.to_csv(path_keyword_full,index=False)
    df_keywords = df_keywords[['id','description']].drop_duplicates()
    df_keywords.to_csv(path_keyword,index=False)

def create_edges_with_authors_venues(dataset, attr='attributed'):

    """This method creates edges files: from authors to topics, from venues to topics"""

    df_rels_pubs = pd.read_csv(f'./agumentation/topic_modelling/topics/{dataset}/pubtopicedges_{attr}.csv')
    df_rels_data = pd.read_csv(f'./agumentation/topic_modelling/topics/{dataset}/datatopicedges_{attr}.csv')

    topics = pd.read_csv(f'./agumentation/topic_modelling/topics/{dataset}/topics_{attr}.csv')
    pubauthedges = pd.read_csv(f'./datasets/{dataset}/split_transductive/train/pubauthedges.csv')
    dataauthedges = pd.read_csv(f'./datasets/{dataset}/split_transductive/train/dataauthedges.csv')
    publications = pd.read_csv(f'./datasets/{dataset}/split_transductive/train/publications.csv')['id'].unique().tolist()

    if 'mes' == dataset:
        publications = pd.read_csv(f'./datasets/{dataset}/split_transductive/train/publications.csv')['id'].unique().tolist()

    datasets = pd.read_csv(f'./datasets/{dataset}/split_transductive/train/datasets.csv')['id'].unique().tolist()
    df_rels_pubs = df_rels_pubs[df_rels_pubs['source'].isin(publications)]
    df_rels_data = df_rels_data[df_rels_data['source'].isin(datasets)]

    df_rels_pubs.drop_duplicates().to_csv(f'./datasets/{dataset}/split_transductive/train/pubtopicedges_{attr}.csv', index=False)
    df_rels_data.drop_duplicates().to_csv(f'./datasets/{dataset}/split_transductive/train/datatopicedges_{attr}.csv', index=False)
    topics.drop_duplicates().to_csv(f'./datasets/{dataset}/split_transductive/train/topics_{attr}.csv', index=False)


    df_rels_pubs.rename(columns={'target': 'target1', 'source': 'source1'}, inplace=True)
    df_rels_data.rename(columns={'target': 'target1', 'source': 'source1'}, inplace=True)

    if dataset != 'mes':
        pubvenedges = pd.read_csv(f'./datasets/{dataset}/split_transductive/train/pubvenuesedges.csv')
        pubvenueentedges = pd.merge(pubvenedges, df_rels_pubs, left_on='source', right_on='source1', how='outer')
        pubvenueentedges = pubvenueentedges[['target', 'target1']]
        pubvenueentedges.rename(columns={'target': 'source', 'target1': 'target'}, inplace=True)

        # pubvenueentedges.to_csv(f'./agumentation/topic_modelling/topics/{dataset}/venuestopicsedges_{attr}.csv', index=False)
        pubvenueentedges.to_csv(f'./datasets/{dataset}/split_transductive/train/venuestopicsedges_{attr}.csv', index=False)
        print(pubvenueentedges.shape)

    print(f'inner1 pub author {attr}')
    st = time.time()
    # f'./datasets/{dataset}/split_transductive/train/
    pubauthedgesentities = pd.merge(pubauthedges, df_rels_pubs, left_on='source', right_on='source1', how='outer')
    pubauthedgesentities = pubauthedgesentities[['target', 'target1']]
    pubauthedgesentities.rename(columns={'target': 'source', 'target1': 'target'}, inplace=True)
    # pubauthedgesentities.to_csv(f'./agumentation/topic_modelling/topics/{dataset}/pubauthtopicsedges_{attr}.csv', index=False)
    pubauthedgesentities.to_csv(f'./datasets/{dataset}/split_transductive/train/pubauthtopicsedges_{attr}.csv', index=False)
    print(str(time.time() - st))
    print(pubauthedgesentities.shape)
    print(f'inner2 data author {attr}')
    st = time.time()
    dataauthedgesentities = pd.merge(dataauthedges, df_rels_data, left_on='source', right_on='source1', how='outer')
    dataauthedgesentities = dataauthedgesentities[['target', 'target1']]
    dataauthedgesentities.rename(columns={'target': 'source', 'target1': 'target'}, inplace=True)
    # dataauthedgesentities.to_csv(f'./agumentation/topic_modelling/topics/{dataset}/dataauthtopicsedges_{attr}.csv', index=False)
    dataauthedgesentities.to_csv(f'./datasets/{dataset}/split_transductive/train/dataauthtopicsedges_{attr}.csv', index=False)
    print(str(time.time() - st))
    print(dataauthedgesentities.shape)

def create_topics_in_graph(dataset,repr_model,cluster_size):
    create_topic_nodes(dataset,repr_model,cluster_size)
    # create_edges_with_authors_venues(dataset,'attributed')
    # create_edges_with_authors_venues(dataset,'keywords')

def main():
    args = parser.parse_args()

    dataset = args.dataset
    print('dataset',dataset)
    repr_model = args.repr_model
    save = args.save
    print(f'save {save}')
    cluster_size = args.cluster_size
    print(f'cluster_size {cluster_size}')
    # for dataset in ['mes','pubmed_kcore','pubmed']:
    #     create_embdeddings(dataset)

    for dataset in ['mes','mes_full','pubmed_kcore','pubmed']:
        path_embedding = f'./datasets/{dataset}/all/embeddings/'
        if os.path.exists(path_embedding + 'content_embeddings.pickle'):
            os.remove(path_embedding + 'content_embeddings.pickle')
        for repr_model in ['keybert']:
            for cluster_size in [10,2,3,5]:
                print(f'{dataset} - {str(cluster_size)}')
                find_topics(dataset,repr_model,cluster_size)
                save_topics_json(dataset,repr_model,cluster_size)
                get_topic_info(dataset,repr_model,save,cluster_size)
        repr_model = 'keybert'
        for cluster_size in  [2,3,5,10]:
            create_topics_in_graph(dataset,repr_model,cluster_size)

if __name__ == '__main__':
    # main()
    args = parser.parse_args()
    dataset = args.dataset
    print('dataset',dataset)
    repr_model = args.repr_model
    save = args.save
    print(f'save {save}')
    cluster_size = args.cluster_size
    print(f'cluster_size {cluster_size}')
    get_topic_info(dataset,repr_model,save,cluster_size)
    path_datatopic_attributed = f'./datasets/{dataset}/split_transductive/train/datatopicedges_attributed_{str(cluster_size)}.csv'
    path_pubatopic_attributed = f'./datasets/{dataset}/split_transductive/train/pubtopicedges_attributed_{str(cluster_size)}.csv'
    pubto = pd.read_csv(path_pubatopic_attributed)
    datato = pd.read_csv(path_datatopic_attributed)
    print(f'publication-topic edges: {pubto.shape[0]}')
    print(f'data-topic edges: {datato.shape[0]}')

