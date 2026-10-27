"""
Prepare files: analyze the following points:
    - number of edges, nodes subdivided per type
    - density
    - number of connected components with and without datasets
    - train and test splits
    - number of duplicated nodes -- different index but same title and description
    - number of max, min, median, avg number of edges of the nodes
    - length of descriptions
"""
import statistics

import sys
import networkx as nx
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, default='pubmed',
                    choices=['mes','pubmed_kcore','pubmed','mes_full'])
args = parser.parse_args()
from collections import Counter
import numpy as np


def get_final_stats(dataset):
    path_topic = f'./topic_modelling/topics/{dataset}/'
    path_entities = f'./topic_modelling/entities/{dataset}/'
    path = f'./datasets/{dataset}/all/final/'
    path_topic = path
    path_entities = path
    f = open('./dataset_preprocessing/results/final_after_tm_graph_stats_'+args.dataset+'.txt','w')
    sys.stdout = f
    # count nodes
    if 'mes' not in path:
        publications = pd.read_csv(path + 'publications.csv')
        pub_obj = publications['id'].unique()
        print(f'THE TOTAL COUNT OF PUBLICATIONS IS {len(pub_obj)}')

    else:
        # publications = pd.read_csv(path + 'publications_filtered.csv')
        publications = pd.read_csv(path + 'publications.csv')
        pub_obj = publications['id'].unique()
        print(f'THE TOTAL COUNT OF PUBLICATIONS IS {len(pub_obj)}')

    datasets = pd.read_csv(path + 'datasets.csv')
    data_obj = datasets['id'].unique()
    print(f'THE TOTAL COUNT OF DATASETS IS {len(data_obj)}')

    authors = pd.read_csv(path + 'authors.csv')
    auth_obj = authors['id'].unique()
    print(f'THE TOTAL COUNT OF AUTHORS IS {len(auth_obj)}')

    topics = pd.read_csv(path_topic + 'topics_attributed_3.csv')
    print(f'THE TOTAL COUNT OF TOPICS IS {topics.shape[0]}')

    entities = pd.read_csv(path_entities + 'entities.csv')
    print(f'THE TOTAL COUNT OF ENTITIES IS {entities.shape[0]}')


    if 'mes' not in path:
        pubpubedges = pd.read_csv(path + 'pubpubedges.csv')
        print(f'THE TOTAL COUNT OF EDGES BETWEEN PUBLICATIONS IS {pubpubedges.shape[0]}')
    else:
        pubpubedges = pd.read_csv(path + 'pubpubedges.csv')
        print(f'THE TOTAL COUNT OF EDGES BETWEEN PUBLICATIONS IS {pubpubedges.shape[0]}')

    pubdataedges = pd.read_csv(path + 'pubdataedges.csv')
    print(f'THE TOTAL COUNT OF EDGES BETWEEN PUBLICATIONS AND DATASETS IS {pubdataedges.shape[0]}')

    datadataedges = pd.read_csv(path + 'datadataedges.csv')
    print(f'THE TOTAL COUNT OF EDGES BETWEEN DATASETS IS {datadataedges.shape[0]}')

    pubauthedges = pd.read_csv(path + 'pubauthedges.csv')
    print(f'THE TOTAL COUNT OF EDGES BETWEEN PUBLICATIONS AND AUTHORS IS {pubauthedges.shape[0]}')

    dataauthedges = pd.read_csv(path + 'dataauthedges.csv')
    print(f'THE TOTAL COUNT OF EDGES BETWEEN DATASETS AND AUTHORS IS {dataauthedges.shape[0]}')

    # topics

    pubtopicedges = pd.read_csv(path_topic + 'pubtopicedges_attributed_3.csv')
    print(f'THE TOTAL COUNT OF EDGES BETWEEN PUBLICATIONS AND TOPICS IS {pubtopicedges.shape[0]}')

    datatopicedges = pd.read_csv(path_topic + 'datatopicedges_attributed_3.csv')
    print(f'THE TOTAL COUNT OF EDGES BETWEEN DATASETS AND TOPICS IS {datatopicedges.shape[0]}')

    # pubauthtopicsedges = pd.read_csv(path_topic + 'pubauthtopicsedges_attributed_3.csv')
    # print(f'THE TOTAL COUNT OF EDGES BETWEEN PUBLICATIONS AUTHORS AND TOPICS IS {pubauthtopicsedges.shape[0]}')
    #
    # dataauthtopicsedges = pd.read_csv(path_topic + 'dataauthtopicsedges_attributed_3.csv')
    # print(f'THE TOTAL COUNT OF EDGES BETWEEN DATASETS AUTHORS AND TOPICS IS {dataauthtopicsedges.shape[0]}')

    # entities
    pubentedges = pd.read_csv(path_entities + 'pubentedges.csv')
    print(f'THE TOTAL COUNT OF EDGES BETWEEN PUBLICATIONS AND ENTITIES IS {pubentedges.shape[0]}')

    dataentedges = pd.read_csv(path_entities + 'dataentedges.csv')
    print(f'THE TOTAL COUNT OF EDGES BETWEEN DATASETS AND ENTITIES IS {dataentedges.shape[0]}')

    # pubauthentedges = pd.read_csv(path_entities + 'pubauthentedges.csv')
    # print(f'THE TOTAL COUNT OF EDGES BETWEEN PUBLICATIONS AUTHORS AND ENTITIES IS {pubauthentedges.shape[0]}')
    #
    # dataauthentedges = pd.read_csv(path_entities + 'dataauthentedges.csv')
    # print(f'THE TOTAL COUNT OF EDGES BETWEEN DATASETS AUTHORS AND ENTITIES IS {dataauthentedges.shape[0]}')

    edges_concat = pd.concat([pubpubedges, pubdataedges, pubauthedges, dataauthedges, datadataedges,
                              pubtopicedges,datatopicedges,
                              pubentedges,dataentedges], ignore_index=True)

    if 'mes' not in path:
        venues = pd.read_csv(path + 'venues.csv')
        venues_obj = venues['id'].unique()
        print(f'THE TOTAL COUNT OF VENUES IS {len(venues_obj)}')

        organizations = pd.read_csv(path + 'organizations.csv')
        organizations_obj = organizations['id'].unique()
        print(f'THE TOTAL COUNT OF ORGANIZATIONS IS {len(organizations_obj)}')

        keywords = pd.read_csv(path + 'keywords.csv')
        keywords_obj = keywords['id'].unique()
        print(f'THE TOTAL COUNT OF KEYWORDS IS {len(keywords_obj)}')

        puborgedges = pd.read_csv(path + 'puborgedges.csv')
        print(f'THE TOTAL COUNT OF EDGES BETWEEN PUBLICATIONS AND ORGANIZATIONS IS {puborgedges.shape[0]}')

        dataorgedges = pd.read_csv(path + 'dataorgedges.csv')
        print(f'THE TOTAL COUNT OF EDGES BETWEEN DATASETS AND ORGANIZATIONS IS {dataorgedges.shape[0]}')

        pubvenuesedges = pd.read_csv(path + 'pubvenuesedges.csv')
        print(f'THE TOTAL COUNT OF EDGES BETWEEN PUBLICATIONS AND VENUES IS {pubvenuesedges.shape[0]}')

        pubkeywedges = pd.read_csv(path + 'pubkeyedges.csv')
        print(f'THE TOTAL COUNT OF EDGES BETWEEN PUBLICATIONS AND KEYWORDS IS {pubkeywedges.shape[0]}')

        datakeywedges = pd.read_csv(path + 'datakeyedges.csv')
        print(f'THE TOTAL COUNT OF EDGES BETWEEN DATASETS AND KEYWORDS IS {datakeywedges.shape[0]}')

        # venuestopicsedges = pd.read_csv(path_topic + 'venuestopicsedges_attributed_3.csv')
        # print(f'THE TOTAL COUNT OF EDGES BETWEEN VENUES AND TOPICS IS {venuestopicsedges.shape[0]}')
        #
        # venuesentedges = pd.read_csv(path_entities + 'venuesentedges.csv')
        # print(f'THE TOTAL COUNT OF EDGES BETWEEN VENUES AND ENTITIES IS {venuestopicsedges.shape[0]}')

        edges_concat = pd.concat([edges_concat, pubkeywedges, datakeywedges, pubvenuesedges, puborgedges, dataorgedges],
                                 ignore_index=True)

    # find max, min, median degree for each edge type

    G = nx.from_pandas_edgelist(pubpubedges, 'source', 'target')
    combined_graph = nx.from_pandas_edgelist(edges_concat, 'source', 'target')

    degrees = [int(deg) for node, deg in G.degree()]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum Degree publication-publication: {max_degree}")
    print(f"Minimum Degree publication-publication: {min_degree}")
    print(f"Median Degree publication-publication: {median_degree}")
    print('\n\n')

    G = nx.from_pandas_edgelist(pubdataedges, 'source', 'target')

    degrees = [int(deg) for node, deg in G.degree()]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum Degree publication-dataset: {max_degree}")
    print(f"Minimum Degree publication-dataset: {min_degree}")
    print(f"Median Degree publication-dataset: {median_degree}")
    print('\n\n')

    G = nx.from_pandas_edgelist(datadataedges, 'source', 'target')

    degrees = [int(deg) for node, deg in G.degree()]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum Degree dataset-dataset: {max_degree}")
    print(f"Minimum Degree dataset-dataset: {min_degree}")
    print(f"Median Degree dataset-dataset: {median_degree}")
    print('\n\n')

    G = nx.from_pandas_edgelist(pubauthedges, 'source', 'target')
    degrees = [int(deg) for node, deg in G.degree()]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum Degree publication-author: {max_degree}")
    print(f"Minimum Degree publication-author: {min_degree}")
    print(f"Median Degree publication-author: {median_degree}")
    print('\n\n')

    G = nx.from_pandas_edgelist(dataauthedges, 'source', 'target')
    degrees = [int(deg) for node, deg in G.degree()]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum Degree dataset-author: {max_degree}")
    print(f"Minimum Degree dataset-author: {min_degree}")
    print(f"Median Degree dataset-author: {median_degree}")
    print('\n\n')

    # count nodes with degrees lower or equal to 10

    # elaboro connected components
    num_connected_components = nx.number_connected_components(combined_graph)
    print(f'The total number of connected components is: {num_connected_components}')

    connected_components = list(nx.connected_components(combined_graph))
    connected_components_sizes = [len(component) for component in connected_components]

    # Get maximum and minimum sizes
    filtered_max_size = max(connected_components_sizes)
    filtered_min_size = min(connected_components_sizes)
    filtered_total_size = sum(connected_components_sizes)

    print(f'Max connected component filtered: {filtered_max_size}')
    print(f'Min connected component filtered: {filtered_min_size}')
    print(f'Total nodes in connected components filtered: {filtered_total_size}')

    print('\n\n')
    auth_edges = pd.concat([pubauthedges,dataauthedges])
    G = nx.from_pandas_edgelist(auth_edges, 'source', 'target')
    degrees = [int(deg) for node, deg in G.degree() if node.startswith('a')]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum Degree author: {max_degree}")
    print(f"Minimum Degree author: {min_degree}")
    print(f"Median Degree author: {median_degree}")

    G = nx.from_pandas_edgelist(pubauthedges, 'source', 'target')
    degrees = [int(deg) for node, deg in G.degree() if node.startswith('p')]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum count authors per pub: {max_degree}")
    print(f"Minimum count authors per pub: {min_degree}")
    print(f"Median count authors per pub: {median_degree}")

    G = nx.from_pandas_edgelist(dataauthedges, 'source', 'target')
    degrees = [int(deg) for node, deg in G.degree() if node.startswith('d')]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum count authors per dataset: {max_degree}")
    print(f"Minimum count authors per dataset: {min_degree}")
    print(f"Median count authors per dataset: {median_degree}")

    if 'mes' not in path:
        G = nx.from_pandas_edgelist(pubvenuesedges, 'source', 'target')
        degrees = [int(deg) for node, deg in G.degree() if node.startswith('v')]
        max_degree = max(degrees)
        min_degree = min(degrees)
        median_degree = statistics.median(degrees)

        print(f"Maximum count publications per venue: {max_degree}")
        print(f"Minimum count publications per venue: {min_degree}")
        print(f"Median count publications per venue: {median_degree}")

        G = nx.from_pandas_edgelist(pubkeywedges, 'source', 'target')
        degrees = [int(deg) for node, deg in G.degree() if node.startswith('k')]
        max_degree = max(degrees)
        min_degree = min(degrees)
        median_degree = statistics.median(degrees)

        print(f"Maximum count publications per keyword: {max_degree}")
        print(f"Minimum count publications per keyword: {min_degree}")
        print(f"Median count publications per keyowrd: {median_degree}")
        degrees = [int(deg) for node, deg in G.degree() if node.startswith('p')]
        max_degree = max(degrees)
        min_degree = min(degrees)
        median_degree = statistics.median(degrees)

        print(f"Maximum count keyword per publication: {max_degree}")
        print(f"Minimum count keyword per publication: {min_degree}")
        print(f"Median count keyword per publication: {median_degree}")

        G = nx.from_pandas_edgelist(datakeywedges, 'source', 'target')
        degrees = [int(deg) for node, deg in G.degree() if node.startswith('k')]
        max_degree = max(degrees)
        min_degree = min(degrees)
        median_degree = statistics.median(degrees)

        print(f"Maximum count datasets per keyword: {max_degree}")
        print(f"Minimum count datasets per keyword: {min_degree}")
        print(f"Median count datasets per keyowrd: {median_degree}")
        degrees = [int(deg) for node, deg in G.degree() if node.startswith('d')]
        max_degree = max(degrees)
        min_degree = min(degrees)
        median_degree = statistics.median(degrees)

        print(f"Maximum count keyword per dataset: {max_degree}")
        print(f"Minimum count keyword per dataset: {min_degree}")
        print(f"Median count keyword per dataset: {median_degree}")

    G = nx.from_pandas_edgelist(pubentedges, 'source', 'target')
    degrees = [int(deg) for node, deg in G.degree() if node.startswith('dbpedia')]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum count publications per entity: {max_degree}")
    print(f"Minimum count publications per entity: {min_degree}")
    print(f"Median count publications per entity: {median_degree}")
    degrees = [int(deg) for node, deg in G.degree() if node.startswith('p')]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum count entity per publications: {max_degree}")
    print(f"Minimum count entity per publications: {min_degree}")
    print(f"Median count entity per publications: {median_degree}")

    G = nx.from_pandas_edgelist(dataentedges, 'source', 'target')
    degrees = [int(deg) for node, deg in G.degree() if node.startswith('dbpedia')]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum count datasets per entity: {max_degree}")
    print(f"Minimum count datasets per entity: {min_degree}")
    print(f"Median count datasets per entity: {median_degree}")
    degrees = [int(deg) for node, deg in G.degree() if node.startswith('d')]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum count entity per datasets: {max_degree}")
    print(f"Minimum count entity per datasets: {min_degree}")
    print(f"Median count entity per datasets: {median_degree}")

    G = nx.from_pandas_edgelist(pubtopicedges, 'source', 'target')
    degrees = [int(deg) for node, deg in G.degree() if node.startswith('t')]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum count publications per topics: {max_degree}")
    print(f"Minimum count publications per topics: {min_degree}")
    print(f"Median count publications per topics: {median_degree}")
    degrees = [int(deg) for node, deg in G.degree() if node.startswith('p')]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum count topics per publications: {max_degree}")
    print(f"Minimum count topics per publications: {min_degree}")
    print(f"Median count topics per publications: {median_degree}")

    G = nx.from_pandas_edgelist(datatopicedges, 'source', 'target')
    degrees = [int(deg) for node, deg in G.degree() if node.startswith('t')]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum count datasets per topics: {max_degree}")
    print(f"Minimum count datasets per topics: {min_degree}")
    print(f"Median count datasets per topics: {median_degree}")
    degrees = [int(deg) for node, deg in G.degree() if node.startswith('d')]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum count topics per datasets: {max_degree}")
    print(f"Minimum count topics per datasets: {min_degree}")
    print(f"Median count topics per datasets: {median_degree}")

    print('\n\n')


def get_stats_after_preprocessing(path):
    f = open('./dataset_preprocessing/results/final_graph_stats_'+args.dataset+'.txt','w')
    sys.stdout = f
    # count nodes
    if 'mes' not in path:
        publications = pd.read_csv(path + 'publications.csv')
        pub_obj = publications['id'].unique()
        print(f'THE TOTAL COUNT OF PUBLICATIONS IS {len(pub_obj)}')

    else:
        publications = pd.read_csv(path + 'publications.csv')
        pub_obj = publications['id'].unique()
        print(f'THE TOTAL COUNT OF PUBLICATIONS IS {len(pub_obj)}')

    datasets = pd.read_csv(path + 'datasets.csv')
    data_obj = datasets['id'].unique()
    print(f'THE TOTAL COUNT OF DATASETS IS {len(data_obj)}')

    authors = pd.read_csv(path + 'authors.csv')
    auth_obj = authors['id'].unique()
    print(f'THE TOTAL COUNT OF AUTHORS IS {len(auth_obj)}')
    if 'mes' not in path:
        pubpubedges = pd.read_csv(path + 'pubpubedges.csv')
        print(f'THE TOTAL COUNT OF EDGES BETWEEN PUBLICATIONS IS {pubpubedges.shape[0]}')
    else:
        pubpubedges = pd.read_csv(path + 'pubpubedges.csv')
        print(f'THE TOTAL COUNT OF EDGES BETWEEN PUBLICATIONS IS {pubpubedges.shape[0]}')

    pubdataedges = pd.read_csv(path + 'pubdataedges.csv')
    print(f'THE TOTAL COUNT OF EDGES BETWEEN PUBLICATIONS AND DATASETS IS {pubdataedges.shape[0]}')

    datadataedges = pd.read_csv(path + 'datadataedges.csv')
    print(f'THE TOTAL COUNT OF EDGES BETWEEN DATASETS IS {datadataedges.shape[0]}')

    pubauthedges = pd.read_csv(path + 'pubauthedges.csv')
    print(f'THE TOTAL COUNT OF EDGES BETWEEN PUBLICATIONS AND AUTHORS IS {pubauthedges.shape[0]}')

    dataauthedges = pd.read_csv(path + 'dataauthedges.csv')
    print(f'THE TOTAL COUNT OF EDGES BETWEEN DATASETS AND AUTHORS IS {dataauthedges.shape[0]}')
    edges_concat = pd.concat([pubpubedges, pubdataedges, pubauthedges, dataauthedges, datadataedges], ignore_index=True)

    if 'mes' not in path:
        venues = pd.read_csv(path + 'venues.csv')
        venues_obj = venues['id'].unique()
        print(f'THE TOTAL COUNT OF VENUES IS {len(venues_obj)}')

        organizations = pd.read_csv(path + 'organizations.csv')
        organizations_obj = organizations['id'].unique()
        print(f'THE TOTAL COUNT OF ORGANIZATIONS IS {len(organizations_obj)}')

        keywords = pd.read_csv(path + 'keywords.csv')
        keywords_obj = keywords['id'].unique()
        print(f'THE TOTAL COUNT OF KEYWORDS IS {len(keywords_obj)}')

        puborgedges = pd.read_csv(path + 'puborgedges.csv')
        print(f'THE TOTAL COUNT OF EDGES BETWEEN PUBLICATIONS AND ORGANIZATIONS IS {puborgedges.shape[0]}')

        dataorgedges = pd.read_csv(path + 'dataorgedges.csv')
        print(f'THE TOTAL COUNT OF EDGES BETWEEN DATASETS AND ORGANIZATIONS IS {dataorgedges.shape[0]}')

        pubvenuesedges = pd.read_csv(path + 'pubvenuesedges.csv')
        print(f'THE TOTAL COUNT OF EDGES BETWEEN PUBLICATIONS AND VENUES IS {pubvenuesedges.shape[0]}')

        pubkeywedges = pd.read_csv(path + 'pubkeyedges.csv')
        print(f'THE TOTAL COUNT OF EDGES BETWEEN PUBLICATIONS AND KEYWORDS IS {pubkeywedges.shape[0]}')

        datakeywedges = pd.read_csv(path + 'datakeyedges.csv')
        print(f'THE TOTAL COUNT OF EDGES BETWEEN DATASETS AND KEYWORDS IS {datakeywedges.shape[0]}')
        edges_concat = pd.concat([edges_concat, pubkeywedges, datakeywedges, pubvenuesedges, puborgedges, dataorgedges],
                                 ignore_index=True)

    # find max, min, median degree for each edge type

    G = nx.from_pandas_edgelist(pubpubedges, 'source', 'target')
    combined_graph = nx.from_pandas_edgelist(edges_concat, 'source', 'target')

    degrees = [int(deg) for node, deg in G.degree()]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum Degree publication-publication: {max_degree}")
    print(f"Minimum Degree publication-publication: {min_degree}")
    print(f"Median Degree publication-publication: {median_degree}")
    print('\n\n')

    G = nx.from_pandas_edgelist(pubdataedges, 'source', 'target')

    degrees = [int(deg) for node, deg in G.degree()]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum Degree publication-dataset: {max_degree}")
    print(f"Minimum Degree publication-dataset: {min_degree}")
    print(f"Median Degree publication-dataset: {median_degree}")
    print('\n\n')

    G = nx.from_pandas_edgelist(datadataedges, 'source', 'target')

    degrees = [int(deg) for node, deg in G.degree()]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum Degree dataset-dataset: {max_degree}")
    print(f"Minimum Degree dataset-dataset: {min_degree}")
    print(f"Median Degree dataset-dataset: {median_degree}")
    print('\n\n')

    G = nx.from_pandas_edgelist(pubauthedges, 'source', 'target')
    degrees = [int(deg) for node, deg in G.degree()]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum Degree publication-author: {max_degree}")
    print(f"Minimum Degree publication-author: {min_degree}")
    print(f"Median Degree publication-author: {median_degree}")
    print('\n\n')

    G = nx.from_pandas_edgelist(dataauthedges, 'source', 'target')
    degrees = [int(deg) for node, deg in G.degree()]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum Degree dataset-author: {max_degree}")
    print(f"Minimum Degree dataset-author: {min_degree}")
    print(f"Median Degree dataset-author: {median_degree}")
    print('\n\n')

    # count nodes with degrees lower or equal to 10

    # elaboro connected components
    num_connected_components = nx.number_connected_components(combined_graph)
    print(f'The total number of connected components is: {num_connected_components}')

    connected_components = list(nx.connected_components(combined_graph))
    connected_components_sizes = [len(component) for component in connected_components]

    # Get maximum and minimum sizes
    filtered_max_size = max(connected_components_sizes)
    filtered_min_size = min(connected_components_sizes)
    filtered_total_size = sum(connected_components_sizes)

    print(f'Max connected component filtered: {filtered_max_size}')
    print(f'Min connected component filtered: {filtered_min_size}')
    print(f'Total nodes in connected components filtered: {filtered_total_size}')

    print('\n\n')


def get_graph_stats(path):

    f = open('./dataset_preprocessing/results/initial_graph_stats_'+args.dataset+'.txt','w')
    sys.stdout = f  # Redirect stdout to the file
    publications_to_remove = []
    datasets_to_remove = []
    authors_to_remove = []
    nodes_to_remove = []
    # count nodes
    publications = pd.read_csv(path+'publications.csv')
    pub_obj = publications['id'].unique()
    print(f'THE TOTAL COUNT OF PUBLICATIONS IS {len(pub_obj)}')

    datasets = pd.read_csv(path+'datasets.csv')
    data_obj = datasets['id'].unique()
    print(f'THE TOTAL COUNT OF DATASETS IS {len(data_obj)}')

    authors = pd.read_csv(path+'authors.csv')
    auth_obj = authors['id'].unique()
    print(f'THE TOTAL COUNT OF AUTHORS IS {len(auth_obj)}')

    pubpubedges = pd.read_csv(path+'pubpubedges.csv')
    print(f'THE TOTAL COUNT OF EDGES BETWEEN PUBLICATIONS IS {pubpubedges.shape[0]}')

    pubdataedges = pd.read_csv(path+'pubdataedges.csv')
    print(f'THE TOTAL COUNT OF EDGES BETWEEN PUBLICATIONS AND DATASETS IS {pubdataedges.shape[0]}')

    datadataedges = pd.read_csv(path+'datadataedges.csv')
    print(f'THE TOTAL COUNT OF EDGES BETWEEN DATASETS IS {datadataedges.shape[0]}')

    pubauthedges = pd.read_csv(path+'pubauthedges.csv')
    print(f'THE TOTAL COUNT OF EDGES BETWEEN PUBLICATIONS AND AUTHORS IS {pubauthedges.shape[0]}')

    dataauthedges = pd.read_csv(path+'dataauthedges.csv')
    print(f'THE TOTAL COUNT OF EDGES BETWEEN DATASETS AND AUTHORS IS {dataauthedges.shape[0]}')
    edges_concat = pd.concat([pubpubedges,pubdataedges,pubauthedges,dataauthedges,datadataedges],ignore_index=True)

    if 'mes' not in path:
        venues = pd.read_csv(path+'venues.csv')
        venues_obj = venues['id'].unique()
        print(f'THE TOTAL COUNT OF VENUES IS {len(venues_obj)}')

        organizations = pd.read_csv(path+'organizations.csv')
        organizations_obj = organizations['id'].unique()
        print(f'THE TOTAL COUNT OF ORGANIZATIONS IS {len(organizations_obj)}')

        keywords = pd.read_csv(path+'keywords.csv')
        keywords_obj = keywords['id'].unique()
        print(f'THE TOTAL COUNT OF KEYWORDS IS {len(keywords_obj)}')

        puborgedges = pd.read_csv(path+'puborgedges.csv')
        print(f'THE TOTAL COUNT OF EDGES BETWEEN PUBLICATIONS AND ORGANIZATIONS IS {puborgedges.shape[0]}')

        dataorgedges = pd.read_csv(path+'dataorgedges.csv')
        print(f'THE TOTAL COUNT OF EDGES BETWEEN DATASETS AND ORGANIZATIONS IS {dataorgedges.shape[0]}')

        pubvenuesedges = pd.read_csv(path+'pubvenueedges.csv')
        print(f'THE TOTAL COUNT OF EDGES BETWEEN PUBLICATIONS AND VENUES IS {pubvenuesedges.shape[0]}')

        pubkeywedges = pd.read_csv(path+'pubkeyedges.csv')
        print(f'THE TOTAL COUNT OF EDGES BETWEEN PUBLICATIONS AND KEYWORDS IS {pubkeywedges.shape[0]}')

        datakeywedges = pd.read_csv(path+'datakeyedges.csv')
        print(f'THE TOTAL COUNT OF EDGES BETWEEN DATASETS AND KEYWORDS IS {datakeywedges.shape[0]}')
        edges_concat = pd.concat([edges_concat,pubkeywedges,datakeywedges,pubvenuesedges,puborgedges,dataorgedges],ignore_index=True)

    # find max, min, median degree for each edge type


    G = nx.from_pandas_edgelist(pubpubedges, 'source', 'target')
    combined_graph = nx.from_pandas_edgelist(edges_concat, 'source', 'target')

    degree_dict = G.degree()
    filtered_nodes = [node for node, degree in degree_dict if degree <= 30]
    print(f'Nodes with degree less than 30 publication-publication: {len(filtered_nodes)}')
    degrees = [degree for node, degree in degree_dict]
    vp = np.percentile(degrees, 75)
    print(f'percentile 75: {vp}')
    vp = np.percentile(degrees, 95)
    print(f'percentile 95: {vp}')
    vp = np.percentile(degrees, 85)
    print(f'percentile 85: {vp}')
    # degrees = sorted(degrees, reverse=True)
    # element_counts = Counter(degrees)
    # # Print the counts
    # for element, count in element_counts.items():
    #     print(f"{element}: {count}")


    filtered_nodes = [node for node, degree in degree_dict if degree > 30]
    nodes_to_remove.extend(filtered_nodes)

    for node in filtered_nodes:
        if node.startswith('p'):
            publications_to_remove.append(node)
        if node.startswith('a'):
            authors_to_remove.append(node)
        if node.startswith('d'):
            datasets_to_remove.append(node)
    print(f'Nodes with degree higher than 30 publication-publication: {len(filtered_nodes)}')
    degrees = [int(deg) for node, deg in G.degree()]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum Degree publication-publication: {max_degree}")
    print(f"Minimum Degree publication-publication: {min_degree}")
    print(f"Median Degree publication-publication: {median_degree}")
    print('\n\n')


    G = nx.from_pandas_edgelist(pubdataedges, 'source', 'target')
    degree_dict = G.degree()
    filtered_nodes = [node for node, degree in degree_dict if degree <= 20]
    print(f'Nodes with degree less than 20 publication-dataset: {len(filtered_nodes)}')
    filtered_nodes = [node for node, degree in degree_dict if degree > 20]
    nodes_to_remove.extend(filtered_nodes)
    for node in filtered_nodes:
        if node.startswith('p'):
            publications_to_remove.append(node)
        if node.startswith('a'):
            authors_to_remove.append(node)
        if node.startswith('d'):
            datasets_to_remove.append(node)


    print(f'Nodes with degree higher than 20 publication-dataset: {len(filtered_nodes)}')
    degrees = [int(deg) for node, deg in G.degree()]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum Degree publication-dataset: {max_degree}")
    print(f"Minimum Degree publication-dataset: {min_degree}")
    print(f"Median Degree publication-dataset: {median_degree}")
    print('\n\n')

    concat = pd.concat([pubdataedges,pubpubedges,datadataedges],ignore_index=True)
    G = nx.from_pandas_edgelist(concat, 'source', 'target')
    degree_dict = G.degree()
    degrees = [degree for node, degree in degree_dict]
    vp = np.percentile(degrees, 75)
    print(f'percentile 75: {vp}')
    vp = np.percentile(degrees, 95)
    print(f'percentile 95: {vp}')
    vp = np.percentile(degrees, 85)
    print(f'percentile 85: {vp}')




    G = nx.from_pandas_edgelist(datadataedges, 'source', 'target')
    degree_dict = G.degree()
    filtered_nodes = [node for node, degree in degree_dict if degree <= 20]
    print(f'Nodes with degree less than 20 dataset-dataset: {len(filtered_nodes)}')
    filtered_nodes = [node for node, degree in degree_dict if degree > 20]
    nodes_to_remove.extend(filtered_nodes)

    for node in filtered_nodes:
        if node.startswith('p'):
            publications_to_remove.append(node)
        if node.startswith('a'):
            authors_to_remove.append(node)
        if node.startswith('d'):
            datasets_to_remove.append(node)
    print(f'Nodes with degree higher than 20 dataset-dataset: {len(filtered_nodes)}')
    degrees = [int(deg) for node, deg in G.degree()]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum Degree dataset-dataset: {max_degree}")
    print(f"Minimum Degree dataset-dataset: {min_degree}")
    print(f"Median Degree dataset-dataset: {median_degree}")
    print('\n\n')

    G = nx.from_pandas_edgelist(pubauthedges, 'source', 'target')
    degree_dict = G.degree()
    filtered_nodes = [node for node, degree in degree_dict if degree <= 50]
    print(f'Nodes with degree less than 50 publication-author: {len(filtered_nodes)}')
    filtered_nodes = [node for node, degree in degree_dict if degree > 50]
    nodes_to_remove.extend(filtered_nodes)
    for node in filtered_nodes:
        if node.startswith('p'):
            publications_to_remove.append(node)
        if node.startswith('a'):
            authors_to_remove.append(node)
        if node.startswith('d'):
            datasets_to_remove.append(node)

    print(f'Nodes with degree higher than 50 publication-author: {len(filtered_nodes)}')
    degrees = [int(deg) for node, deg in G.degree()]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum Degree publication-author: {max_degree}")
    print(f"Minimum Degree publication-author: {min_degree}")
    print(f"Median Degree publication-author: {median_degree}")
    print('\n\n')

    G = nx.from_pandas_edgelist(dataauthedges, 'source', 'target')
    degree_dict = G.degree()
    filtered_nodes = [node for node, degree in degree_dict if degree <= 50]
    print(f'Nodes with degree less than 50 dataset-author: {len(filtered_nodes)}')
    filtered_nodes = [node for node, degree in degree_dict if degree > 50]
    nodes_to_remove.extend(filtered_nodes)

    for node in filtered_nodes:
        if node.startswith('p'):
            publications_to_remove.append(node)
        if node.startswith('a'):
            authors_to_remove.append(node)
        if node.startswith('d'):
            datasets_to_remove.append(node)
    print(f'Nodes with degree higher than 50 dataset-author: {len(filtered_nodes)}')
    degrees = [int(deg) for node, deg in G.degree()]
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = statistics.median(degrees)

    print(f"Maximum Degree dataset-author: {max_degree}")
    print(f"Minimum Degree dataset-author: {min_degree}")
    print(f"Median Degree dataset-author: {median_degree}")
    print('\n\n')

    # count nodes with degrees lower or equal to 10

    # elaboro connected components
    num_connected_components = nx.number_connected_components(combined_graph)
    print(f'The total number of connected components is: {num_connected_components}')

    connected_components = list(nx.connected_components(combined_graph))
    filtered_components = [component for component in connected_components if
                           not any(str(node).startswith("d_") for node in component)]
    filtered_component_sizes = [len(component) for component in filtered_components]
    print(f'The total number of connected components without any dataset is: {len(filtered_components)}')
    to_keep_components = [component for component in connected_components if
                            any(str(node).startswith("d_") for node in component)]
    to_keep_component_sizes = [len(component) for component in to_keep_components]
    print(f'The total number of connected components with some dataset is: {len(to_keep_components)}')
    # Get maximum and minimum sizes
    if len(filtered_component_sizes) > 0:
        filtered_max_size = max(filtered_component_sizes)
        filtered_min_size = min(filtered_component_sizes)
        filtered_total_size = sum(filtered_component_sizes)
        print(f'Max connected component filtered: {filtered_max_size}')
        print(f'Min connected component filtered: {filtered_min_size}')
        print(f'Total nodes in connected components filtered: {filtered_total_size}')
    if len(to_keep_components) > 0:
        to_keep_max_size = max(to_keep_component_sizes)
        to_keep_min_size = min(to_keep_component_sizes)
        to_keep_total_size = sum(to_keep_component_sizes)

        print(f'Max connected component to keep: {to_keep_max_size}')
        print(f'Min connected component to keep: {to_keep_min_size}')
        print(f'Total connected components to keep: {to_keep_total_size}')
    print('\n\n')

    print('nodes to remove',len(nodes_to_remove))

    # check removing nodes with degree higher than 10 how the graph changes
    nodes_to_remove = publications_to_remove + authors_to_remove + datasets_to_remove
    print('nodes to remove',len(nodes_to_remove))
    combined_graph.remove_nodes_from(nodes_to_remove)
    num_connected_components = nx.number_connected_components(combined_graph)
    print('\n\nAFTER DEG ANALYSES')
    print(f'The total number of connected components is: {num_connected_components}')

    connected_components = list(nx.connected_components(combined_graph))
    filtered_components = [component for component in connected_components if
                           not any(str(node).startswith("d_") for node in component)]
    filtered_component_sizes = [len(component) for component in filtered_components]
    print(f'The total number of connected components without any dataset is: {len(filtered_components)}')
    to_keep_components = [component for component in connected_components if
                          any(str(node).startswith("d_") for node in component)]
    to_keep_component_sizes = [len(component) for component in to_keep_components]
    print(f'The total number of connected components with some dataset is: {len(to_keep_components)}')
    # Get maximum and minimum sizes
    filtered_max_size = max(filtered_component_sizes)
    filtered_min_size = min(filtered_component_sizes)
    filtered_total_size = sum(filtered_component_sizes)
    to_keep_max_size = max(to_keep_component_sizes)
    to_keep_min_size = min(to_keep_component_sizes)
    to_keep_total_size = sum(to_keep_component_sizes)
    print(f'Max connected component filtered: {filtered_max_size}')
    print(f'Min connected component filtered: {filtered_min_size}')
    print(f'Total nodes connected components filtered: {filtered_total_size}')
    print(f'Max connected component to keep: {to_keep_max_size}')
    print(f'Min connected component to keep: {to_keep_min_size}')
    print(f'Total connected components to keep: {to_keep_total_size}')

if __name__ == '__main__':
    path_final = './datasets/'+args.dataset+'/all/final/'
    path_initial = './datasets/'+args.dataset+'/all/mapping/'
    # keywords = pd.read_csv(path_final + 'keywords.csv')
    # print(keywords.shape[0])
    # keywords = pd.read_csv(path_initial + 'keywords.csv')
    # print(keywords.shape[0])
    get_graph_stats(path_initial)
    get_stats_after_preprocessing(path_final)
    get_final_stats(args.dataset)









