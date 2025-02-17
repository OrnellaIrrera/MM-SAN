import json
import os
import numpy as np
import networkx as nx
import pandas as pd
import argparse
import statistics
import time
parser = argparse.ArgumentParser()
parser.add_argument("-dataset", default='mes',choices=['mes','pubmed','pubmed_kcore'],
                    type=str)
parser.add_argument("-type",default='light')

def get_count(path):

    print(f"Publications: {pd.read_csv(path + 'publications.csv').shape[0]}")
    print(f"Datasets: {pd.read_csv(path + 'datasets.csv').shape[0]}")
    print(f"Authors: {pd.read_csv(path + 'authors.csv').shape[0]}")
    print(f"pub-pub: {pd.read_csv(path + 'pubpubedges.csv').shape[0]}")
    print(f"pub-data: {pd.read_csv(path + 'pubdataedges.csv').shape[0]}")
    print(f"data-data: {pd.read_csv(path + 'datadataedges.csv').shape[0]}")
    print(f"pub-auth: {pd.read_csv(path + 'pubauthedges.csv').shape[0]}")
    print(f"data-auth: {pd.read_csv(path + 'dataauthedges.csv').shape[0]}")
    if os.path.exists(path + 'keywords.csv'):
        print(f"Keywords: {pd.read_csv(path + 'keywords.csv').shape[0]}")
        print(f"Organizations: {pd.read_csv(path + 'organizations.csv').shape[0]}")
        print(f"Venues: {pd.read_csv(path + 'venues.csv').shape[0]}")
        # print(f"pub-venue: {pd.read_csv(path + 'pubvenuesedges.csv').shape[0]}")
        print(f"pub-key: {pd.read_csv(path + 'pubkeyedges.csv').shape[0]}")
        # print(f"pub-org: {pd.read_csv(path + 'puborgedges.csv').shape[0]}")
        print(f"data-key: {pd.read_csv(path + 'datakeyedges.csv').shape[0]}")
        # print(f"data-org: {pd.read_csv(path + 'dataorgedges.csv').shape[0]}")





def mapping_nodes(path_dataset):
    publications = pd.read_csv(path_dataset+'original/publications.csv')
    mapping_publications = {valore.replace('"',''):'p_'+str(id) for id, valore in enumerate(publications['id'].replace('"','').tolist())}
    publications_ids = [mapping_publications[v.replace('"','')] for v in publications['id'].replace('"','').tolist()]
    publications['id'] = publications_ids
    publications.to_csv(path_dataset+'mapping/publications.csv',index=False)

    datasets = pd.read_csv(path_dataset+'original/datasets.csv',low_memory=False)
    mapping_datasets = {valore.replace('"',''):'d_'+str(id) for id, valore in enumerate(datasets['id'].replace('"','').tolist())}
    datasets_ids = [mapping_datasets[v.replace('"','')] for v in datasets['id'].replace('"','').tolist()]
    datasets['id'] = datasets_ids
    datasets.to_csv(path_dataset+'mapping/datasets.csv',index=False)


    authors = pd.read_csv(path_dataset+'original/authors.csv')
    mapping_authors = {str(valore).replace('"',''): 'a_'+str(id) for id, valore in enumerate(authors['id'].replace('"','').tolist())}
    authors_ids = [mapping_authors[str(v).replace('"','')] for v in authors['id'].replace('"','').tolist()]
    authors['id'] = authors_ids
    json_r = {}
    json_r['publications'] = mapping_publications
    json_r['datasets'] = mapping_datasets
    json_r['authors'] = mapping_authors
    authors.to_csv(path_dataset+'mapping/authors.csv',index=False)

    if 'mes' not in path_dataset:
        venues = pd.read_csv(path_dataset+'original/venues.csv')
        mapping_venues = {valore.replace('"',''): 'v_'+str(id) for id, valore in enumerate(venues['id'].replace('"','').tolist())}
        venues_ids = [mapping_venues[v.replace('"','')] for v in venues['id'].replace('"','').tolist()]
        venues['id'] = venues_ids
        venues.to_csv(path_dataset + 'mapping/venues.csv', index=False)

        organizations = pd.read_csv(path_dataset+'original/organizations.csv')
        mapping_organizations = {valore.replace('"',''):'o_'+str(id) for id, valore in enumerate(organizations['id'].replace('"','').tolist())}
        organizations_ids = [mapping_organizations[v.replace('"','')] for v in organizations['id'].replace('"','').tolist()]
        organizations['id'] = organizations_ids
        organizations.to_csv(path_dataset + 'mapping/organizations.csv', index=False)

        keywords = pd.read_csv(path_dataset+'original/keywords.csv')
        mapping_keywords = {valore.replace('"',''):'k_'+str(id) for id, valore in enumerate(keywords['id'].replace('"','').tolist())}
        keywords_ids = [mapping_keywords[v.replace('"','')] for v in keywords['id'].replace('"','').tolist()]
        keywords['id'] = keywords_ids
        keywords.to_csv(path_dataset + 'mapping/keywords.csv', index=False)
        json_r['venues'] = mapping_venues
        json_r['keywords'] = mapping_keywords
        json_r['organizations'] = mapping_organizations


    f = open(path_dataset+'mapping/mapping_nodes.json','w')
    json.dump(json_r,f,indent=4)



def mapping_edges(path_dataset):

    """Map edges"""

    mapping = json.load(open(path_dataset+'mapping/mapping_nodes.json','r'))

    pubpubedges = pd.read_csv(path_dataset + 'original/pubpubedges.csv')
    sources_ids = [mapping['publications'][v] for v in pubpubedges['source'].replace('"','').tolist()]
    targets_ids = [mapping['publications'][v] for v in pubpubedges['target'].replace('"','').tolist()]
    pubpubedges['source'] = sources_ids
    pubpubedges['target'] = targets_ids
    pubpubedges.to_csv(path_dataset + 'mapping/pubpubedges.csv', index=False)

    pubdataedges = pd.read_csv(path_dataset + 'original/pubdataedges.csv')
    sources_ids = [mapping['publications'].get(v,'') for v in pubdataedges['source'].replace('"','').tolist()]
    targets_ids = [mapping['datasets'].get(v,'') for v in pubdataedges['target'].replace('"','').tolist()]
    pubdataedges['source'] = sources_ids
    pubdataedges['target'] = targets_ids
    pubdataedges.to_csv(path_dataset + 'mapping/pubdataedges.csv', index=False)

    datadataedges = pd.read_csv(path_dataset + 'original/datadataedges.csv')
    sources_ids = [mapping['datasets'][v] for v in datadataedges['source'].replace('"','').tolist()]
    targets_ids = [mapping['datasets'][v] for v in datadataedges['target'].replace('"','').tolist()]
    datadataedges['source'] = sources_ids
    datadataedges['target'] = targets_ids
    datadataedges.to_csv(path_dataset + 'mapping/datadataedges.csv', index=False)

    pubauthedges = pd.read_csv(path_dataset + 'original/pubauthedges.csv')
    sources_ids = [mapping['publications'][v] for v in pubauthedges['source'].replace('"','').tolist()]
    targets_ids = [mapping['authors'][(str(v))] for v in pubauthedges['target'].replace('"','').tolist()]
    pubauthedges['source'] = sources_ids
    pubauthedges['target'] = targets_ids
    pubauthedges.to_csv(path_dataset + 'mapping/pubauthedges.csv', index=False)

    dataauthedges = pd.read_csv(path_dataset + 'original/dataauthedges.csv')
    sources_ids = [mapping['datasets'][v] for v in dataauthedges['source'].replace('"','').tolist()]
    targets_ids = [mapping['authors'][(str(v))] for v in dataauthedges['target'].replace('"','').tolist()]
    dataauthedges['source'] = sources_ids
    dataauthedges['target'] = targets_ids
    dataauthedges.to_csv(path_dataset + 'mapping/dataauthedges.csv', index=False)

    if 'mes' not in path_dataset:
        puborgedges = pd.read_csv(path_dataset + 'original/puborgedges.csv')
        sources_ids = [mapping['publications'].get(v,'') for v in puborgedges['source'].replace('"','').tolist()]
        targets_ids = [mapping['organizations'].get(v,'') for v in puborgedges['target'].replace('"','').tolist()]
        puborgedges['source'] = sources_ids
        puborgedges['target'] = targets_ids
        puborgedges.to_csv(path_dataset + 'mapping/puborgedges.csv', index=False)
    #
        dataorgedges = pd.read_csv(path_dataset + 'original/dataorgedges.csv')
        sources_ids = [mapping['datasets'][v] for v in dataorgedges['source'].replace('"','').tolist()]
        targets_ids = [mapping['organizations'][v] for v in dataorgedges['target'].replace('"','').tolist()]
        dataorgedges['source'] = sources_ids
        dataorgedges['target'] = targets_ids
        dataorgedges.to_csv(path_dataset + 'mapping/dataorgedges.csv', index=False)
    #
        pubvenuesedges = pd.read_csv(path_dataset + 'original/pubvenueedges.csv')
        sources_ids = [mapping['publications'][v] for v in pubvenuesedges['source'].replace('"','').tolist()]
        targets_ids = [mapping['venues'][v] for v in pubvenuesedges['target'].replace('"','').tolist()]
        pubvenuesedges['source'] = [x for x in sources_ids if x != '']
        pubvenuesedges['target'] = [x for x in targets_ids if x != '']
        pubvenuesedges.to_csv(path_dataset + 'mapping/pubvenueedges.csv', index=False)

        pubkeywedges = pd.read_csv(path_dataset + 'original/pubkeywedges.csv')
        sources_ids = [mapping['publications'][v] for v in pubkeywedges['source'].replace('"','').tolist()]
        targets_ids = [mapping['keywords'][v] for v in pubkeywedges['target'].replace('"','').tolist()]
        pubkeywedges['source'] = sources_ids
        pubkeywedges['target'] = targets_ids
        pubkeywedges.to_csv(path_dataset + 'mapping/pubkeywedges.csv', index=False)
    #
        datakeywedges = pd.read_csv(path_dataset + 'original/datakeywedges.csv')
        sources_ids = [mapping['datasets'][v] for v in datakeywedges['source'].replace('"','').tolist()]
        targets_ids = [mapping['keywords'][v] for v in datakeywedges['target'].replace('"','').tolist()]
        datakeywedges['source'] = sources_ids
        datakeywedges['target'] = targets_ids
        datakeywedges.to_csv(path_dataset + 'mapping/datakeywedges.csv', index=False)




def map_json_files(path):
    for file in os.listdir(path+'/original/'):
        if file.endswith('json'):
            json_data = json.load(open(path+'/original/'+file,'r',encoding='utf-8-sig'))

            df = pd.json_normalize(json_data)

            new_path = path+'/original/'+file.replace('.json','.csv')
            df.to_csv(new_path,index=False)


def mapping(path):

    """PHASE 0: This is the first passage to remove very long ids and mapping them to shorter strings"""

    map_json_files(path)
    mapping_nodes(path)
    mapping_edges(path)


def filter_by_percentile(path1):
    """
        PHASE 1 : FILTER BY DEGREE AND REMOVE OF CONNECTED COMPONENTS WITH NO DATASETS
        Remove nodes with degree higher than 20 and 50 between nodes of the same type or another type, NOT ALL THE GRAPH. Example: publication-publication, publication-dataset, publication-author, dataset-author, dataset-dataset
        Then exclude the connected components with no datasets.
        """

    nodes_to_remove = []
    path = path1 + '/mapping/'
    get_count(path)

    pubpubedges = pd.read_csv(path + 'pubpubedges.csv')
    pubdataedges = pd.read_csv(path + 'pubdataedges.csv')
    datadataedges = pd.read_csv(path + 'datadataedges.csv')
    pubauthedges = pd.read_csv(path + 'pubauthedges.csv')
    dataauthedges = pd.read_csv(path + 'dataauthedges.csv')
    if 'mes' not in path:
        pubkeyedges = pd.read_csv(path + 'pubkeyedges.csv')
        datakeyedges = pd.read_csv(path + 'datakeyedges.csv')
        pubvenueedges = pd.read_csv(path + 'pubvenueedges.csv')



    # remove everything under the 95th percentile considering publications--datasets
    pubdata = pd.concat([pubpubedges,pubdataedges,datadataedges],ignore_index=True)
    G = nx.from_pandas_edgelist(pubdata, 'source', 'target')
    degree_dict = G.degree()
    # compute percentile
    degrees = [degree for node, degree in degree_dict]
    percentile = np.percentile(degrees,95)

    print(f'the percentile of publications and datasets is: {percentile}')
    filtered_nodes = [node for node, degree in degree_dict if degree <= percentile]
    print(f'Nodes with degree less than {percentile} publication and datasets: {len(filtered_nodes)}')
    filtered_nodes = [node for node, degree in degree_dict if degree > percentile]
    print(f'Nodes with degree greater than {percentile} publication and datasets: {len(filtered_nodes)}')
    nodes_to_remove.extend(filtered_nodes)

    authedges = pd.concat([pubauthedges,dataauthedges],ignore_index=True)
    G = nx.from_pandas_edgelist(authedges, 'source', 'target')
    degree_dict = G.degree()
    # compute percentile
    degrees = [degree for node, degree in degree_dict]
    percentile = np.percentile(degrees,95)
    print(f'the percentile of authors is: {percentile}')
    filtered_nodes = [node for node, degree in degree_dict if degree <= percentile]
    print(f'Nodes with degree less than {percentile} author: {len(filtered_nodes)}')
    filtered_nodes = [node for node, degree in degree_dict if degree > percentile]
    print(f'Nodes with degree greater than {percentile} author: {len(filtered_nodes)}')
    nodes_to_remove.extend(filtered_nodes)

    if 'mes' not in path:
        keyedges = pd.concat([datakeyedges,pubkeyedges],ignore_index=True)
        G = nx.from_pandas_edgelist(keyedges, 'source', 'target')
        degree_dict = G.degree()
        # compute percentile
        degrees = [degree for node, degree in degree_dict if node.startswith('k')]
        percentile = np.percentile(degrees, 95)
        print(f'the percentile of keywords is: {percentile}')
        filtered_nodes = [node for node, degree in degree_dict if degree <= percentile and not node.startswith('p') and not node.startswith('d')]
        print(f'Nodes with degree less than {percentile} keyword: {len(filtered_nodes)}')
        filtered_nodes = [node for node, degree in degree_dict if degree > percentile and not node.startswith('p') and not node.startswith('d')]
        print(f'Nodes with degree more than {percentile} keyword: {len(filtered_nodes)}')

        nodes_to_remove.extend(filtered_nodes)

        G = nx.from_pandas_edgelist(pubvenueedges, 'source', 'target')
        degree_dict = G.degree()
        # compute percentile
        degrees = [degree for node, degree in degree_dict if node.startswith('v')]
        print(len(degrees))
        percentile = np.percentile(degrees, 95)
        print(f'the percentile of venues is: {percentile}')
        filtered_nodes = [node for node, degree in degree_dict if degree <= percentile and not node.startswith('p') and not node.startswith('d')]
        print(f'Nodes with degree less than {percentile} venues: {len(filtered_nodes)}')
        filtered_nodes = [node for node, degree in degree_dict if degree > percentile and not node.startswith('p') and not node.startswith('d')]
        print(f'Nodes with degree more than {percentile} venues: {len(filtered_nodes)}')
        nodes_to_remove.extend(filtered_nodes)


    edges_concat = pd.concat([pubpubedges, pubdataedges, datadataedges, pubauthedges, dataauthedges],
                             ignore_index=True)
    combined_graph = nx.from_pandas_edgelist(edges_concat, 'source', 'target')
    combined_graph.remove_nodes_from(nodes_to_remove)

    connected_components = list(nx.connected_components(combined_graph))
    filtered_components = [component for component in connected_components if
                           not any(str(node).startswith("d_") for node in component)]
    filtered_component_sizes = [len(component) for component in filtered_components]
    print(f'The total number of connected components without any dataset is: {len(filtered_components)}')
    print(f'Total nodes in connected components filtered: {sum(filtered_component_sizes)}')

    to_keep_components = [component for component in connected_components if
                          any(str(node).startswith("d_") for node in component)]
    to_keep_component_sizes = [len(component) for component in to_keep_components]
    print(f'The total number of connected components with some dataset is: {len(to_keep_components)}')
    print(f'Total nodes in connected components to keep: {sum(to_keep_component_sizes)}')

    for component in filtered_components:
        # remove components without datasets
        # Get nodes of the component and convert to a list
        component_nodes = list(component)
        nodes_to_remove.extend(component_nodes)

    nodes_to_remove = list(set(nodes_to_remove))

    print(f'The total number of nodes to remove is: {len(nodes_to_remove)}')
    print(f'publications to remove: {len([p for p in nodes_to_remove if p.startswith("p")])}')
    print(f'datasets to remove: {len([p for p in nodes_to_remove if p.startswith("d")])}')
    print(f'authors to remove: {len([p for p in nodes_to_remove if p.startswith("a")])}')
    print(f'venues to remove: {len([p for p in nodes_to_remove if p.startswith("v")])}')
    print(f'organizations to remove: {len([p for p in nodes_to_remove if p.startswith("o")])}')
    print(f'keywords to remove: {len([p for p in nodes_to_remove if p.startswith("k")])}')
    # remove nodes
    print(path1)
    new_path = path1 + '/processed/'
    publications = pd.read_csv(path + 'publications.csv')
    publications = publications[~publications['id'].isin(nodes_to_remove)]
    publications.to_csv(new_path + 'publications.csv', index=False)

    datasets = pd.read_csv(path + 'datasets.csv', low_memory=False)
    datasets = datasets[~datasets['id'].isin(nodes_to_remove)]
    datasets.to_csv(new_path + 'datasets.csv', index=False)

    authors = pd.read_csv(path + 'authors.csv')
    authors = authors[~authors['id'].isin(nodes_to_remove)]
    authors.to_csv(new_path + '/authors.csv', index=False)

    pubpubedges = pd.read_csv(path + 'pubpubedges.csv')
    pubpubedges = pubpubedges[
        ~pubpubedges['source'].isin(nodes_to_remove) & ~pubpubedges['target'].isin(nodes_to_remove)]
    pubpubedges.to_csv(new_path + 'pubpubedges.csv', index=False)

    pubdataedges = pd.read_csv(path + 'pubdataedges.csv')
    pubdataedges = pubdataedges[
        ~pubdataedges['source'].isin(nodes_to_remove) & ~pubdataedges['target'].isin(nodes_to_remove)]
    pubdataedges.to_csv(new_path + 'pubdataedges.csv', index=False)

    datadataedges = pd.read_csv(path + 'datadataedges.csv')
    datadataedges = datadataedges[
        ~datadataedges['source'].isin(nodes_to_remove) & ~datadataedges['target'].isin(nodes_to_remove)]
    datadataedges.to_csv(new_path + 'datadataedges.csv', index=False)

    pubauthedges = pd.read_csv(path + 'pubauthedges.csv')
    pubauthedges = pubauthedges[
        ~pubauthedges['source'].isin(nodes_to_remove) & ~pubauthedges['target'].isin(nodes_to_remove)]
    pubauthedges.to_csv(new_path + 'pubauthedges.csv', index=False)

    dataauthedges = pd.read_csv(path + 'dataauthedges.csv')
    dataauthedges = dataauthedges[
        ~dataauthedges['source'].isin(nodes_to_remove) & ~dataauthedges['target'].isin(nodes_to_remove)]


    dataauthedges.to_csv(new_path + 'dataauthedges.csv', index=False)

    if 'mes' not in path:
        pubkeyedges = pd.read_csv(path + 'pubkeyedges.csv')
        pubkeyedges = pubkeyedges[
            ~pubkeyedges['source'].isin(nodes_to_remove) & ~pubkeyedges['target'].isin(nodes_to_remove)]
        pubkeyedges.to_csv(new_path + 'pubkeyedges.csv', index=False)

        datakeyedges = pd.read_csv(path + 'datakeyedges.csv')
        datakeyedges = datakeyedges[
            ~datakeyedges['source'].isin(nodes_to_remove) & ~datakeyedges['target'].isin(nodes_to_remove)]
        datakeyedges.to_csv(new_path + 'datakeyedges.csv', index=False)
        pubvenueedges = pubvenueedges[
            ~pubvenueedges['source'].isin(nodes_to_remove) & ~pubvenueedges['target'].isin(nodes_to_remove)]
        pubvenueedges.to_csv(new_path + 'pubvenueedges.csv', index=False)

    get_count(new_path)



def filter_by_degree(path1):

    """
    PHASE 1 : FILTER BY DEGREE AND REMOVE OF CONNECTED COMPONENTS WITH NO DATASETS
    Remove nodes with degree higher than 20 and 50 between nodes of the same type or another type, NOT ALL THE GRAPH. Example: publication-publication, publication-dataset, publication-author, dataset-author, dataset-dataset
    Then exclude the connected components with no datasets.
    """


    nodes_to_remove = []
    path = path1+'/mapping/'

    pubpubedges = pd.read_csv(path+'pubpubedges.csv')
    pubdataedges = pd.read_csv(path+'pubdataedges.csv')
    datadataedges = pd.read_csv(path+'datadataedges.csv')
    pubauthedges = pd.read_csv(path+'pubauthedges.csv')
    dataauthedges = pd.read_csv(path+'dataauthedges.csv')
    if 'mes' not in path:
        pubkeyedges = pd.read_csv(path+'pubkeyedges.csv')
        datakeyedges = pd.read_csv(path+'datakeyedges.csv')

    G = nx.from_pandas_edgelist(pubpubedges, 'source', 'target')
    degree_dict = G.degree()
    filtered_nodes = [node for node, degree in degree_dict if degree <= 30]
    print(f'Nodes with degree less than 30 publication-publication: {len(filtered_nodes)}')
    filtered_nodes = [node for node, degree in degree_dict if degree > 30]
    nodes_to_remove.extend(filtered_nodes)


    G = nx.from_pandas_edgelist(pubdataedges, 'source', 'target')
    degree_dict = G.degree()
    filtered_nodes = [node for node, degree in degree_dict if degree <= 20]
    print(f'Nodes with degree less than 20 publication-dataset: {len(filtered_nodes)}')
    filtered_nodes = [node for node, degree in degree_dict if degree > 20]
    nodes_to_remove.extend(filtered_nodes)


    G = nx.from_pandas_edgelist(datadataedges, 'source', 'target')
    degree_dict = G.degree()
    filtered_nodes = [node for node, degree in degree_dict if degree <= 20]
    print(f'Nodes with degree less than 20 dataset-dataset: {len(filtered_nodes)}')
    filtered_nodes = [node for node, degree in degree_dict if degree > 20]
    nodes_to_remove.extend(filtered_nodes)


    G = nx.from_pandas_edgelist(pubauthedges, 'source', 'target')
    degree_dict = G.degree()
    filtered_nodes = [node for node, degree in degree_dict if degree <= 30]
    print(f'Nodes with degree less than 50 publication-author: {len(filtered_nodes)}')
    filtered_nodes = [node for node, degree in degree_dict if degree > 30]
    nodes_to_remove.extend(filtered_nodes)



    G = nx.from_pandas_edgelist(dataauthedges, 'source', 'target')
    degree_dict = G.degree()
    filtered_nodes = [node for node, degree in degree_dict if degree <= 30]
    print(f'Nodes with degree less than 50 dataset-author: {len(filtered_nodes)}')
    filtered_nodes = [node for node, degree in degree_dict if degree > 30]
    print(f'Nodes with degree greater than 50 dataset-author: {len(filtered_nodes)}')
    # degs = [degree for node, degree in degree_dict if degree > 50]
    # print(max(degs))

    nodes_to_remove.extend(filtered_nodes)

    if 'mes' not in path:
        G = nx.from_pandas_edgelist(pubkeyedges, 'source', 'target')
        degree_dict = G.degree()
        filtered_nodes = [node for node, degree in degree_dict if degree <= 50 and not node.startswith('p')]
        print(f'Nodes with degree less than 50 publication-keyword: {len(filtered_nodes)}')
        filtered_nodes = [node for node, degree in degree_dict if degree > 50 and not node.startswith('p')]
        print(f'Nodes with degree more than 50 publication-keyword: {len(filtered_nodes)}')

        nodes_to_remove.extend(filtered_nodes)



        G = nx.from_pandas_edgelist(datakeyedges, 'source', 'target')
        degree_dict = G.degree()
        filtered_nodes = [node for node, degree in degree_dict if degree <= 50 and not node.startswith('d')]
        print(f'Nodes with degree less than 50 dataset-keyword: {len(filtered_nodes)}')
        filtered_nodes = [node for node, degree in degree_dict if degree > 50 and not node.startswith('d')]
        print(f'Nodes with degree more than 50 dataset-keyword: {len(filtered_nodes)},{filtered_nodes[0]}')

        nodes_to_remove.extend(filtered_nodes)


    print(f'The total number of nodes to remove is: {len(nodes_to_remove)}')

    edges_concat = pd.concat([pubpubedges, pubdataedges, datadataedges, pubauthedges, dataauthedges],
                             ignore_index=True)
    combined_graph = nx.from_pandas_edgelist(edges_concat, 'source', 'target')
    combined_graph.remove_nodes_from(nodes_to_remove)


    connected_components = list(nx.connected_components(combined_graph))
    filtered_components = [component for component in connected_components if
                           not any(str(node).startswith("d_") for node in component)]
    filtered_component_sizes = [len(component) for component in filtered_components]
    print(f'The total number of connected components without any dataset is: {len(filtered_components)}')
    print(f'Total nodes in connected components filtered: {sum(filtered_component_sizes)}')


    to_keep_components = [component for component in connected_components if
                            any(str(node).startswith("d_") for node in component)]
    to_keep_component_sizes = [len(component) for component in to_keep_components]
    print(f'The total number of connected components with some dataset is: {len(to_keep_components)}')
    print(f'Total nodes in connected components filtered: {sum(to_keep_component_sizes)}')


    for component in filtered_components:
        # remove components without datasets
        # Get nodes of the component and convert to a list
        component_nodes = list(component)
        nodes_to_remove.extend(component_nodes)

    nodes_to_remove = list(set(nodes_to_remove))
    # remove nodes
    print(path1)
    new_path =path1+'/processed/'
    publications = pd.read_csv(path+'publications.csv')
    publications = publications[~publications['id'].isin(nodes_to_remove)]
    publications.to_csv(new_path+'publications.csv',index=False)

    datasets = pd.read_csv(path+'datasets.csv',low_memory=False)
    datasets = datasets[~datasets['id'].isin(nodes_to_remove)]
    datasets.to_csv(new_path+'datasets.csv',index=False)

    authors = pd.read_csv(path+'authors.csv')
    authors = authors[~authors['id'].isin(nodes_to_remove)]
    authors.to_csv(new_path+'/authors.csv',index=False)

    pubpubedges = pd.read_csv(path+'pubpubedges.csv')
    pubpubedges = pubpubedges[~pubpubedges['source'].isin(nodes_to_remove) & ~pubpubedges['target'].isin(nodes_to_remove)]
    pubpubedges.to_csv(new_path+'pubpubedges.csv',index=False)

    pubdataedges = pd.read_csv(path+'pubdataedges.csv')
    pubdataedges = pubdataedges[~pubdataedges['source'].isin(nodes_to_remove) & ~pubdataedges['target'].isin(nodes_to_remove)]
    pubdataedges.to_csv(new_path+'pubdataedges.csv',index=False)

    datadataedges = pd.read_csv(path+'datadataedges.csv')
    datadataedges = datadataedges[~datadataedges['source'].isin(nodes_to_remove) & ~datadataedges['target'].isin(nodes_to_remove)]
    datadataedges.to_csv(new_path+'datadataedges.csv',index=False)

    pubauthedges = pd.read_csv(path+'pubauthedges.csv')
    pubauthedges = pubauthedges[~pubauthedges['source'].isin(nodes_to_remove) & ~pubauthedges['target'].isin(nodes_to_remove)]
    pubauthedges.to_csv(new_path+'pubauthedges.csv',index=False)

    dataauthedges = pd.read_csv(path+'dataauthedges.csv')
    dataauthedges = dataauthedges[~dataauthedges['source'].isin(nodes_to_remove) & ~dataauthedges['target'].isin(nodes_to_remove)]
    G = nx.from_pandas_edgelist(dataauthedges, 'source', 'target')
    degree_dict = G.degree()
    filtered_nodes = [node for node, degree in degree_dict if degree > 50]
    print(f'Nodes with degree greater than 50 dataset-author: {len(filtered_nodes)}')


    dataauthedges.to_csv(new_path+'dataauthedges.csv',index=False)

    if 'mes' not in path:
        pubkeyedges = pd.read_csv(path+'pubkeyedges.csv')
        pubkeyedges = pubkeyedges[~pubkeyedges['source'].isin(nodes_to_remove) & ~pubkeyedges['target'].isin(nodes_to_remove)]
        pubkeyedges.to_csv(new_path+'pubkeyedges.csv',index=False)

        datakeyedges = pd.read_csv(path+'datakeyedges.csv')
        datakeyedges = datakeyedges[~datakeyedges['source'].isin(nodes_to_remove) & ~datakeyedges['target'].isin(nodes_to_remove)]
        datakeyedges.to_csv(new_path+'datakeyedges.csv',index=False)


import re

def analyze_attributes(path1):
    def process_text(text):
        # Lowercase the text and remove numbers using regex
        processed_text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        # Split the text into a list of words
        return ' '.join(processed_text.split())

    def concatenate_ids(ids):
        return ' '.join(map(str, ids))

    def count_ids(ids):
        return len(ids)

    path = path1 + '/processed/'
    path2 = path1 + '/mapping/'
    final_path = path1 + '/tmp/'
    pubpubedges = pd.read_csv(path+'pubpubedges.csv')
    pubdataedges = pd.read_csv(path+'pubdataedges.csv')
    datadataedges = pd.read_csv(path+'datadataedges.csv')
    pubauthedges = pd.read_csv(path+'pubauthedges.csv')
    dataauthedges = pd.read_csv(path+'dataauthedges.csv')
    G = nx.from_pandas_edgelist(dataauthedges, 'source', 'target')
    degree_dict = G.degree()
    # filtered_nodes = [node for node, degree in degree_dict if degree > 50]
    # print(f'Nodes with degree greater than 50 dataset-author: {len(filtered_nodes)}')

    if 'mes' not in path:
        puborgedges = pd.read_csv(path2+'puborgedges.csv')
        dataorgedges = pd.read_csv(path2+'dataorgedges.csv')
        pubvenuesedges = pd.read_csv(path2+'pubvenueedges.csv')
        pubkeyedges = pd.read_csv(path2+'pubkeyedges.csv')
        datakeyedges = pd.read_csv(path2+'datakeyedges.csv')

    # process and remove content (=abstract + title) with less than 10 words
    min_words = 20
    if 'pubmed/' in path:
        min_words = 50
    publications = pd.read_csv(path+'publications.csv')
    publications['title'] = publications['title'].apply(process_text)
    publications['description'] = publications['description'].apply(process_text)
    publications['content'] = publications['title'] + ' ' + publications['description']
    publications = publications[publications['content'].apply(lambda x: (len(x.split()) > min_words and 'occurrence download' not in x))]

    # rimuovo tutti i nodi con troppi titoli duplicati che compaiono più di 30 volte
    grouped_df = publications.groupby('title')['id'].agg([concatenate_ids, count_ids]).reset_index()
    grouped_df.rename(columns={'concatenate_ids': 'concatenated_ids', 'count_ids': 'count'}, inplace=True)
    occurrences_filtered_pubs = grouped_df.sort_values(by='count', ascending=False)
    occurrences = occurrences_filtered_pubs['count'].tolist()
    percentile = np.percentile(occurrences,95)
    print(percentile)
    occurrences_filtered_pubsg = occurrences_filtered_pubs[occurrences_filtered_pubs['count'] > percentile]
    print('occurrences_filtered_pubsg',occurrences_filtered_pubsg.shape[0])
    occurrences_filtered_pubsl = occurrences_filtered_pubs[occurrences_filtered_pubs['count'] <= percentile]
    print('occurrences_filtered_pubsl',occurrences_filtered_pubsl.shape[0])

    datasets = pd.read_csv(path + 'datasets.csv',low_memory=False)
    datasets['title'] = datasets['title'].apply(process_text)
    datasets['description'] = datasets['description'].apply(process_text)
    datasets['content'] = datasets['title'] + ' ' + datasets['description']
    datasets = datasets[datasets['content'].apply(lambda x: len(x.split()) > min_words  and 'occurrence download' not in x)]

    # rimuovo tutti i nodi con troppi titoli duplicati che compaiono più di 30 volte
    grouped_df = datasets.groupby('title')['id'].agg([concatenate_ids, count_ids]).reset_index()
    grouped_df.rename(columns={'concatenate_ids': 'concatenated_ids', 'count_ids': 'count'}, inplace=True)
    occurrences_filtered_data = grouped_df.sort_values(by='count', ascending=False)
    occurrences = occurrences_filtered_pubs['count'].tolist()
    percentile = np.percentile(occurrences,95)
    print(percentile)
    occurrences_filtered_datag = occurrences_filtered_data[occurrences_filtered_data['count'] > percentile]
    print('occurrences_filtered_datag',occurrences_filtered_datag.shape[0])
    occurrences_filtered_datal = occurrences_filtered_data[occurrences_filtered_data['count'] <= percentile]
    print('occurrences_filtered_datal',occurrences_filtered_datal.shape[0])


def process_attributes(path1):

    """Merge nodes whose description is the same*.
     First, filter the descriptions with more than 10 words and then, base the exact match on that
     Then, merge duplicated contents"""

    def process_text(text):
        # Lowercase the text and remove numbers using regex
        processed_text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        # Split the text into a list of words
        return ' '.join(processed_text.split())

    def concatenate_ids(ids):
        return ' '.join(map(str, ids))

    def count_ids(ids):
        return len(ids)

    path = path1 + '/processed/'
    get_count(path)
    path2 = path1 + '/mapping/'
    final_path = path1 + '/tmp/'
    pubpubedges = pd.read_csv(path+'pubpubedges.csv')
    pubdataedges = pd.read_csv(path+'pubdataedges.csv')
    datadataedges = pd.read_csv(path+'datadataedges.csv')
    pubauthedges = pd.read_csv(path+'pubauthedges.csv')
    dataauthedges = pd.read_csv(path+'dataauthedges.csv')
    # G = nx.from_pandas_edgelist(dataauthedges, 'source', 'target')
    # degree_dict = G.degree()
    # filtered_nodes = [node for node, degree in degree_dict if degree > 50]
    # print(f'Nodes with degree greater than 50 dataset-author: {len(filtered_nodes)}')

    if 'mes' not in path:
        puborgedges = pd.read_csv(path2+'puborgedges.csv')
        dataorgedges = pd.read_csv(path2+'dataorgedges.csv')
        pubvenuesedges = pd.read_csv(path2+'pubvenueedges.csv')
        pubkeyedges = pd.read_csv(path2+'pubkeyedges.csv')
        datakeyedges = pd.read_csv(path2+'datakeyedges.csv')

    # process and remove content (=abstract + title) with less than 10 words
    min_words = 20
    if 'pubmed/' in path:
        min_words = 50
    authors = pd.read_csv(path+'authors.csv')
    publications = pd.read_csv(path+'publications.csv')
    publications['title'] = publications['title'].apply(process_text)
    publications['description'] = publications['description'].apply(process_text)
    publications['content'] = publications['title'] + ' ' + publications['description']
    publications = publications[publications['content'].apply(lambda x: (len(x.split()) > min_words and 'occurrence download' not in x))]

    # rimuovo tutti i nodi con troppi titoli duplicati che compaiono più di 30 volte
    grouped_df = publications.groupby('title')['id'].agg([concatenate_ids, count_ids]).reset_index()
    grouped_df.rename(columns={'concatenate_ids': 'concatenated_ids', 'count_ids': 'count'}, inplace=True)
    occurrences_filtered_pubs = grouped_df.sort_values(by='count', ascending=False)
    occurrences = occurrences_filtered_pubs['count'].tolist()
    percentile = np.percentile(occurrences,95)
    occurrences_filtered_pubs = occurrences_filtered_pubs[occurrences_filtered_pubs['count'] > percentile]
    data_to_del = occurrences_filtered_pubs['concatenated_ids'].tolist()
    data_to_del = [elem for a in data_to_del for elem in a.split()]
    print(f'more than {percentile} publications: {len(list(set(data_to_del)))}')
    publications = publications[~publications['id'].isin(data_to_del)]
    # publications.to_csv(path + 'publications_processed.csv',index=False)

    datasets = pd.read_csv(path + 'datasets.csv',low_memory=False)
    datasets['title'] = datasets['title'].apply(process_text)
    datasets['description'] = datasets['description'].apply(process_text)
    datasets['content'] = datasets['title'] + ' ' + datasets['description']
    print(datasets.shape[0])

    datasets = datasets[datasets['content'].apply(lambda x: len(x.split()) > min_words  and 'occurrence download' not in x)]
    print(datasets.shape[0])

    # rimuovo tutti i nodi con troppi titoli duplicati che compaiono più di 30 volte
    grouped_df = datasets.groupby('title')['id'].agg([concatenate_ids, count_ids]).reset_index()
    grouped_df.rename(columns={'concatenate_ids': 'concatenated_ids', 'count_ids': 'count'}, inplace=True)
    occurrences_filtered_data = grouped_df.sort_values(by='count', ascending=False)
    occurrences = occurrences_filtered_data['count'].tolist()
    percentile = np.percentile(occurrences,95)
    occurrences_filtered_data = occurrences_filtered_data[occurrences_filtered_data['count'] > percentile]
    data_to_del = occurrences_filtered_data['concatenated_ids'].tolist()
    data_to_del = [elem for a in data_to_del for elem in a.split()]

    print(f'more than {percentile} datasets: {len(list(set(data_to_del)))}')
    print(datasets.shape[0])

    datasets = datasets[~datasets['id'].isin(data_to_del)]
    print(datasets.shape[0])
    # datasets.to_csv(path + '/datasets_processed.csv', index=False)



    # publications = pd.read_csv(path + 'publications_processed.csv')
    # print('grouping')
    # grouped_df = publications.groupby('content')['id'].agg([concatenate_ids, count_ids]).reset_index()
    # grouped_df.rename(columns={'concatenate_ids': 'concatenated_ids', 'count_ids': 'count'}, inplace=True)
    # occurrences_filtered_pubs = grouped_df.sort_values(by='count', ascending=False)
    # occurrences_filtered_pubs = occurrences_filtered_pubs[occurrences_filtered_pubs['count'] > 2]
    # pubs_to_merge = occurrences_filtered_pubs['concatenated_ids'].tolist()
    # pubs_to_merge = [a.split() for a in pubs_to_merge]
    # print(len(pubs_to_merge))
    # print(pubs_to_merge[0:10])
    #
    # pubs_to_merge = sorted(pubs_to_merge, key=len)
    # for p in pubs_to_merge[0:10]:
    #     print(p)
    # pubs_to_merge = []
    # for p in pubs_to_merge:
    #     if pubs_to_merge.index(p) % 10 == 0:
    #         print(pubs_to_merge.index(p))
    #     target = p[0]
    #     to_remove = p[1:]
    #     publications = publications[~publications['id'].isin(to_remove)]
    #     pubpubedges = pubpubedges.replace({col: {val: target for val in to_remove} for col in ['source', 'target']})
    #     pubdataedges = pubdataedges.replace({col: {val: target for val in to_remove} for col in ['source', 'target']})
    #     pubauthedges = pubauthedges.replace({col: {val: target for val in to_remove} for col in ['source', 'target']})
    #     dataauthedges = dataauthedges.replace({col: {val: target for val in to_remove} for col in ['source', 'target']})
    #     datadataedges = datadataedges.replace({col: {val: target for val in to_remove} for col in ['source', 'target']})
    #     if 'mes' not in path:
    #         puborgedges = puborgedges.replace({col: {val: target for val in to_remove} for col in ['source', 'target']})
    #         pubvenuesedges = pubvenuesedges.replace({col: {val: target for val in to_remove} for col in ['source', 'target']})
    #         pubkeyedges = pubkeyedges.replace({col: {val: target for val in to_remove} for col in ['source', 'target']})
    #
    # print('grouping')

    # datasets = pd.read_csv(path + 'datasets_processed.csv',low_memory=False)
    # grouped_df = datasets.groupby('content')['id'].agg([concatenate_ids, count_ids]).reset_index()
    # grouped_df.rename(columns={'concatenate_ids': 'concatenated_ids', 'count_ids': 'count'}, inplace=True)
    # occurrences_filtered_data = grouped_df.sort_values(by='count', ascending=False)
    # occurrences_filtered_data = occurrences_filtered_data[occurrences_filtered_data['count'] > 2]
    # datasets_to_merge = occurrences_filtered_data['concatenated_ids'].tolist()
    # data_to_merge = [a.split() for a in datasets_to_merge]
    # print(len(data_to_merge))
    # data_to_merge = sorted(data_to_merge, key=len,reverse=True)
    # data_to_merge = []
    # for p in data_to_merge:
    #     if data_to_merge.index(p) % 10 == 0:
    #         print(data_to_merge.index(p))
    #     target = p[0]
    #     to_remove = p[1:]
    #     datasets = datasets[~datasets['id'].isin(to_remove)]
    #     pubpubedges = pubpubedges.replace({col: {val: target for val in to_remove} for col in ['source', 'target']})
    #     pubdataedges = pubdataedges.replace({col: {val: target for val in to_remove} for col in ['source', 'target']})
    #     pubauthedges = pubauthedges.replace({col: {val: target for val in to_remove} for col in ['source', 'target']})
    #     dataauthedges = dataauthedges.replace({col: {val: target for val in to_remove} for col in ['source', 'target']})
    #     datadataedges = datadataedges.replace({col: {val: target for val in to_remove} for col in ['source', 'target']})
    #     if 'mes' not in path:
    #         dataorgedges = dataorgedges.replace({col: {val: target for val in to_remove} for col in ['source', 'target']})
    #         datakeyedges = datakeyedges.replace({col: {val: target for val in to_remove} for col in ['source', 'target']})
    #

    publications.to_csv(final_path+'publications.csv',index=False)
    datasets.to_csv(final_path+'datasets.csv',index=False)
    authors.to_csv(final_path+'authors.csv',index=False)

    publications = publications['id'].unique().tolist()
    datasets = datasets['id'].unique().tolist()

    pubpubedges = pubpubedges[pubpubedges['source'].isin(publications) & pubpubedges['target'].isin(publications)]
    pubdataedges = pubdataedges[pubdataedges['source'].isin(publications) & pubdataedges['target'].isin(datasets)]
    datadataedges = datadataedges[datadataedges['source'].isin(datasets) & datadataedges['target'].isin(datasets)]
    pubauthedges = pubauthedges[pubauthedges['source'].isin(publications)]
    dataauthedges = dataauthedges[dataauthedges['source'].isin(datasets)]
    if 'mes' not in path:
        datakeyedges = datakeyedges[datakeyedges['source'].isin(datasets)]
        dataorgedges = dataorgedges[dataorgedges['source'].isin(datasets)]
        puborgedges = puborgedges[puborgedges['source'].isin(publications)]
        pubkeyedges = pubkeyedges[pubkeyedges['source'].isin(publications)]
        pubvenuesdges = pubvenuesedges[pubvenuesedges['source'].isin(publications)]

    pubpubedges.drop_duplicates().to_csv(final_path+'pubpubedges.csv',index=False)
    pubdataedges.drop_duplicates().to_csv(final_path+'pubdataedges.csv',index=False)
    datadataedges.drop_duplicates().to_csv(final_path+'datadataedges.csv',index=False)
    pubauthedges.drop_duplicates().to_csv(final_path+'pubauthedges.csv',index=False)
    dataauthedges.drop_duplicates().to_csv(final_path+'dataauthedges.csv',index=False)
    if 'mes' not in path:
        datakeyedges.drop_duplicates().to_csv(final_path+'datakeyedges.csv',index=False)
        dataorgedges.drop_duplicates().to_csv(final_path+'dataorgedges.csv',index=False)
        puborgedges.drop_duplicates().to_csv(final_path+'puborgedges.csv',index=False)
        pubkeyedges.drop_duplicates().to_csv(final_path+'pubkeyedges.csv',index=False)
        pubvenuesdges.drop_duplicates().to_csv(final_path+'pubvenuesedges.csv',index=False)
    get_count(final_path)


def clean_all(path1):

    """Merge nodes whose description is the same*.
     First, filter the descriptions with more than 10 words and then, base the exact match on that
     Then, merge duplicated contents"""


    path = path1 + '/tmp/'
    path_map = path1 + '/mapping/'
    pubpubedges = pd.read_csv(path+'pubpubedges.csv')
    pubdataedges = pd.read_csv(path+'pubdataedges.csv')
    datadataedges = pd.read_csv(path+'datadataedges.csv')
    pubauthedges = pd.read_csv(path+'pubauthedges.csv')
    dataauthedges = pd.read_csv(path+'dataauthedges.csv')

    if 'mes' not in path:
        puborgedges = pd.read_csv(path+'puborgedges.csv')
        dataorgedges = pd.read_csv(path+'dataorgedges.csv')
        pubvenuesedges = pd.read_csv(path+'pubvenuesedges.csv')
        pubkeyedges = pd.read_csv(path+'pubkeyedges.csv')
        datakeyedges = pd.read_csv(path+'datakeyedges.csv')


    publications = pd.read_csv(path+'publications.csv')
    datasets = pd.read_csv(path+'datasets.csv')
    publications = publications['id'].unique().tolist()
    datasets = datasets['id'].unique().tolist()

    pubpubedges = pubpubedges[pubpubedges['source'].isin(publications) & pubpubedges['target'].isin(publications)]
    pubdataedges = pubdataedges[pubdataedges['source'].isin(publications) & pubdataedges['target'].isin(datasets)]
    datadataedges = datadataedges[datadataedges['source'].isin(datasets) & datadataedges['target'].isin(datasets)]
    pubauthedges = pubauthedges[pubauthedges['source'].isin(publications)]
    dataauthedges = dataauthedges[dataauthedges['source'].isin(datasets)]
    auth_0 = dataauthedges['target'].unique().tolist() + pubauthedges['target'].unique().tolist()
    if 'mes' not in path:
        datakeyedges = datakeyedges[datakeyedges['source'].isin(datasets)]
        keys_0 = datakeyedges['target'].unique().tolist()
        dataorgedges = dataorgedges[dataorgedges['source'].isin(datasets)]
        org_0 = dataorgedges['target'].unique().tolist()
        puborgedges = puborgedges[puborgedges['source'].isin(publications)]
        org_0 += puborgedges['target'].unique().tolist()
        pubkeyedges = pubkeyedges[pubkeyedges['source'].isin(publications)]
        keys_0 += pubkeyedges['target'].unique().tolist()
        pubvenuesdges = pubvenuesedges[pubvenuesedges['source'].isin(publications)]
        ven_0 = pubvenuesdges['target'].unique().tolist()

    pubpubedges.drop_duplicates().to_csv(path+'pubpubedges.csv',index=False)
    pubdataedges.drop_duplicates().to_csv(path+'pubdataedges.csv',index=False)
    datadataedges.drop_duplicates().to_csv(path+'datadataedges.csv',index=False)
    pubauthedges.drop_duplicates().to_csv(path+'pubauthedges.csv',index=False)
    dataauthedges.drop_duplicates().to_csv(path+'dataauthedges.csv',index=False)
    authors = pd.read_csv(path_map + 'authors.csv')
    authors = authors[authors['id'].isin(auth_0)]
    authors = authors[['id', 'fullname']]
    authors.to_csv(path + 'authors.csv', index=False)

    if 'mes' not in path:
        organizations = pd.read_csv(path_map+'organizations.csv')
        organizations = organizations[organizations['id'].isin(org_0)]
        organizations = organizations[['id','name']]
        organizations.to_csv(path+'organizations.csv',index=False)

        venues = pd.read_csv(path_map+'venues.csv')
        venues = venues[venues['id'].isin(ven_0)]
        venues.rename(columns={'venue': 'name'}, inplace=True)
        venues = venues[['id','name']]
        venues.to_csv(path+'venues.csv',index=False)

        keywords = pd.read_csv(path_map+'keywords.csv')
        keywords = keywords[keywords['id'].isin(keys_0)]
        keywords = keywords[['id','name']]
        keywords.to_csv(path+'keywords.csv',index=False)

        datakeyedges.drop_duplicates().to_csv(path+'datakeyedges.csv',index=False)
        dataorgedges.drop_duplicates().to_csv(path+'dataorgedges.csv',index=False)
        puborgedges.drop_duplicates().to_csv(path+'puborgedges.csv',index=False)
        pubkeyedges.drop_duplicates().to_csv(path+'pubkeyedges.csv',index=False)
        pubvenuesdges.drop_duplicates().to_csv(path+'pubvenuesedges.csv',index=False)
    get_count(path)

def final_check(path1):
    """Check the correctess of the dataset"""

    path = path1 + '/tmp/'
    path2 = path1 + '/mapping/'
    authors = pd.read_csv(path2+'authors.csv')
    if 'mes' not in path1:
        keywords = pd.read_csv(path2+'keywords.csv')
        keywords = keywords['id'].unique().tolist()

        venues = pd.read_csv(path2+'venues.csv')
        venues = venues['id'].unique().tolist()

        organizations = pd.read_csv(path2+'organizations.csv')
        organizations = organizations['id'].unique().tolist()

    publications = pd.read_csv(path+'publications.csv')
    datasets = pd.read_csv(path+'datasets.csv',low_memory=False)
    publications = publications['id'].unique().tolist()
    datasets = datasets['id'].unique().tolist()
    authors = authors['id'].unique().tolist()

    pubpubedges = pd.read_csv(path+'pubpubedges.csv')
    pubpubedges_f = pubpubedges[pubpubedges['source'].isin(publications) & pubpubedges['target'].isin(publications)]
    if pubpubedges_f.shape[0] != pubpubedges.shape[0]:
        print('PUBPUBEDGES WRONG',pubpubedges_f.shape[0],pubpubedges.shape[0])

    pubdataedges = pd.read_csv(path+'pubdataedges.csv')
    pubpubedges = pd.read_csv(path + 'pubpubedges.csv')
    unique_pp = pubpubedges['target'].unique().tolist() + pubpubedges['source'].unique().tolist()

    # publications = pd.read_csv(path+'publications.csv')
    # unique_p = pubdataedges['source'].unique().tolist()
    # notind = publications[~publications['id'].isin(unique_p)]
    # inp = notind[notind['id'].isin(unique_pp)]
    # inp1 = pubpubedges[~pubpubedges['source'].isin(unique_p) & ~pubpubedges['target'].isin(unique_p)]
    # publications__q = publications[~publications['id'].isin(unique_p+unique_pp)]
    # print(len(unique_p),inp1.shape[0],notind.shape[0],inp.shape[0],publications.shape[0],publications__q.shape[0])

    pubdataedges_f = pubdataedges[pubdataedges['source'].isin(publications) & pubdataedges['target'].isin(datasets)]

    if pubdataedges_f.shape[0] != pubdataedges.shape[0]:
        print('PUBDATEDGES WRONG',pubdataedges_f.shape[0],pubdataedges.shape[0])
    #
    datadataedges = pd.read_csv(path+'datadataedges.csv')
    datadataedges_f = datadataedges[datadataedges['source'].isin(datasets) & datadataedges['target'].isin(datasets)]
    if datadataedges_f.shape[0] != datadataedges.shape[0]:
        print('datdatedges WRONG')
    #
    pubauthedges = pd.read_csv(path+'pubauthedges.csv')
    pubauthedges_f = pubauthedges[pubauthedges['source'].isin(publications) & pubauthedges['target'].isin(authors)]
    if pubauthedges_f.shape[0] != pubauthedges.shape[0]:
        print('pubauthedges WRONG')
    #
    dataauthedges = pd.read_csv(path+'dataauthedges.csv')
    dataauthedges_f = dataauthedges[dataauthedges['source'].isin(datasets) & dataauthedges['target'].isin(authors)]
    if dataauthedges_f.shape[0] != dataauthedges.shape[0]:
        print('dataauthedges WRONG')
    #
    if 'mes' not in path:
        puborgedges = pd.read_csv(path+'puborgedges.csv')
        puborgedges_f = puborgedges[puborgedges['source'].isin(publications) & puborgedges['target'].isin(organizations)]
        if puborgedges_f.shape[0] != puborgedges.shape[0]:
            print('puborgedges WRONG')
    #
        dataorgedges = pd.read_csv(path+'dataorgedges.csv')
        dataorgedges_f = dataorgedges[dataorgedges['source'].isin(datasets) & dataorgedges['target'].isin(organizations)]
        if dataorgedges_f.shape[0] != dataorgedges.shape[0]:
            print('dataorgedges WRONG')
    #
        pubvenuesedges = pd.read_csv(path+'pubvenuesedges.csv')
        pubvenuesedges_f = pubvenuesedges[pubvenuesedges['source'].isin(publications) & pubvenuesedges['target'].isin(venues)]
        if pubvenuesedges_f.shape[0] != pubvenuesedges.shape[0]:
            print('pubvenuesedges WRONG')

        pubkeyedges = pd.read_csv(path+'pubkeyedges.csv')
        pubkeyedges_f = pubkeyedges[pubkeyedges['source'].isin(publications) & pubkeyedges['target'].isin(keywords)]
        if pubkeyedges_f.shape[0] != pubkeyedges.shape[0]:
            print('pubkeyedges WRONG')
    #
        datakeyedges = pd.read_csv(path+'datakeyedges.csv')
        datakeyedges_f = datakeyedges[datakeyedges['source'].isin(datasets) & datakeyedges['target'].isin(keywords)]
        if datakeyedges_f.shape[0] != datakeyedges.shape[0]:
            print('datakeyedges WRONG')

def preprocess_mes(path1):

    """Filter publications citation network leaving only publications connected to datasets."""

    path = path1 + '/tmp/'
    publications = pd.read_csv(path+'publications.csv')
    publications_connected_to_data = pd.read_csv(path+'pubdataedges.csv')['source'].unique().tolist()
    publications = publications[publications['id'].isin(publications_connected_to_data)]
    pubpub = pd.read_csv(path+'pubpubedges.csv')
    pubpub = pubpub[pubpub['source'].isin(publications_connected_to_data) & pubpub['target'].isin(publications_connected_to_data)]
    print('publications',publications.shape[0])
    print('pubpub',pubpub.shape[0])
    pubpub.to_csv(path+'pubpubedges.csv',index=False)
    publications.to_csv(path+'publications.csv',index=False)
    get_count(path)

def clean_keywords(dataset):
    """ This method finds the degree max of an entity"""


    pubs = pd.read_csv(f'./datasets/{dataset}/all/final/old/pubkeyedges_old.csv')
    keys = pd.read_csv(f'./datasets/{dataset}/all/final/old/keywords_old.csv')

    dats = pd.read_csv(f'./datasets/{dataset}/all/final/old/datakeyedges_old.csv')
    print(pubs.shape[0])
    print(dats.shape[0])
    print(keys.shape[0])


    edges = pd.concat([pubs,dats],ignore_index=True)

    sources = edges.groupby('source').size().reset_index(name='Count')

    sources = sources['Count'].tolist()
    targets = edges.groupby('target').size().reset_index(name='Count')
    counts = targets['Count']
    percentile = np.percentile(counts,95)
    print(f'keyword percentile: {percentile}')
    e = targets[(targets['Count'] < percentile) & (targets['Count'] > 1)]['target'].tolist()
    pubs = pubs[pubs['target'].isin(e)]
    dats = dats[dats['target'].isin(e)]
    keys = keys[keys['id'].isin(e)]
    targets = targets['Count'].tolist()

    print(statistics.median(sources),max(sources),min(sources))
    print(statistics.median(targets),max(targets),min(targets))
    print(pubs.shape[0])
    print(dats.shape[0])
    print(keys.shape[0])

    pubs.to_csv(f'./datasets/{dataset}/all/tmp/pubkeyedges.csv',index=False)
    dats.to_csv(f'./datasets/{dataset}/all/tmp/datakeyedges.csv',index=False)
    keys.to_csv(f'./datasets/{dataset}/all/tmp/keywords.csv',index=False)


    # df_rels_pubs = pubs
    # df_rels_data = dats
    # pubauthedges = pd.read_csv(f'./datasets/{dataset}/all/final/pubauthedges.csv')
    # dataauthedges = pd.read_csv(f'./datasets/{dataset}/all/final/dataauthedges.csv')
    #
    # df_rels_pubs.rename(columns={'target': 'target1', 'source': 'source1'}, inplace=True)
    # df_rels_data.rename(columns={'target': 'target1', 'source': 'source1'}, inplace=True)

    # if dataset != 'mes':
    #     pubvenedges = pd.read_csv(f'./datasets/{dataset}/all/final/pubvenuesedges.csv')
    #     pubvenueentedges = pd.merge(pubvenedges, df_rels_pubs, left_on='source', right_on='source1', how='outer')
    #     pubvenueentedges = pubvenueentedges[['target', 'target1']]
    #     pubvenueentedges.rename(columns={'target': 'source', 'target1': 'target'}, inplace=True)
    #
    #     pubvenueentedges.dropna().to_csv(f'./datasets/{dataset}/all/final/venuekeyedges.csv',index=False)
    #     print(pubvenueentedges.shape)
    #
    # print('inner1')
    # st = time.time()
    # pubauthedgesentities = pd.merge(pubauthedges, df_rels_pubs, left_on='source', right_on='source1', how='outer')
    # pubauthedgesentities = pubauthedgesentities[['target','target1']]
    # pubauthedgesentities.rename(columns={'target': 'source', 'target1': 'target'}, inplace=True)
    # pubauthedgesentities.dropna().to_csv(f'./datasets/{dataset}/all/final/pubauthkeyedges.csv',index=False)
    # print(str(time.time()-st))
    # print(pubauthedgesentities.shape)
    # print('inner2')
    # st = time.time()
    # dataauthedgesentities = pd.merge(dataauthedges, df_rels_data, left_on='source', right_on='source1', how='outer')
    # dataauthedgesentities = dataauthedgesentities[['target', 'target1']]
    # dataauthedgesentities.rename(columns={'target': 'source', 'target1': 'target'}, inplace=True)
    # dataauthedgesentities.dropna().to_csv(f'./datasets/{dataset}/all/final/dataauthkeyedges.csv',index=False)
    # print(str(time.time()-st))
    # print(dataauthedgesentities.shape)


def remove_extra_data_and_pubs(path1):
    # path = path1 + '/final/'
    path2 = path1 + '/tmp/'
    path = path2
    if not os.path.exists(path2):
        os.makedirs(path2)
    pubpub = pd.read_csv(path+'pubpubedges.csv')['source'].unique().tolist() + pd.read_csv(path+'pubpubedges.csv')['target'].unique().tolist()
    datdat = pd.read_csv(path+'datadataedges.csv')['source'].unique().tolist() + pd.read_csv(path+'datadataedges.csv')['target'].unique().tolist()
    pubdat = pd.read_csv(path+'pubdataedges.csv')

    pubs_to_keep = list(set(pubpub + pubdat['source'].unique().tolist()))
    data_to_keep = list(set(datdat + pubdat['target'].unique().tolist()))

    publications = pd.read_csv(path+'publications.csv')
    publications = publications[publications['id'].isin(pubs_to_keep)]
    print(publications.shape[0],len(pubs_to_keep))

    datasets = pd.read_csv(path+'datasets.csv')
    datasets = datasets[datasets['id'].isin(data_to_keep)]
    print(datasets.shape[0],len(data_to_keep))

    nodes = pubs_to_keep + data_to_keep
    publications.to_csv(path2 + '/publications.csv', index=False)
    datasets.to_csv(path2 + '/datasets.csv', index=False)

    for file in os.listdir(path):
        print(file)
        if file.endswith('pubpubedges.csv') or file.endswith('pubdatedges.csv') or file.endswith('datadataedges.csv'):
            print('both')
            csv = pd.read_csv(path+'/'+file)
            csv = csv[csv['source'].isin(nodes) & csv['target'].isin(nodes)]
            csv.to_csv(path2+'/'+file,index=False)
            print(csv.shape[0])
        elif file.endswith('edges.csv'):
            print('normal')
            csv = pd.read_csv(path + '/' + file)
            csv = csv[csv['source'].isin(nodes)]
            csv.to_csv(path2+'/'+file,index=False)
            print(csv.shape[0])


    # authors.to_csv(path2 + '/authors.csv', index=False)
    if 'mes' not in path:
        orgs = pd.read_csv(path + 'organizations.csv')
        venues = pd.read_csv(path + 'venues.csv')
        orgs.to_csv(path2 + 'organizations.csv', index=False)
        venues.to_csv(path2 + 'venues.csv', index=False)

    get_count(path)



def compare_final_tmp(dataset):
    final_folder = f'./datasets/{dataset}/all/final'
    tmp_folder = f'./datasets/{dataset}/all/tmp'

    for file in os.listdir(tmp_folder):
        fin = pd.read_csv(final_folder+'/'+file)
        tmp = pd.read_csv(tmp_folder+'/'+file)
        print(file,fin.shape[0],tmp.shape[0])



if __name__ == '__main__':
    args = parser.parse_args()

    dataset = args.dataset
    print('dataset',dataset)

    path = f'./datasets/{dataset}/all'

    # mapping(path)
    # filter_by_degree(path)
    filter_by_percentile(path)
    analyze_attributes(path)
    process_attributes(path)
    if 'mes' not in path:
        clean_keywords(dataset)
    # if 'mes' in path and args.type == 'light':
    #     preprocess_mes(path)
    # #
    clean_all(path)
    remove_extra_data_and_pubs(path)
    final_check(path)
    compare_final_tmp(dataset)
