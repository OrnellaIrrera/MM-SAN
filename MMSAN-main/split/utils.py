import networkx as nx
import pandas as pd
def create_graph_csv(set,dataset,nodes):

    """Create graph with specific nodes"""

    path_final = f'./datasets/{dataset}/{set}'
    path_topics = path_final
    path_entities = path_final
    edges = []
    if 'publications' in nodes or 'all' in nodes or 'original' in nodes:
        if dataset == 'mes':
            pubpubedges = pd.read_csv(path_final+'/pubpubedges.csv')
        else:
            pubpubedges = pd.read_csv(path_final+'/pubpubedges.csv')
        pubdataedges = pd.read_csv(path_final + '/pubdataedges.csv')
        edges.append(pubpubedges)
        edges.append(pubdataedges)

    if ('venues' in nodes or 'all' in nodes or 'original' in nodes) and dataset != 'mes':
        pubvenuesedges = pd.read_csv(path_final+'/pubvenuesedges.csv')

        edges.append(pubvenuesedges)


    if 'datasets' in nodes or'all' in nodes or 'original' in nodes:
        datadataedges = pd.read_csv(path_final+'/datadataedges.csv')
        edges.append(datadataedges)

    if 'authors' in nodes or'all' in nodes or 'original' in nodes:
        pubauthedges = pd.read_csv(path_final+'/pubauthedges.csv')
        dataauthedges = pd.read_csv(path_final+'/dataauthedges.csv')
        edges.append(dataauthedges)
        edges.append(pubauthedges)


    if ('organizations' in nodes or 'all' in nodes or 'original' in nodes) and dataset != 'mes':
        puborgedges = pd.read_csv(path_final + '/puborgedges.csv')
        dataorgedges = pd.read_csv(path_final + '/dataorgedges.csv')
        edges.append(dataorgedges)
        edges.append(puborgedges)

    if ('keywords' in nodes or 'all' in nodes or 'original' in nodes) and dataset != 'mes':
        pubkeyedges = pd.read_csv(path_final + '/pubkeyedges.csv')
        datakeyedges = pd.read_csv(path_final + '/datakeyedges.csv')

        edges.append(pubkeyedges)
        edges.append(datakeyedges)


    if 'entities' in nodes or 'all' in nodes:
        pubentedges = pd.read_csv(path_entities + '/pubentedges.csv')
        dataentedges = pd.read_csv(path_entities + '/dataentedges.csv')
        edges.append(dataentedges)
        edges.append(pubentedges)


    if 'topics' in nodes or 'all' in nodes:
        pubtopicedges = pd.read_csv(path_entities + '/pubtopicedges_keywords_2.csv')
        datatopicedges = pd.read_csv(path_entities + '/datatopicedges_keywords_2.csv')

        edges.append(pubtopicedges)
        edges.append(datatopicedges)



    edges_concat = pd.concat(edges,
                             ignore_index=True)
    G = nx.from_pandas_edgelist(edges_concat, 'source', 'target')

    return G








