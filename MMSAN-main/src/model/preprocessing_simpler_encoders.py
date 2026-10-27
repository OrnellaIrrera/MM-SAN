import re
import numpy as np
import pandas as pd
import pickle
import networkx as nx
import args_list
import utils
import tqdm
import gensim.downloader as gensim_api
from node2vec import Node2Vec
from gensim.models import KeyedVectors
import os


def simple_tokenize(text):
    """Tokenizzatore minimale: lowercase + split su non-alfanumerici."""
    if not isinstance(text, str):
        return []
    return re.findall(r"\w+", text.lower())


class Preprocessor:
    def __init__(self, args):
        self.dataset = args.dataset
        self.path = f'datasets/{args.dataset}/'
        # Nome del modello GloVe pre-addestrato da scaricare tramite gensim.downloader
        # Esempi disponibili: 'glove-wiki-gigaword-50/100/200/300', 'glove-twitter-100', ecc.
        self.glove_model_name = getattr(args, 'glove_model_name', 'glove-wiki-gigaword-100')
        self._glove = None  # caricato pigramente e riusato tra i metodi

    def _load_glove(self):
        if self._glove is None:
            print(f'Caricamento GloVe pre-addestrato: {self.glove_model_name} ...')
            self._glove = gensim_api.load(self.glove_model_name)
        return self._glove

    def _phrase_vector(self, tokens, glove):
        """Media dei vettori GloVe delle parole presenti nel vocabolario (mean pooling)."""
        vecs = [glove[t] for t in tokens if t in glove]
        if len(vecs) == 0:
            return np.zeros(glove.vector_size)
        return np.mean(vecs, axis=0)

    # ------------------------------------------------------------------
    # CORE NODES (pubblicazioni, dataset) -> media di vettori GloVe (livello documento)
    # ------------------------------------------------------------------
    def create_core_nodes_embeddings(self):
        glove = self._load_glove()

        publications_df = pd.read_csv(f'{self.path}/publications.csv')
        publications = publications_df[['content']]
        publications_id = publications_df[['id']]
        datasets_df = pd.read_csv(f'{self.path}/datasets.csv')
        datasets = datasets_df[['content']]
        datasets_id = datasets_df[['id']]

        publication_embeddings = [
            self._phrase_vector(simple_tokenize(t), glove) for t in publications['content']
        ]
        dataset_embeddings = [
            self._phrase_vector(simple_tokenize(t), glove) for t in datasets['content']
        ]

        os.makedirs(f"model/data_simpler/{self.dataset}/embeddings", exist_ok=True)
        with open(f"model/data_simpler/{self.dataset}/embeddings/publications_embeddings.pkl", "wb") as fOut:
            pickle.dump({"ids": publications_id['id'].tolist(), "embeddings": publication_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"model/data_simpler/{self.dataset}/embeddings/datasets_embeddings.pkl", "wb") as fOut:
            pickle.dump({"ids": datasets_id['id'].tolist(), "embeddings": dataset_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    # ------------------------------------------------------------------
    # KEYWORDS / ENTITIES / TOPICS -> media di vettori GloVe (livello termine/frase breve)
    # ------------------------------------------------------------------
    def create_hub_keys_embeddings(self):
        glove = self._load_glove()

        has_keywords = "mes" not in self.path
        if has_keywords:
            keywords_df = pd.read_csv(f'{self.path}/keywords.csv')
            keywords_id = keywords_df[['id']]
            keywords_embeddings = [
                self._phrase_vector(simple_tokenize(t), glove) for t in keywords_df['name']
            ]
            with open(f"model/data_simpler/{self.dataset}/embeddings/keywords_embeddings.pkl", "wb") as fOut:
                pickle.dump({"ids": keywords_id['id'].tolist(), "embeddings": keywords_embeddings}, fOut,
                            protocol=pickle.HIGHEST_PROTOCOL)

            venues_df = pd.read_csv(f'{self.path}/venues.csv')
            venues_id = venues_df[['id']]
            venues_embeddings = [
                self._phrase_vector(simple_tokenize(t), glove) for t in venues_df['name']
            ]
            with open(f"model/data_simpler/{self.dataset}/embeddings/venues_embeddings.pkl", "wb") as fOut:
                pickle.dump({"ids": venues_id['id'].tolist(), "embeddings": venues_embeddings}, fOut,
                            protocol=pickle.HIGHEST_PROTOCOL)


            orgs_df = pd.read_csv(f'{self.path}/organizations.csv')
            orgs_id = orgs_df[['id']]
            orgs_embeddings = [
                self._phrase_vector(simple_tokenize(t), glove) for t in orgs_df['name']
            ]
            with open(f"model/data_simpler/{self.dataset}/embeddings/organizations_embeddings.pkl", "wb") as fOut:
                pickle.dump({"ids": orgs_id['id'].tolist(), "embeddings": orgs_embeddings}, fOut,
                            protocol=pickle.HIGHEST_PROTOCOL)

        entities_df = pd.read_csv(f'{self.path}/entities.csv')
        entities_id = entities_df[['id']]
        entities_embeddings = [
            self._phrase_vector(simple_tokenize(t), glove) for t in entities_df['name']
        ]
        print(len(entities_embeddings[0]))
        with open(f"model/data_simpler/{self.dataset}/embeddings/entities_embeddings.pkl", "wb") as fOut:
            pickle.dump({"ids": entities_id['id'].tolist(), "embeddings": entities_embeddings}, fOut,
                        protocol=pickle.HIGHEST_PROTOCOL)

        topics_df = pd.read_csv(f'{self.path}/topics_keywords_2.csv')
        topics_id = topics_df[['id']]
        topics_embeddings = [
            self._phrase_vector(simple_tokenize(t), glove) for t in topics_df['description']
        ]
        with open(f"model/data_simpler/{self.dataset}/embeddings/topics_embeddings.pkl", "wb") as fOut:
            pickle.dump({"ids": topics_id['id'].tolist(), "embeddings": topics_embeddings}, fOut,
                        protocol=pickle.HIGHEST_PROTOCOL)





        # authors
        authors_df = pd.read_csv(f'{self.path}/authors.csv')
        authors_id = authors_df[['id']]
        authors_embeddings = [
            self._phrase_vector(simple_tokenize(t), glove) for t in authors_df['fullname']
        ]
        with open(f"model/data_simpler/{self.dataset}/embeddings/authors_embeddings.pkl", "wb") as fOut:
            pickle.dump({"ids": authors_id['id'].tolist(), "embeddings": authors_embeddings}, fOut,
                        protocol=pickle.HIGHEST_PROTOCOL)


    # ------------------------------------------------------------------
    # GRAFO -> DeepWalk (Node2Vec con p=1, q=1: random walk non biasati)
    # Salvato in una cartella dedicata "deepwalk", separata da "node2vec"
    # ------------------------------------------------------------------
    def create_hub_top_embeddings(self):
        G = utils.create_graph_csv(f'datasets/{self.dataset}/', self.dataset, ['all'])
        deepwalk = Node2Vec(
            G,
            dimensions=128,
            walk_length=15,
            num_walks=10,
            p=1,   # p=q=1 => passeggiate puramente casuali, cioè DeepWalk
            q=1,
            workers=4
        )

        model = deepwalk.fit(
            window=10,
            min_count=1,
            batch_words=4
        )
        EMBEDDING_MODEL = f'model/data_simpler/{self.dataset}/embeddings/deepwalk'
        os.makedirs(EMBEDDING_MODEL, exist_ok=True)
        model.wv.save_word2vec_format(
            f'{EMBEDDING_MODEL}/deepwalk_model.bin',
            binary=True
        )


    def save_embedding(self):
        EMBEDDING_MODEL = f'model/data_simpler/{self.dataset}/embeddings/deepwalk'
        model = KeyedVectors.load_word2vec_format(
            f"{EMBEDDING_MODEL}/deepwalk_model.bin",
            binary=True
        )
        print(model)

        embeddings = []
        ids_list = []
        for id in model.key_to_index:
            embeddings.append(model[str(id)])
            ids_list.append(str(id))

        with open(f"{EMBEDDING_MODEL}/nodenetwork_embeddings.pkl", "wb") as fOut:
            pickle.dump({"ids": ids_list, "embeddings": embeddings}, fOut,
                        protocol=pickle.HIGHEST_PROTOCOL)

        orgs_list, orgs_embeddings = [], []
        venues_list, venues_embeddings = [], []
        authors_list, authors_embeddings = [], []
        pubs_list, pubs_embeddings = [], []
        data_list, data_embeddings = [], []
        entities_list, entities_embeddings = [], []
        topics_list, topics_embeddings = [], []
        keys_list, keys_embeddings = [], []

        for id in model.key_to_index:
            print(id)
            if id[0] == 'a':
                authors_list.append(str(id))
                authors_embeddings.append(model[str(id)])
            elif id[0] == 'o':
                orgs_list.append(str(id))
                orgs_embeddings.append(model[str(id)])
            elif id[0] == 'v':
                venues_list.append(str(id))
                venues_embeddings.append(model[str(id)])
            elif id[0] == 'p':
                pubs_list.append(str(id))
                pubs_embeddings.append(model[str(id)])
            elif id[0:2] == 'db':
                entities_list.append(str(id))
                entities_embeddings.append(model[str(id)])
            elif id[0] == 'd':
                data_list.append(str(id))
                data_embeddings.append(model[str(id)])
            elif id[0] == 't':
                topics_list.append(str(id))
                topics_embeddings.append(model[str(id)])
            elif id[0] == 'k':
                keys_list.append(str(id))
                keys_embeddings.append(model[str(id)])

        def dump(name, ids, embs):
            if len(ids) > 0:
                with open(f"{EMBEDDING_MODEL}/nodenetwork_embeddings.pkl", "wb") as fOut:
                    pickle.dump({"ids": ids, "embeddings": embs}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

        dump("authors", authors_list, authors_embeddings)
        dump("organizations", orgs_list, orgs_embeddings)
        dump("venues", venues_list, venues_embeddings)
        dump("publications", pubs_list, pubs_embeddings)
        dump("datasets", data_list, data_embeddings)
        dump("keywords", keys_list, keys_embeddings)
        dump("topics", topics_list, topics_embeddings)
        print(len(entities_list))
        dump("entities", entities_list, entities_embeddings)



if __name__ == '__main__':
    args = args_list.get_args()
    for dataset in ['mes', 'pubmed']:
        print(f'DATASET {dataset}')
        args.dataset = dataset
        preproc = Preprocessor(args)
        preproc.create_core_nodes_embeddings()
        preproc.create_hub_keys_embeddings()
        preproc.create_hub_top_embeddings()
        preproc.save_embedding()