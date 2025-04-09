from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle
import networkx as nx
# from node2vec import Node2Vec
import args_list
import utils
import tqdm
from nodevectors import Node2Vec
from gensim.models import KeyedVectors


class Preprocessor:
    def __init__(self,args):
        self.dataset = args.dataset
        self.path = f'datasets/mes/{args.dataset}/final'
        self.core_pre_model = args.core_sentence_transformer
        self.key_sentence_transformer = args.key_sentence_transformer
        # self.universal_mapping = self.create_mapping()


    def create_core_nodes_embeddings(self):
        publications_df = pd.read_csv(f'{self.path}/publications.csv')
        publications = publications_df[['content']]
        publications_id = publications_df[['id']]
        datasets_df = pd.read_csv(f'{self.path}/datasets.csv')
        datasets = datasets_df[['content']]
        datasets_id = datasets_df[['id']]
        model = SentenceTransformer(f'sentence-transformers/{self.core_pre_model}')

        publication_embeddings = model.encode(publications['content'])
        dataset_embeddings = model.encode(datasets['content'])
        print(type(publication_embeddings))
        with open(f"model/data/{self.dataset}/embeddings/publications_embeddings.pkl", "wb") as fOut:
            pickle.dump({"ids": publications_id['id'].tolist(), "embeddings": publication_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
        fOut.close()
        with open(f"model/data/{self.dataset}/embeddings/datasets_embeddings.pkl", "wb") as fOut:
            pickle.dump({"ids": datasets_id['id'].tolist(), "embeddings": dataset_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
        fOut.close()

    def create_hub_keys_embeddings(self):

        model = SentenceTransformer(f'{self.key_sentence_transformer}')

        if "mes" not in self.path:
            keywords_df = pd.read_csv(f'{self.path}/keywords.csv')
            keywords = keywords_df[['name']]
            keywords_id = keywords_df[['id']]
            keywords_embeddings = model.encode(keywords['name'])
            with open(f"model/data/{self.dataset}/embeddings/keywords_embeddings.pkl", "wb") as fOut:
                pickle.dump({"ids": keywords_id['id'].tolist(), "embeddings": keywords_embeddings}, fOut,
                            protocol=pickle.HIGHEST_PROTOCOL)
            fOut.close()

        entities_df = pd.read_csv(f'{self.path}/entities.csv')
        entities = entities_df[['name']]
        entities_id = entities_df[['id']]
        entities_embeddings = model.encode(entities['name'])
        print(len(entities_embeddings[0]))
        with open(f"model/data/{self.dataset}/embeddings/entities_embeddings.pkl", "wb") as fOut:
            pickle.dump({"ids": entities_id['id'].tolist(), "embeddings": entities_embeddings}, fOut,
                        protocol=pickle.HIGHEST_PROTOCOL)
        fOut.close()


        topics_df = pd.read_csv(f'{self.path}/topics_keywords_2.csv')
        topics = topics_df[['description']]
        topics_id = topics_df[['id']]
        topics_embeddings = model.encode(topics['description'])
        with open(f"model/data/{self.dataset}/embeddings/topics_embeddings.pkl", "wb") as fOut:
            pickle.dump({"ids": topics_id['id'].tolist(), "embeddings": topics_embeddings}, fOut,
                        protocol=pickle.HIGHEST_PROTOCOL)
        fOut.close()

    def create_hub_top_embeddings(self):
        G = utils.create_graph_csv(f'datasets/{self.dataset}/all/final', self.dataset, ['all'])
        g2v = Node2Vec(
            n_components=128,
            walklen=15
        )
        EMBEDDING_MODEL = f'model/data/{self.dataset}/embeddings/node2vec'
        # way faster than other node2vec implementations
        # Graph edge weights are handled automatically
        g2v.fit(G)
        g2v.save(f'{EMBEDDING_MODEL}/node2vec')

        g2v = Node2Vec.load(f'{EMBEDDING_MODEL}/node2vec.zip')

        # Save model to gensim.KeyedVector format
        g2v.save_vectors(f'{EMBEDDING_MODEL}/node2vec_model.bin')

    def save_embedding(self):
        EMBEDDING_MODEL = f'model/data/{self.dataset}/embeddings/node2vec'
        model = KeyedVectors.load_word2vec_format(f"{EMBEDDING_MODEL}/node2vec_model.bin")
        print(model)

        embeddings = []
        ids_list = []
        # for id in ids_to_search:
        for id in model.vocab:
            embeddings.append(model[str(id)])
            ids_list.append(str(id))

        with open(f"{EMBEDDING_MODEL}/nodenetwork_embeddings.pkl", "wb") as fOut:
            pickle.dump({"ids": ids_list, "embeddings": embeddings}, fOut,
                        protocol=pickle.HIGHEST_PROTOCOL)
        fOut.close()

        orgs_list = []
        orgs_embeddings = []
        venues_list = []
        venues_embeddings = []
        authors_list = []
        authors_embeddings = []
        pubs_list = []
        pubs_embeddings = []
        data_list = []
        data_embeddings = []
        entities_list = []
        entities_embeddings = []
        topics_list = []
        topics_embeddings = []
        keys_list = []
        keys_embeddings = []

        for id in model.vocab:
            print(id)

            embeddings.append(model[str(id)])
            ids_list.append(str(id))
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

        with open(f"{EMBEDDING_MODEL}/authors_net_embeddings.pkl", "wb") as fOut:
            pickle.dump({"ids": authors_list, "embeddings": authors_embeddings}, fOut,
                        protocol=pickle.HIGHEST_PROTOCOL)
        fOut.close()

        if len(orgs_list) > 0:
            with open(f"{EMBEDDING_MODEL}/organizations_net_embeddings.pkl", "wb") as fOut:
                pickle.dump({"ids": orgs_list, "embeddings": orgs_embeddings}, fOut,
                            protocol=pickle.HIGHEST_PROTOCOL)
            fOut.close()

        if len(venues_list) > 0:
            with open(f"{EMBEDDING_MODEL}/venues_net_embeddings.pkl", "wb") as fOut:
                pickle.dump({"ids": venues_list, "embeddings": venues_embeddings}, fOut,
                            protocol=pickle.HIGHEST_PROTOCOL)
            fOut.close()

        if len(pubs_list) > 0:
            with open(f"{EMBEDDING_MODEL}/publications_net_embeddings.pkl", "wb") as fOut:
                pickle.dump({"ids": pubs_list, "embeddings": pubs_embeddings}, fOut,
                            protocol=pickle.HIGHEST_PROTOCOL)
            fOut.close()

        if len(data_list) > 0:
            with open(f"{EMBEDDING_MODEL}/datasets_net_embeddings.pkl", "wb") as fOut:
                pickle.dump({"ids": data_list, "embeddings": data_embeddings}, fOut,
                            protocol=pickle.HIGHEST_PROTOCOL)
            fOut.close()

        if len(keys_list) > 0:
            with open(f"{EMBEDDING_MODEL}/keywords_net_embeddings.pkl", "wb") as fOut:
                pickle.dump({"ids": keys_list, "embeddings": keys_embeddings}, fOut,
                            protocol=pickle.HIGHEST_PROTOCOL)
            fOut.close()

        if len(topics_list) > 0:
            with open(f"{EMBEDDING_MODEL}/topics_net_embeddings.pkl", "wb") as fOut:
                pickle.dump({"ids": topics_list, "embeddings": topics_embeddings}, fOut,
                            protocol=pickle.HIGHEST_PROTOCOL)
            fOut.close()
        print(len(entities_list))
        if len(entities_list) > 0:
            with open(f"{EMBEDDING_MODEL}/entities_net_embeddings.pkl", "wb") as fOut:
                pickle.dump({"ids": entities_list, "embeddings": entities_embeddings}, fOut,
                            protocol=pickle.HIGHEST_PROTOCOL)
            fOut.close()


if __name__ == '__main__':
    args = args_list.get_args()
    preproc = Preprocessor(args)
#     preproc.create_core_nodes_embeddings()
#     preproc.create_hub_keys_embeddings()
    preproc.create_hub_top_embeddings()
