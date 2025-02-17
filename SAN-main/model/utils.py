import networkx as nx
import pandas as pd
import numpy as np
import torch
import os
import random
seed=42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.enabled = False
# torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)


class EarlyStoppingClass:

    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience, verbose=True, save_epoch_path=None, save_embeddings_path=None, save_early_path=None, best_score=None):

        self.patience = patience
        self.counter = 0
        self.best_epoch = 0
        self.best_score = best_score
        self.verbose = verbose
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.save_epoch_path = save_epoch_path
        self.save_early_path = save_early_path
        self.save = False

    def __call__(self, val_loss, model,optimizer,epoch,embeddings=None):
        self.save = False
        score = val_loss
        # self.save_epoch(model,optimizer,epoch,val_loss)


        if epoch > 0:
            print('saving epoch 100')
            self.save_checkpoint(val_loss, model, optimizer, epoch)
            self.save = True


        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model,optimizer,epoch)


        elif score > self.best_score:
            self.save=False
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_epoch = epoch
            self.best_score = score
            self.save_checkpoint(val_loss, model,optimizer,epoch,best=True)
            self.counter = 0


    # def save_epoch(self,model,optimizer,epoch,val_loss):
    #
    #     """Saves model at each epoch."""
    #     torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': val_loss,
    #         'best_score': self.best_score,
    #         'counter': self.counter
    #     }, self.save_epoch_path)
    #     print(f'saved model for epoch: {epoch}')


    def save_checkpoint(self, val_loss, model, optimizer,epoch,best=False):

        """Saves model when validation loss decrease."""

        # self.save_early_path = self.save_early_path.split('_epoch')[0]+'_epoch_'+str(epoch)+'.pt'
        if  best:
            path = self.save_early_path
        else:
            path = self.save_epoch_path
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model {path}')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'best_score': self.best_score,
            'counter': self.counter
        }, path)
        # torch.save(model.state_dict(), path)
        # if not best:
        #     torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def create_graph_csv(path_final,dataset,nodes):

    """Create graph with specific nodes"""

    # path_final = f'./datasets/{dataset}/{set}'
    # path_topics = path_final
    # path_entities = path_final
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
        pubentedges = pd.read_csv(path_final + '/pubentedges.csv')
        dataentedges = pd.read_csv(path_final + '/dataentedges.csv')
        # print(pubentedges.shape[0])
        # print(dataentedges.shape[0])
        edges.append(dataentedges)
        edges.append(pubentedges)


    if 'topics' in nodes or 'all' in nodes:
        pubtopicedges = pd.read_csv(path_final + '/pubtopicedges_keywords_2.csv')
        datatopicedges = pd.read_csv(path_final + '/datatopicedges_keywords_2.csv')

        edges.append(pubtopicedges)
        edges.append(datatopicedges)



    edges_concat = pd.concat(edges,
                             ignore_index=True)
    G = nx.from_pandas_edgelist(edges_concat, 'source', 'target')

    return G
