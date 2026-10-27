# import os.path
# from args_list import get_args
# from utils import create_graph_csv
# import random
# import networkx as nx
# import walker
# import multiprocessing as mp
# from collections import Counter
# import time
# import pickle
# import numpy as np
# from loader import ScholarlyDataset
# from sklearn.preprocessing import normalize
# import pandas as pd
# import torch
# import numpy as np
# import torch
# import random
# import multiprocessing
# import concurrent.futures
# from functools import partial
# from torch_geometric import seed_everything
#
# from multiprocessing import Pool
# from collections import Counter
# from functools import lru_cache
#
#
# @lru_cache(maxsize=1000000)
# def cached_cosine(seed_id, n_id):
#     return cosine_similarity(seed_id, n_id)
#
# def seed_torch(seed=42):
#     random.seed(seed)
#     seed_everything(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
#     torch.use_deterministic_algorithms(True)
#
#
# seed_torch()
#
#
# def cosine_similarity(emb1, emb2):
#     dot_product = np.dot(emb1, emb2)
#     norm_vector1 = np.linalg.norm(emb1)
#     norm_vector2 = np.linalg.norm(emb2)
#     similarity = dot_product / (norm_vector1 * norm_vector2)
#     return similarity
#
#
# # --------------------------------------------------------------------------- #
# # WORKER-SIDE GLOBALS
# #
# # process_list() only ever touches: publications_features, datasets_features,
# # args, and the type-predicate helpers (is_publication/is_dataset/...) plus,
# # per-call, `walks` / `seed_to_start_idx`. It never touches self.G / self.G_conv.
# #
# # Previously select_walks_mp() called `self.executor.map(self.process_list, ...)`.
# # Since self.process_list is a *bound method*, pickling it for dispatch to a
# # worker process pickles the whole `self` object -- including self.G / self.G_conv
# # (the full NetworkX graph) which is completely unused by process_list. This
# # happened on *every single mini-batch*, and was almost certainly the dominant
# # cost behind the ~30-40s/it seen in profiling.
# #
# # Fix: move the per-seed logic into a plain module-level function that reads
# # from process-local globals. Those globals are populated exactly ONCE per
# # worker process via the Pool's `initializer`, at pool-creation time -- not
# # re-sent on every map() call. Only the genuinely per-call data (the current
# # walks + which seeds to process) travels through the task queue each time.
# # --------------------------------------------------------------------------- #
#
# _WORKER_PUB_FEATS = None
# _WORKER_DS_FEATS = None
# _WORKER_ARGS = None
# _WORKER_CORE_FEATURES = None
#
# def _is_publication(node):
#     return node.startswith('p_')
#
#
# def _is_dataset(node):
#     return node.startswith('d_')
#
#
# def _is_entity(node):
#     return node.startswith('dbpedia_')
#
#
# def _is_keyword(node):
#     return node.startswith('k_')
#
#
# def _is_topic(node):
#     return node.startswith('t')
#
#
# def _is_author(node):
#     return node.startswith('a_')
#
#
# def _is_venue(node):
#     return node.startswith('v_')
#
#
# def _is_organization(node):
#     return node.startswith('o_')
#
#
# def _init_worker(pub_feats, ds_feats, args,core_features):
#     """Runs once per worker process at Pool-creation time. Everything stored
#     here stays resident in that worker for its whole lifetime -- it is never
#     re-pickled/re-sent on subsequent map() calls."""
#     global _WORKER_PUB_FEATS
#     global _WORKER_DS_FEATS
#     global _WORKER_ARGS
#     global _WORKER_CORE_FEATURES
#
#     _WORKER_PUB_FEATS = pub_feats
#     _WORKER_DS_FEATS = ds_feats
#     _WORKER_ARGS = args
#     _WORKER_CORE_FEATURES = core_features
#
#
# def _process_list(seeds_list, walks, seed_to_start_idx, all_walks):
#     """Module-level twin of RandomWalkWithRestart.process_list(), operating on
#     worker-local globals instead of `self`. Logic is unchanged from the
#     original method."""
#     args = _WORKER_ARGS
#     publications_features = _WORKER_PUB_FEATS
#     datasets_features = _WORKER_DS_FEATS
#     core_features = _WORKER_CORE_FEATURES
#     selected_walks_list, selected_cores_list, selected_hubs_top_list, selected_hubs_key_list = [], [], [], []
#
#     walk_importance_score = [[] for _ in range(len(seeds_list))]
#     cores = [[] for _ in range(len(seeds_list))]
#     hubs_top = [[] for _ in range(len(seeds_list))]
#     hubs_key = [[] for _ in range(len(seeds_list))]
#
#     for j, seed in enumerate(seeds_list):
#         if _is_publication(seed):
#             seed_id = publications_features['mapping'][seed]
#             seed_id = publications_features.x[seed_id]
#         else:
#             seed_id = datasets_features['mapping'][seed]
#             seed_id = datasets_features.x[seed_id]
#
#         ind_seed = seed_to_start_idx[seed]
#         cur_walks = walks[ind_seed: ind_seed + all_walks]
#         for i, w in enumerate(cur_walks):
#             seen = set()
#             walk = []
#             walk = list(dict.fromkeys(w))
#             freq = Counter(w)
#             cores_walk = []
#             hubs_key_walk = []
#             hubs_top_walk = []
#             neighs = walk[1:]
#             cosine_total = 0
#             neigh_count = 0
#             for n in neighs:
#                 if n == seed:
#                     neigh_count = 1
#                 else:
#                     neigh_count += 1
#                     frequency = freq[n]
#                     #frequency = len([a for a in walk if a == n])
#                     distance = neigh_count
#                     penalty = frequency / distance
#                     if _is_publication(n) or _is_dataset(n):
#                         # if _is_publication(n):
#                         #     n_id = publications_features.mapping[n]
#                         #     n_id = publications_features.x[n_id]
#                         # else:
#                         #     n_id = datasets_features.mapping[n]
#                         #     n_id = datasets_features.x[n_id]
#                         n_id = core_features[n]
#                         #
#                         #cosine_0 = cached_cosine(seed_id, n_id)
#                         #cosine_0 = cosine_similarity(seed_id, n_id)
#                         #seed_np = seed_id.detach().cpu().numpy() if torch.is_tensor(seed_id) else seed_id
#                         #n_id_np = n_id.detach().cpu().numpy() if torch.is_tensor(n_id) else n_id
#
#                         cosine_0 = np.dot(seed_id, n_id).item()
#                         #cosine_0 = 1
#                         cosine = cosine_0 * penalty
#                         cores_walk.append((n, cosine_0))
#
#                         cosine_total += cosine
#                     elif _is_author(n) or _is_organization(n) or _is_venue(n):
#                         hubs_top_walk.append((n, penalty))
#                     else:
#                         hubs_key_walk.append((n, penalty))
#
#             if len(cores_walk) == 0:
#                 cosine_total = 0
#             else:
#                 cosine_total = cosine_total / len(cores_walk)
#
#             walk_importance_score[j].append(cosine_total)
#             cores[j].append(cores_walk)
#             hubs_top[j].append(hubs_top_walk)
#             hubs_key[j].append(hubs_key_walk)
#
#         selected_walks = cur_walks  # fallback in case the loop below never runs
#         sorted_walks_ind = sorted(
#             range(len(walk_importance_score[j])),
#             key=lambda i: walk_importance_score[j][i],
#             reverse=True
#         )
#
#
#         for index in range(
#                 args.num_selected_walks,
#                 min(args.num_selected_walks + 1, args.num_random_walks)
#         ):
#             # selected_walks_ind = sorted(range(len(walk_importance_score[j])),
#             #                             key=lambda i: walk_importance_score[j][i], reverse=True)[:index]
#             selected_walks_ind = sorted_walks_ind[:index]
#             selected_walks = [cur_walks[i] for i in selected_walks_ind]
#             selected_cores = [cores[j][i] for i in selected_walks_ind]
#             selected_cores = list(set(item for sublist in selected_cores for item in sublist if item != []))
#
#             selected_hubs_top = [hubs_top[j][i] for i in selected_walks_ind]
#             selected_hubs_top = list(set(item for sublist in selected_hubs_top for item in sublist if item != []))
#
#             selected_hubs_key = [hubs_key[j][i] for i in selected_walks_ind]
#             selected_hubs_key = list(set(item for sublist in selected_hubs_key for item in sublist if item != []))
#
#             if args.test == False:
#                 unique_key_elements = {}
#                 for tup in selected_hubs_key:
#                     if tup[0] not in unique_key_elements:
#                         unique_key_elements[tup[0]] = tup[1]
#
#                 selected_hubs_key = [tuple([k, v]) for k, v in unique_key_elements.items()]
#                 unique_top_elements = {}
#                 for tup in selected_hubs_top:
#                     if tup[0] not in unique_top_elements:
#                         unique_top_elements[tup[0]] = tup[1]
#                 selected_hubs_top = [tuple([k, v]) for k, v in unique_top_elements.items()]
#
#                 unique_core_elements = {}
#                 for tup in selected_cores:
#                     if tup[0] not in unique_core_elements:
#                         unique_core_elements[tup[0]] = tup[1]
#                 selected_cores = [tuple([k, v]) for k, v in unique_core_elements.items()]
#
#             if (len(selected_cores) >= args.n_cores and
#                     len(selected_hubs_key) >= args.n_keys_hubs and
#                     len(selected_hubs_top) >= args.n_top_hubs):
#                 break
#
#         selected_walks_list.append(selected_walks)
#         selected_cores_list.append(selected_cores)
#         selected_hubs_top_list.append(selected_hubs_top)
#         selected_hubs_key_list.append(selected_hubs_key)
#
#     return [selected_walks_list, selected_cores_list, selected_hubs_top_list, selected_hubs_key_list]
#
#
# class RandomWalkWithRestart:
#     """
#     args:
#     - dataset
#     - setup -- inductive light (il), inductive full (if), inductive hard (ih), transductive (t)
#     - set -- train, validation, test
#     - max number of paths per node -- the paths selected to choose neighbours from -- default 10
#     - number of selected neighbours
#     - number of core nodes
#     - max path's length
#     - seed_node
#     - restart prob
#     - nx_graph
#     - distance penalty -- the penalty applied to distant nodes
#     - reward frequency -- the reward applied to frequent nodes
#
#     """
#
#     def __init__(self, args, sch_dataset, root, set_type=None, test=False):
#         # root = sch_dataset.root
#         # sch_dataset = sch_dataset[0]
#         self.args = args
#         self.set_type = set_type
#         self.path_data = f'datasets/{args.dataset}/'
#         self.data = sch_dataset
#         self.max_walk_length = args.max_walk_length
#         self.all_walks = args.num_random_walks  # at the beginning compute 100 walks per node and store them, sampling will be random on these walks
#         self.alpha = args.restart_probability
#
#         self.G = create_graph_csv(self.path_data, self.args.dataset, ['all'])
#         self.node_type = {}
#
#         # Filter edges based on the selected source and target nodes
#         if 'transductive' in root:
#             # rimuovo gli archi tra p e d, li aggiungo dopo dipendentemente da cosa mi serve
#             # qua sono in transduttivo e divido gli archi per il message passing, tenendo solo la kcore per il mp
#             if 'train' in root:
#                 edge_index = sch_dataset['publication', 'cites', 'dataset'].edge_index_train
#             elif 'validation' in root:
#                 edge_index = sch_dataset['publication', 'cites', 'dataset'].edge_index_validation
#             elif 'test' in root:
#                 edge_index = sch_dataset['publication', 'cites', 'dataset'].edge_index_test
#             else:
#                 exit(1)
#             edges_to_remove = [(u, v) for u, v in self.G.edges if
#                                (u.startswith('p_') and v.startswith('d_')) or (
#                                        u.startswith('d_') and v.startswith('p'))]
#
#             # rimuovo tutti gli edges tra p e d e aggiungo quelli che ho trovato sopra
#             edges_to_add = [
#                 tuple([sch_dataset['publication'].rev_mapping[s[0]], sch_dataset['dataset'].rev_mapping[s[1]]]) for s in
#                 edge_index.t().tolist()]
#
#             edges_to_remove_topic = []
#             edges_to_remove_entity = []
#             if args.nodes_to_remove == 'entity':
#                 edges_to_remove_entity = [(u, v) for u, v in self.G.edges if
#                                           (u.startswith('p_') and v.startswith('db')) or (
#                                                   u.startswith('d_') and v.startswith('db'))]
#                 edges_to_remove_entity.extend([(u, v) for u, v in self.G.edges if
#                                                (u.startswith('p_') and v.startswith('e')) or (
#                                                        u.startswith('d_') and v.startswith('e'))])
#             if args.nodes_to_remove == 'topic':
#                 edges_to_remove_topic = [(u, v) for u, v in self.G.edges if
#                                          (u.startswith('p_') and v.startswith('t')) or (
#                                                  u.startswith('d_') and v.startswith('t'))]
#
#             self.G.remove_edges_from(edges_to_remove)
#             self.G.remove_edges_from(edges_to_remove_entity)
#             self.G.remove_edges_from(edges_to_remove_topic)
#             self.G.add_edges_from(edges_to_add)
#
#             # edges_to_remove = [(u, v) for u, v in self.G.edges if v.startswith('tk')]
#             # self.G.remove_edges_from(edges_to_remove)
#
#         # tolgo pubblicazioni e dataset in ind e semi dal grafo e poi alleno
#         edge_index_test_trans = sch_dataset['publication', 'cites', 'dataset'].edge_label_index_test_trans
#         edge_index_test_semi = sch_dataset['publication', 'cites', 'dataset'].edge_label_index_test_semi
#         edge_index_test_ind = sch_dataset['publication', 'cites', 'dataset'].edge_label_index_test_ind
#
#         sources = list(set(edge_index_test_ind[0].tolist() + edge_index_test_semi[0].tolist()))
#         targets = list(set(edge_index_test_ind[1].tolist()))
#
#         if not test:
#             sources = [sch_dataset['publication'].rev_mapping[s] for s in sources]
#             targets = [sch_dataset['dataset'].rev_mapping[s] for s in targets]
#             self.removed = sources + targets
#             self.G.remove_nodes_from(sources)
#             self.G.remove_nodes_from(targets)
#
#         else:
#             if self.args.inductive_type == 'light':
#                 print('light')
#                 sources = list(set(edge_index_test_ind[0].tolist()))
#                 targets = list(set(edge_index_test_ind[1].tolist()))
#                 sources = [sch_dataset['publication'].rev_mapping[s] for s in sources]
#                 targets = [sch_dataset['dataset'].rev_mapping[s] for s in targets]
#                 self.removed = sources + targets
#                 print(len(self.G.edges))
#                 self.G.remove_edges_from(list(self.G.edges(self.removed)))
#                 print(len(self.G.edges))
#
#                 # self.G.remove_nodes_from(self.removed)
#                 # self.G.remove_nodes_from(targets)
#
#             elif self.args.inductive_type == 'trans':
#                 print('trans')
#                 sources = list(set(edge_index_test_ind[0].tolist() + edge_index_test_semi[0].tolist()))
#                 targets = list(set(edge_index_test_ind[1].tolist()))
#                 sources = [sch_dataset['publication'].rev_mapping[s] for s in sources]
#                 targets = [sch_dataset['dataset'].rev_mapping[s] for s in targets]
#                 self.removed = sources + targets
#                 print(len(self.G.edges))
#                 self.G.remove_edges_from(list(self.G.edges(self.removed)))
#                 print(len(self.G.edges))
#
#                 # self.G.remove_nodes_from(self.removed)
#                 # self.G.remove_nodes_from(sources)
#                 # self.G.remove_nodes_from(targets)
#
#         self.G_conv = nx.convert_node_labels_to_integers(self.G)
#         self.mapping = {j: i for j, i in zip(self.G_conv, self.G)}
#         self.rev_mapping = {i: j for j, i in zip(self.G_conv, self.G)}
#         self.publications_features, self.datasets_features = sch_dataset['publication'], sch_dataset['dataset']
#         self.publications_features.x = torch.from_numpy(
#             normalize(self.publications_features.x, axis=1)
#         ).float()
#
#         self.datasets_features.x = torch.from_numpy(
#             normalize(self.datasets_features.x, axis=1)
#         ).float()
#
#         self.core_features = {}
#
#         for k, v in self.publications_features['mapping'].items():
#             self.core_features[k] = self.publications_features.x[v]
#
#         for k, v in self.datasets_features['mapping'].items():
#             self.core_features[k] = self.datasets_features.x[v]
#
#         # self.publications_features,self.datasets_features = self.load_features()
#
#         sources = list(set(edge_index_test_ind[0].tolist()))
#         targets = list(set(edge_index_test_ind[1].tolist()))
#         sources = [sch_dataset['publication'].rev_mapping[s] for s in sources]
#         targets = [sch_dataset['dataset'].rev_mapping[s] for s in targets]
#         print(f'intersection with ind {len(list(set(sources).intersection(set(self.G))))}')
#         print(f'intersection with ind {len(list(set(targets).intersection(self.G)))}')
#
#         sources = list(set(edge_index_test_semi[0].tolist()))
#         targets = list(set(edge_index_test_semi[1].tolist()))
#         sources = [sch_dataset['publication'].rev_mapping[s] for s in sources]
#         targets = [sch_dataset['dataset'].rev_mapping[s] for s in targets]
#         print(f'intersection with semi {len(list(set(sources).intersection(set(self.G))))}')
#         print(f'intersection with semi {len(list(set(targets).intersection(self.G)))}')
#         print(f'length datasets semi {len(list(set(targets)))}')
#
#         sources = list(set(edge_index_test_trans[0].tolist()))
#         targets = list(set(edge_index_test_trans[1].tolist()))
#         sources = [sch_dataset['publication'].rev_mapping[s] for s in sources]
#         targets = [sch_dataset['dataset'].rev_mapping[s] for s in targets]
#         print(targets)
#         print(f'intersection with trans {len(list(set(sources).intersection(set(self.G))))}')
#         print(f'intersection with trans {len(list(set(targets).intersection(self.G)))}')
#         print(f'length datasets trans {len(list(set(targets)))}')
#
#         # Persistent worker pool: created ONCE per walker instance instead of being
#         # spawned/torn down on every call to select_walks_mp (i.e. every mini-batch,
#         # every epoch). Re-creating a 40-worker pool that many times was one of the
#         # biggest sources of overhead in the training loop.
#         #
#         # `initializer`/`initargs` push the (constant, per-run) feature tensors and
#         # args into each worker EXACTLY ONCE, at pool start-up. This is what lets
#         # select_walks_mp() below avoid re-pickling self.G / self.G_conv / features
#         # on every mini-batch -- previously the single biggest source of overhead.
#         self.executor = concurrent.futures.ProcessPoolExecutor(
#             max_workers=10,
#             initializer=_init_worker,
#             initargs=(
#                 self.publications_features,
#                 self.datasets_features,
#                 self.args,
#                 self.core_features,
#             ),
#         )
#
#     def get_seeds(self):
#         return self.seeds
#
#     def reset_seeds(self):
#         self.seeds = []
#
#     def __getstate__(self):
#         state = self.__dict__.copy()
#         # ProcessPoolExecutor holds locks/threads - never picklable. Kept as a
#         # safety net; select_walks_mp() no longer sends `self` to workers at all,
#         # but this protects against anything else in the codebase pickling the
#         # walker instance (e.g. checkpointing).
#         state.pop('executor', None)
#         return state
#
#     def __setstate__(self, state):
#         self.__dict__.update(state)
#         self.executor = None
#
#     def close(self):
#         """Call this once training/inference is done to shut down the pool and
#         avoid leaking 40 worker processes for the lifetime of the run."""
#         if getattr(self, 'executor', None) is not None:
#             self.executor.shutdown(wait=True)
#             self.executor = None
#
#     def __del__(self):
#         try:
#             self.close()
#         except Exception:
#             pass
#
#     def create_random_walks(self, seeds_in=False, all=False):
#         if all:
#             seeds = [node for node in self.G.nodes() if self.is_publication(node) or self.is_dataset(node)]
#             self.seeds = seeds
#         else:
#             if not seeds_in:
#                 seeds = self.seeds
#             else:
#                 self.seeds = seeds_in
#                 seeds = seeds_in
#
#         # print(f'TEST JOIN {len(list(set(self.seeds).intersection(self.removed)))}')
#         # print(f'TEST JOIN {len(list(set(self.seeds).intersection(self.G.nodes())))}')
#         # print(f'TEST JOIN {len(list(set(self.removed).intersection(self.G.nodes())))}')
#         if seeds is not None:
#             converted_seeds = [self.rev_mapping[x] for x in self.seeds]
#             walks = walker.random_walks(self.G_conv, n_walks=self.all_walks, alpha=self.alpha,
#                                         walk_len=self.max_walk_length, start_nodes=converted_seeds, q=0.3, p=1.0)
#         else:
#             walks = walker.random_walks(self.G_conv, n_walks=self.all_walks, alpha=self.alpha,
#                                         walk_len=random.randint(1, self.max_walk_length), p=1.0, q=0.1)
#         walks = sorted(walks, key=lambda x: x[0])
#         walks_rev = []
#         for walk in walks:
#             walks_rev.append([self.mapping[p] for p in walk])
#
#         return walks_rev
#
#     def set_seeds(self, seeds=None):
#         if seeds is None:
#             seeds = [node for node in self.G.nodes() if self.is_publication(node) or self.is_dataset(node)]
#             self.seeds = seeds
#         else:
#             self.seeds = seeds
#
#     def is_publication(self, node):
#         return node.startswith('p_')
#
#     def is_dataset(self, node):
#         return node.startswith('d_')
#
#     def is_entity(self, node):
#         return node.startswith('dbpedia_')
#
#     def is_keyword(self, node):
#         return node.startswith('k_')
#
#     def is_topic(self, node):
#         return node.startswith('t')
#
#     def is_author(self, node):
#         return node.startswith('a_')
#
#     def is_venue(self, node):
#         return node.startswith('v_')
#
#     def is_organization(self, node):
#         return node.startswith('o_')
#
#     def get_walks_threshold(self, walk_scores):
#         return np.percentile(walk_scores, 75)
#
#     def select_walks_mp(self, walks):
#         """Select the walks such that
#             - the mean cos sim is maximized
#             - min num neighbours is reached
#
#         Only `walks`, `seed_to_start_idx` and the seed sublists are shipped to
#         the (already-running) worker pool per call -- these are the only things
#         that actually change between mini-batches. Everything constant
#         (features, args) was loaded into the workers once, in __init__.
#         """
#
#         def chunk_list(main_list, chunk_size):
#             for i in range(0, len(main_list), chunk_size):
#                 yield main_list[i:i + chunk_size]
#
#         selected_seeds_cores = [[] for _ in range(len(self.seeds))]
#         selected_seeds_hubs_top = [[] for _ in range(len(self.seeds))]
#         selected_seeds_hubs_key = [[] for _ in range(len(self.seeds))]
#         selected_seeds_walks = [[] for _ in range(len(self.seeds))]
#
#         first_ind = [w[0] for w in walks]
#         self.first_ind = first_ind
#         self.walks = walks
#
#         # O(len(walks)) precompute of seed -> starting index, instead of the
#         # O(seeds * walks) `list.index()` scan the non-mp select_walks() still does.
#         seed_to_start_idx = {}
#         for idx, s in enumerate(first_ind):
#             if s not in seed_to_start_idx:
#                 seed_to_start_idx[s] = idx
#         self.seed_to_start_idx = seed_to_start_idx
#
#         chunk_size = 2000
#         sublists = list(chunk_list(self.seeds, chunk_size))
#
#         # `partial` binds walks/seed_to_start_idx/all_walks once; only the varying
#         # `seeds_list` argument differs per task. walks/seed_to_start_idx are the
#         # only genuinely per-mini-batch payload sent over the task queue now.
#         worker_fn = partial(_process_list, walks=walks, seed_to_start_idx=seed_to_start_idx,
#                             all_walks=self.all_walks)
#         try:
#             t0 = time.time()
#
#             results = list(self.executor.map(worker_fn, sublists))
#
#             print(
#                 "executor time:",
#                 time.time() - t0
#             )
# #            results = list(self.executor.map(worker_fn, sublists))
#         except Exception as e:
#             import traceback
#             traceback.print_exc()
#             raise e
#         j = 0
#         for result in results:
#             walk_list, core_list, hubtop_list, hubkey_list = result[0], result[1], result[2], result[3]
#             for walk, core, hubtop, hubkey in zip(walk_list, core_list, hubtop_list, hubkey_list):
#                 selected_seeds_walks[j] = walk
#                 selected_seeds_cores[j] = core
#                 selected_seeds_hubs_top[j] = hubtop
#                 selected_seeds_hubs_key[j] = hubkey
#                 j += 1
#
#         final_cores, final_hubs_key, final_hubs_top = [], [], []
#         selected_seeds_cores = [sorted(s, key=lambda x: x[1], reverse=True) for s in selected_seeds_cores]
#         for i, s in enumerate(selected_seeds_cores):
#             final_cores.append(
#                 sorted(list({x[0]: x for x in reversed(s)}.values())[::-1], key=lambda p: (p[1], p[0]), reverse=True))
#
#         selected_seeds_hubs_top = [sorted(s, key=lambda x: x[1], reverse=True) for s in selected_seeds_hubs_top]
#         for i, s in enumerate(selected_seeds_hubs_top):
#             final_hubs_top.append(
#                 sorted(list({x[0]: x for x in reversed(s)}.values())[::-1], key=lambda p: (p[1], p[0]), reverse=True)[
#                 :self.args.n_top_hubs])
#
#         selected_seeds_hubs_key = [sorted(s, key=lambda x: x[1], reverse=True) for s in selected_seeds_hubs_key]
#         for i, s in enumerate(selected_seeds_hubs_key):
#             final_hubs_key.append(
#                 sorted(list({x[0]: x for x in reversed(s)}.values())[::-1], key=lambda p: (p[1], p[0]), reverse=True)[
#                 :self.args.n_keys_hubs])
#
#         return selected_seeds_walks, final_cores, final_hubs_key, final_hubs_top
#
#
#     def get_neighbours_vector(self, sel_cores, sel_keys, sel_top):
#
#         publications_vectors = [[] for _ in range(len(self.seeds))]
#         publications_net_vectors = [[] for _ in range(len(self.seeds))]
#         datasets_vectors = [[] for _ in range(len(self.seeds))]
#         datasets_net_vectors = [[] for _ in range(len(self.seeds))]
#         authors_vectors = [[] for _ in range(len(self.seeds))]
#         venues_vectors = [[] for _ in range(len(self.seeds))]
#         keywords_vectors = [[] for _ in range(len(self.seeds))]
#         organizations_vectors = [[] for _ in range(len(self.seeds))]
#         entities_vectors = [[] for _ in range(len(self.seeds))]
#         topics_vectors = [[] for _ in range(len(self.seeds))]
#
#         ret_cores = [[] for _ in range(len(self.seeds))]
#         ret_net_cores = [[] for _ in range(len(self.seeds))]
#         ret_tops = [[] for _ in range(len(self.seeds))]
#         ret_net_tops = [[] for _ in range(len(self.seeds))]
#         ret_keys = [[] for _ in range(len(self.seeds))]
#         ret_net_keys = [[] for _ in range(len(self.seeds))]
#         vectors = []
#         vectors_net = []
#         for i, seed in enumerate(self.seeds):
#             if self.is_publication(seed):
#                 vectors.append([seed, seed, 1.0, self.data['publication'].mapping[seed],
#                                 self.data['publication'].x[self.data['publication'].mapping[seed]]])
#                 vectors_net.append([seed, seed, 1.0, self.data['publication'].mapping[seed],
#                                     self.data['publication'].net_x[self.data['publication'].mapping[seed]]])
#             elif self.is_dataset(seed):
#                 vectors.append([seed, seed, 1.0, self.data['dataset'].mapping[seed],
#                                 self.data['dataset'].x[self.data['dataset'].mapping[seed]]])
#                 vectors_net.append([seed, seed, 1.0, self.data['dataset'].mapping[seed],
#                                     self.data['dataset'].net_x[self.data['dataset'].mapping[seed]]])
#
#             cores = sel_cores[i]
#             publications_net = [[seed, c[0], c[1], self.data['publication'].mapping[c[0]],
#                                  self.data['publication'].net_x[self.data['publication'].mapping[c[0]]]] for c in cores
#                                 if self.is_publication(c[0])]
#             publications = [[seed, c[0], c[1], self.data['publication'].mapping[c[0]],
#                              self.data['publication'].x[self.data['publication'].mapping[c[0]]]] for c in cores if
#                             self.is_publication(c[0])]
#             datasets_net = [[seed, c[0], c[1], self.data['dataset'].mapping[c[0]],
#                              self.data['dataset'].net_x[self.data['dataset'].mapping[c[0]]]] for c in cores if
#                             self.is_dataset(c[0])]
#
#             datasets = [[seed, c[0], c[1], self.data['dataset'].mapping[c[0]],
#                          self.data['dataset'].x[self.data['dataset'].mapping[c[0]]]] for c in cores if
#                         self.is_dataset(c[0])]
#
#             tops = sel_top[i]
#             authors = [[seed, c[0], c[1], self.data['author'].mapping[c[0]],
#                         self.data['author'].x[self.data['author'].mapping[c[0]]]] for c in tops if
#                        self.is_author(c[0])]
#             authors_net = [[seed, c[0], c[1], self.data['author'].mapping[c[0]],
#                             self.data['author'].net_x[self.data['author'].mapping[c[0]]]] for c in tops if
#                            self.is_author(c[0])]
#             venues = [[seed, c[0], c[1], self.data['venue'].mapping[c[0]],
#                        self.data['venue'].x[self.data['venue'].mapping[c[0]]]] for c in tops if self.is_venue(c[0])]
#             venues_net = [[seed, c[0], c[1], self.data['venue'].mapping[c[0]],
#                            self.data['venue'].net_x[self.data['venue'].mapping[c[0]]]] for c in tops if
#                           self.is_venue(c[0])]
#
#             organizations = [[seed, c[0], c[1], self.data['organization'].mapping[c[0]],
#                               self.data['organization'].x[self.data['organization'].mapping[c[0]]]] for c in tops if
#                              self.is_organization(c[0])]
#             organizations_net = [[seed, c[0], c[1], self.data['organization'].mapping[c[0]],
#                                   self.data['organization'].net_x[self.data['organization'].mapping[c[0]]]] for c in
#                                  tops if
#                                  self.is_organization(c[0])]
#
#             keys = sel_keys[i]
#             keywords = [[seed, c[0], c[1], self.data['keyword'].mapping[c[0]],
#                          self.data['keyword'].x[self.data['keyword'].mapping[c[0]]]] for c in keys if
#                         self.is_keyword(c[0])]
#             entities = [[seed, c[0], c[1], self.data['entity'].mapping[c[0]],
#                          self.data['entity'].x[self.data['entity'].mapping[c[0]]]] for c in keys if
#                         self.is_entity(c[0])]
#             topics = [[seed, c[0], c[1], self.data['topic'].mapping[c[0]],
#                        self.data['topic'].x[self.data['topic'].mapping[c[0]]]] for c in keys if self.is_topic(c[0])]
#             keywords_net = [[seed, c[0], c[1], self.data['keyword'].mapping[c[0]],
#                              self.data['keyword'].net_x[self.data['keyword'].mapping[c[0]]]] for c in keys if
#                             self.is_keyword(c[0])]
#             entities_net = [[seed, c[0], c[1], self.data['entity'].mapping[c[0]],
#                              self.data['entity'].net_x[self.data['entity'].mapping[c[0]]]] for c in keys if
#                             self.is_entity(c[0])]
#             topics_net = [[seed, c[0], c[1], self.data['topic'].mapping[c[0]],
#                            self.data['topic'].net_x[self.data['topic'].mapping[c[0]]]] for c in keys if
#                           self.is_topic(c[0])]
#
#             publications_vectors[i] = publications
#             publications_net_vectors[i] = publications_net
#             datasets_net_vectors[i] = datasets_net
#             datasets_vectors[i] = datasets
#             entities_vectors[i] = entities
#             topics_vectors[i] = topics
#             authors_vectors[i] = authors
#             venues_vectors[i] = venues
#             organizations_vectors[i] = organizations
#             keywords_vectors[i] = keywords
#
#             if self.args.split_cores:
#                 if self.is_publication(seed):
#                     if len(datasets) >= self.args.n_cores:
#                         ret_cores[i] = datasets[0:self.args.n_cores]
#                         ret_net_cores[i] = datasets_net[0:self.args.n_cores]
#                     else:
#                         ret_cores[i] = datasets + publications
#                         ret_cores[i] = ret_cores[i][0:self.args.n_cores]
#                         ret_net_cores[i] = datasets_net + publications_net
#                         ret_net_cores[i] = ret_net_cores[i][0:self.args.n_cores]
#                 elif self.is_dataset(seed):
#                     if len(publications) >= self.args.n_cores:
#                         ret_cores[i] = publications[0:self.args.n_cores]
#                         ret_net_cores[i] = publications_net[0:self.args.n_cores]
#                     else:
#                         ret_cores[i] = publications + datasets
#                         ret_cores[i] = ret_cores[i][0:self.args.n_cores]
#                         ret_net_cores[i] = publications_net + datasets_net
#                         ret_net_cores[i] = ret_net_cores[i][0:self.args.n_cores]
#             else:
#                 ret_cores[i] = datasets + publications
#                 ret_net_cores[i] = datasets_net + publications_net
#                 if self.is_dataset(seed):
#                     ret_cores[i] = publications + datasets
#                     ret_net_cores[i] = publications_net + datasets_net
#
#                 ret_cores[i] = ret_cores[i][:self.args.n_cores]
#                 ret_net_cores[i] = ret_net_cores[i][:self.args.n_cores]
#
#             ret_keys[i] = topics + entities + keywords
#             ret_net_keys[i] = topics_net + entities_net + keywords_net
#             if self.args.no_aug:
#                 ret_keys[i] = keywords
#                 ret_net_keys[i] = keywords_net
#             if self.args.nodes_to_remove == 'entity':
#                 ret_keys[i] = topics + keywords
#                 ret_net_keys[i] = topics_net + keywords_net
#             elif self.args.nodes_to_remove == 'topic':
#                 ret_keys[i] = entities + keywords
#                 ret_net_keys[i] = entities_net + keywords_net
#
#             ret_tops[i] = venues + organizations + authors
#             ret_net_tops[i] = venues_net + organizations_net + authors_net
#
#             if len(ret_cores[i]) < self.args.n_cores:
#                 pad = self.args.n_cores - len(ret_cores[i])
#                 for _ in range(pad):
#                     ret_cores[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.core_dim)])
#
#             elif len(ret_cores[i]) == 0:
#                 pad = self.args.n_cores
#                 for _ in range(pad):
#                     ret_cores[i].append(vectors[i])
#
#             if len(ret_net_cores[i]) < self.args.n_cores:
#                 pad = self.args.n_cores - len(ret_net_cores[i])
#                 for _ in range(pad):
#                     ret_net_cores[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.top_dim)])
#
#             elif len(ret_net_cores[i]) == 0:
#                 pad = self.args.n_cores
#                 for _ in range(pad):
#                     ret_net_cores[i].append(vectors_net[i])
#
#             if len(ret_tops[i]) < self.args.n_top_hubs:
#                 pad = self.args.n_top_hubs - len(ret_tops[i])
#                 for _ in range(pad):
#                     ret_tops[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.key_dim)])
#
#             elif len(ret_tops[i]) == 0:
#                 pad = self.args.n_top_hubs
#                 for _ in range(pad):
#                     ret_tops[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.key_dim)])
#
#             if len(ret_net_tops[i]) < self.args.n_top_hubs:
#                 pad = self.args.n_top_hubs - len(ret_net_tops[i])
#                 for _ in range(pad):
#                     ret_net_tops[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.top_dim)])
#
#             elif len(ret_net_tops[i]) == 0:
#                 pad = self.args.n_top_hubs
#                 for _ in range(pad):
#                     ret_net_tops[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.top_dim)])
#
#             if len(ret_keys[i]) < self.args.n_keys_hubs:
#                 pad = self.args.n_keys_hubs - len(ret_keys[i])
#                 for _ in range(pad):
#                     ret_keys[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.key_dim)])
#
#             elif len(ret_keys[i]) == 0:
#                 pad = self.args.n_keys_hubs
#                 for _ in range(pad):
#                     ret_keys[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.key_dim)])
#
#             if len(ret_net_keys[i]) < self.args.n_keys_hubs:
#                 pad = self.args.n_keys_hubs - len(ret_net_keys[i])
#                 for _ in range(pad):
#                     ret_net_keys[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.top_dim)])
#
#             elif len(ret_net_keys[i]) == 0:
#                 pad = self.args.n_keys_hubs
#                 for _ in range(pad):
#                     ret_net_keys[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.top_dim)])
#
#         return vectors, vectors_net, ret_net_cores, ret_cores, ret_net_keys, ret_keys, ret_net_tops, ret_tops, publications_vectors, datasets_vectors, authors_vectors, entities_vectors, topics_vectors, keywords_vectors, venues_vectors, organizations_vectors



import os.path
from args_list import get_args
from utils import create_graph_csv
import random
import networkx as nx
import walker
import multiprocessing as mp
from collections import Counter
import time
import pickle
import numpy as np
from loader import ScholarlyDataset
from sklearn.preprocessing import normalize
import pandas as pd
import torch
import numpy as np
import torch
import random
import multiprocessing
import concurrent.futures
from functools import partial
from torch_geometric import seed_everything

from multiprocessing import Pool
from collections import Counter
from functools import lru_cache


def seed_torch(seed=42):
    random.seed(seed)
    seed_everything(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)


seed_torch()


def cosine_similarity(emb1, emb2):
    dot_product = np.dot(emb1, emb2)
    norm_vector1 = np.linalg.norm(emb1)
    norm_vector2 = np.linalg.norm(emb2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity


# --------------------------------------------------------------------------- #
# WORKER-SIDE GLOBALS
#
# process_list() only ever touches: publications_features, datasets_features,
# args, and the type-predicate helpers (is_publication/is_dataset/...) plus,
# per-call, `walks` / `seed_to_start_idx`. It never touches self.G / self.G_conv.
#
# Previously select_walks_mp() called `self.executor.map(self.process_list, ...)`.
# Since self.process_list is a *bound method*, pickling it for dispatch to a
# worker process pickles the whole `self` object -- including self.G / self.G_conv
# (the full NetworkX graph) which is completely unused by process_list. This
# happened on *every single mini-batch*, and was almost certainly the dominant
# cost behind the ~30-40s/it seen in profiling.
#
# Fix: move the per-seed logic into a plain module-level function that reads
# from process-local globals. Those globals are populated exactly ONCE per
# worker process via the Pool's `initializer`, at pool-creation time -- not
# re-sent on every map() call. Only the genuinely per-call data (the current
# walks + which seeds to process) travels through the task queue each time.
#
# SECOND FIX (this revision): `walks` was still being re-sent in full to every
# chunk via functools.partial (a "fixed" argument of the partial is still
# pickled and shipped on every task dispatch -- partial does not make it
# worker-resident). select_walks_mp() now slices only the sub-range of `walks`
# that each chunk's seeds actually need before dispatching, so the amount of
# data shipped per mini-batch no longer scales with the *total* number of
# walks, only with the walks actually owned by that chunk.
#
# THIRD FIX (this revision): cosine similarity between a seed and a core
# neighbour depends only on the two *static* feature vectors (core_features
# never changes during a run), so the same (seed, neighbour) pair recomputes
# an identical dot product every single time it's seen across mini-batches
# and epochs. That is now cached per worker process with an LRU cache keyed
# on the node *names* (hashable, stable) instead of on tensor objects (whose
# identity changes every time a slice is taken).
# --------------------------------------------------------------------------- #

_WORKER_PUB_FEATS = None
_WORKER_DS_FEATS = None
_WORKER_ARGS = None
_WORKER_CORE_FEATURES = None

def _is_publication(node):
    return node.startswith('p_')


def _is_dataset(node):
    return node.startswith('d_')


def _is_entity(node):
    return node.startswith('dbpedia_')


def _is_keyword(node):
    return node.startswith('k_')


def _is_topic(node):
    return node.startswith('t')


def _is_author(node):
    return node.startswith('a_')


def _is_venue(node):
    return node.startswith('v_')


def _is_organization(node):
    return node.startswith('o_')


def _init_worker(pub_feats, ds_feats, args, core_features):
    """Runs once per worker process at Pool-creation time. Everything stored
    here stays resident in that worker for its whole lifetime -- it is never
    re-pickled/re-sent on subsequent map() calls."""
    global _WORKER_PUB_FEATS
    global _WORKER_DS_FEATS
    global _WORKER_ARGS
    global _WORKER_CORE_FEATURES

    _WORKER_PUB_FEATS = pub_feats
    _WORKER_DS_FEATS = ds_feats
    _WORKER_ARGS = args

    # Convert once, per worker, from torch tensors to plain numpy arrays.
    # - Avoids paying tensor->array conversion cost on every single dot
    #   product inside the hot loop of _process_list.
    # - Makes the values usable as stable inputs to a name-keyed cache below
    #   (the tensors themselves are not safely hashable/cacheable, since a
    #   fresh slice is a fresh object every time).
    _WORKER_CORE_FEATURES = {
        k: (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v))
        for k, v in core_features.items()
    }
    # Reset the cache for this (fresh) worker process/feature set.
    _cached_cosine.cache_clear()


@lru_cache(maxsize=4_000_000)
def _cached_cosine(seed_name, n_name):
    """Cosine-style dot product between two node embeddings, cached per
    worker process. Safe ONLY because core_features is static for the whole
    run (computed once in RandomWalkWithRestart.__init__ and never mutated
    afterwards) -- if that ever changes (e.g. embeddings start being updated
    by the model during training), this cache must be invalidated per call
    or removed."""
    a = _WORKER_CORE_FEATURES[seed_name]
    b = _WORKER_CORE_FEATURES[n_name]
    return float(np.dot(a, b))


def _process_list(seeds_list, walks, seed_to_start_idx, all_walks):
    """Module-level twin of RandomWalkWithRestart.process_list(), operating on
    worker-local globals instead of `self`. Logic is unchanged from the
    original method, except that the seed/core cosine similarity now goes
    through the per-worker cache instead of being recomputed from tensors
    every time."""
    args = _WORKER_ARGS
    selected_walks_list, selected_cores_list, selected_hubs_top_list, selected_hubs_key_list = [], [], [], []

    walk_importance_score = [[] for _ in range(len(seeds_list))]
    cores = [[] for _ in range(len(seeds_list))]
    hubs_top = [[] for _ in range(len(seeds_list))]
    hubs_key = [[] for _ in range(len(seeds_list))]

    for j, seed in enumerate(seeds_list):
        ind_seed = seed_to_start_idx[seed]
        cur_walks = walks[ind_seed: ind_seed + all_walks]
        for i, w in enumerate(cur_walks):
            walk = list(dict.fromkeys(w))
            freq = Counter(w)
            cores_walk = []
            hubs_key_walk = []
            hubs_top_walk = []
            neighs = walk[1:]
            cosine_total = 0
            neigh_count = 0
            for n in neighs:
                if n == seed:
                    neigh_count = 1
                else:
                    neigh_count += 1
                    frequency = freq[n]
                    distance = neigh_count
                    penalty = frequency / distance
                    if _is_publication(n) or _is_dataset(n):
                        # Cached: identical (seed, n) pairs recur constantly
                        # across mini-batches/epochs since core_features is
                        # static for the whole run.
                        cosine_0 = _cached_cosine(seed, n)
                        cosine = cosine_0 * penalty
                        cores_walk.append((n, cosine_0))

                        cosine_total += cosine
                    elif _is_author(n) or _is_organization(n) or _is_venue(n):
                        hubs_top_walk.append((n, penalty))
                    else:
                        hubs_key_walk.append((n, penalty))

            if len(cores_walk) == 0:
                cosine_total = 0
            else:
                cosine_total = cosine_total / len(cores_walk)

            walk_importance_score[j].append(cosine_total)
            cores[j].append(cores_walk)
            hubs_top[j].append(hubs_top_walk)
            hubs_key[j].append(hubs_key_walk)

        selected_walks = cur_walks  # fallback in case the loop below never runs
        sorted_walks_ind = sorted(
            range(len(walk_importance_score[j])),
            key=lambda i: walk_importance_score[j][i],
            reverse=True
        )

        for index in range(
                args.num_selected_walks,
                min(args.num_selected_walks +5, args.num_random_walks)
        ):
            selected_walks_ind = sorted_walks_ind[:index]
            selected_walks = [cur_walks[i] for i in selected_walks_ind]
            selected_cores = [cores[j][i] for i in selected_walks_ind]
            selected_cores = list(set(item for sublist in selected_cores for item in sublist if item != []))

            selected_hubs_top = [hubs_top[j][i] for i in selected_walks_ind]
            selected_hubs_top = list(set(item for sublist in selected_hubs_top for item in sublist if item != []))

            selected_hubs_key = [hubs_key[j][i] for i in selected_walks_ind]
            selected_hubs_key = list(set(item for sublist in selected_hubs_key for item in sublist if item != []))

            if args.test == False:
                unique_key_elements = {}
                for tup in selected_hubs_key:
                    if tup[0] not in unique_key_elements:
                        unique_key_elements[tup[0]] = tup[1]

                selected_hubs_key = [tuple([k, v]) for k, v in unique_key_elements.items()]
                unique_top_elements = {}
                for tup in selected_hubs_top:
                    if tup[0] not in unique_top_elements:
                        unique_top_elements[tup[0]] = tup[1]
                selected_hubs_top = [tuple([k, v]) for k, v in unique_top_elements.items()]

                unique_core_elements = {}
                for tup in selected_cores:
                    if tup[0] not in unique_core_elements:
                        unique_core_elements[tup[0]] = tup[1]
                selected_cores = [tuple([k, v]) for k, v in unique_core_elements.items()]

            if (len(selected_cores) >= args.n_cores and
                    len(selected_hubs_key) >= args.n_keys_hubs and
                    len(selected_hubs_top) >= args.n_top_hubs):
                break

        selected_walks_list.append(selected_walks)
        selected_cores_list.append(selected_cores)
        selected_hubs_top_list.append(selected_hubs_top)
        selected_hubs_key_list.append(selected_hubs_key)

    return [selected_walks_list, selected_cores_list, selected_hubs_top_list, selected_hubs_key_list]


class RandomWalkWithRestart:
    """
    args:
    - dataset
    - setup -- inductive light (il), inductive full (if), inductive hard (ih), transductive (t)
    - set -- train, validation, test
    - max number of paths per node -- the paths selected to choose neighbours from -- default 10
    - number of selected neighbours
    - number of core nodes
    - max path's length
    - seed_node
    - restart prob
    - nx_graph
    - distance penalty -- the penalty applied to distant nodes
    - reward frequency -- the reward applied to frequent nodes

    """

    def __init__(self, args, sch_dataset, root, set_type=None, test=False):
        # root = sch_dataset.root
        # sch_dataset = sch_dataset[0]
        self.args = args
        self.set_type = set_type
        self.path_data = f'datasets/{args.dataset}/'
        self.data = sch_dataset
        self.max_walk_length = args.max_walk_length
        self.all_walks = args.num_random_walks  # at the beginning compute 100 walks per node and store them, sampling will be random on these walks
        self.alpha = args.restart_probability

        self.G = create_graph_csv(self.path_data, self.args.dataset, ['all'])

        self.node_type = {}

        # Filter edges based on the selected source and target nodes
        if 'transductive' in root:
            # rimuovo gli archi tra p e d, li aggiungo dopo dipendentemente da cosa mi serve
            # qua sono in transduttivo e divido gli archi per il message passing, tenendo solo la kcore per il mp
            if 'train' in root:
                edge_index = sch_dataset['publication', 'cites', 'dataset'].edge_index_train
            elif 'validation' in root:
                edge_index = sch_dataset['publication', 'cites', 'dataset'].edge_index_validation
            elif 'test' in root:
                edge_index = sch_dataset['publication', 'cites', 'dataset'].edge_index_test
            else:
                exit(1)
            edges_to_remove = [(u, v) for u, v in self.G.edges if
                               (u.startswith('p_') and v.startswith('d_')) or (
                                       u.startswith('d_') and v.startswith('p'))]

            # rimuovo tutti gli edges tra p e d e aggiungo quelli che ho trovato sopra
            edges_to_add = [
                tuple([sch_dataset['publication'].rev_mapping[s[0]], sch_dataset['dataset'].rev_mapping[s[1]]]) for s in
                edge_index.t().tolist()]

            edges_to_remove_topic = []
            edges_to_remove_entity = []
            self.G.remove_edges_from(edges_to_remove)
            self.G.add_edges_from(edges_to_add)

            if args.nodes_to_remove == 'entity':
                edges_to_remove_entity = [(u, v) for u, v in self.G.edges if
                                          (u.startswith('p_') and v.startswith('db')) or (
                                                  u.startswith('d_') and v.startswith('db'))]
                edges_to_remove_entity.extend([(u, v) for u, v in self.G.edges if
                                               (u.startswith('p_') and v.startswith('e')) or (
                                                       u.startswith('d_') and v.startswith('e'))])
                entity = [u for u in self.G.nodes() if u.startswith('db')]
                self.removed = entity
                self.G.remove_nodes_from(self.removed)

            if args.nodes_to_remove == 'topic':
                edges_to_remove_topic = [(u, v) for u, v in self.G.edges if
                                         (u.startswith('p_') and v.startswith('t')) or (
                                                 u.startswith('d_') and v.startswith('t'))]
                topic = [u for u in self.G.nodes() if u.startswith('t')]
                self.removed = topic
                self.G.remove_nodes_from(self.removed)

            if args.nodes_to_remove == 'all':
                edges_to_remove_topic = [(u, v) for u, v in self.G.edges if
                                         (u.startswith('p_') and v.startswith('t')) or (
                                                 u.startswith('d_') and v.startswith('t'))]
                edges_to_remove_entity = [(u, v) for u, v in self.G.edges if
                                          (u.startswith('p_') and v.startswith('db')) or (
                                                  u.startswith('d_') and v.startswith('db'))]
                edges_to_remove_entity.extend([(u, v) for u, v in self.G.edges if
                                               (u.startswith('p_') and v.startswith('e')) or (
                                                       u.startswith('d_') and v.startswith('e'))])
                entity = [u for u in self.G.nodes() if u.startswith('db')]
                topic = [u for u in self.G.nodes() if u.startswith('t')]
                self.removed = entity + topic
                self.G.remove_nodes_from(self.removed)

            self.G.remove_edges_from(edges_to_remove_entity)
            self.G.remove_edges_from(edges_to_remove_topic)


            # edges_to_remove = [(u, v) for u, v in self.G.edges if v.startswith('tk')]
            # self.G.remove_edges_from(edges_to_remove)

        # tolgo pubblicazioni e dataset in ind e semi dal grafo e poi alleno
        edge_index_test_trans = sch_dataset['publication', 'cites', 'dataset'].edge_label_index_test_trans
        edge_index_test_semi = sch_dataset['publication', 'cites', 'dataset'].edge_label_index_test_semi
        edge_index_test_ind = sch_dataset['publication', 'cites', 'dataset'].edge_label_index_test_ind

        sources = list(set(edge_index_test_ind[0].tolist() + edge_index_test_semi[0].tolist()))
        targets = list(set(edge_index_test_ind[1].tolist()))

        print("nodo presente subito dopo create_graph_csv:", 'd_1948' in self.G.nodes())
        print("nodo presente subito dopo create_graph_csv:", 'd_15' in self.G.nodes())



        if not test:
            sources = [sch_dataset['publication'].rev_mapping[s] for s in sources]
            targets = [sch_dataset['dataset'].rev_mapping[s] for s in targets]
            self.removed = sources + targets
            self.G.remove_nodes_from(sources)
            self.G.remove_nodes_from(targets)

        else:
            if self.args.inductive_type == 'light':
                print('light')
                print("nodo presente subito dopo create_graph_csv:", 'd_15' in self.G.nodes())

                sources = list(set(edge_index_test_ind[0].tolist()))
                targets = list(set(edge_index_test_ind[1].tolist()))
                sources = [sch_dataset['publication'].rev_mapping[s] for s in sources]
                targets = [sch_dataset['dataset'].rev_mapping[s] for s in targets]

                self.removed = sources + targets
                print(len(self.G.edges))
                self.G.remove_edges_from(list(self.G.edges(self.removed)))
                print(len(self.G.edges))
                print("nodo presente subito dopo create_graph_csv:", 'd_15' in self.G.nodes())

                # self.G.remove_nodes_from(self.removed)
                # self.G.remove_nodes_from(targets)

            elif self.args.inductive_type == 'trans':
                print('trans')
                sources = list(set(edge_index_test_ind[0].tolist() + edge_index_test_semi[0].tolist()))
                targets = list(set(edge_index_test_ind[1].tolist()))
                sources = [sch_dataset['publication'].rev_mapping[s] for s in sources]
                targets = [sch_dataset['dataset'].rev_mapping[s] for s in targets]
                self.removed = sources + targets
                print(len(self.G.edges))
                self.G.remove_edges_from(list(self.G.edges(self.removed)))
                print(len(self.G.edges))

                # self.G.remove_nodes_from(self.removed)
                # self.G.remove_nodes_from(sources)
                # self.G.remove_nodes_from(targets)

        self.G_conv = nx.convert_node_labels_to_integers(self.G)
        self.mapping = {j: i for j, i in zip(self.G_conv, self.G)}
        self.rev_mapping = {i: j for j, i in zip(self.G_conv, self.G)}
        self.publications_features, self.datasets_features = sch_dataset['publication'], sch_dataset['dataset']
        self.publications_features.x = torch.from_numpy(
            normalize(self.publications_features.x, axis=1)
        ).float()

        self.datasets_features.x = torch.from_numpy(
            normalize(self.datasets_features.x, axis=1)
        ).float()
        print("nodo presente subito dopo create_graph_csv:", 'd_15' in self.G.nodes())

        self.core_features = {}

        for k, v in self.publications_features['mapping'].items():
            self.core_features[k] = self.publications_features.x[v]

        for k, v in self.datasets_features['mapping'].items():
            self.core_features[k] = self.datasets_features.x[v]

        # self.publications_features,self.datasets_features = self.load_features()

        sources = list(set(edge_index_test_ind[0].tolist()))
        targets = list(set(edge_index_test_ind[1].tolist()))
        sources = [sch_dataset['publication'].rev_mapping[s] for s in sources]
        targets = [sch_dataset['dataset'].rev_mapping[s] for s in targets]
        print(f'intersection with ind {len(list(set(sources).intersection(set(self.G))))}')
        print(f'intersection with ind {len(list(set(targets).intersection(self.G)))}')

        sources = list(set(edge_index_test_semi[0].tolist()))
        targets = list(set(edge_index_test_semi[1].tolist()))
        sources = [sch_dataset['publication'].rev_mapping[s] for s in sources]
        targets = [sch_dataset['dataset'].rev_mapping[s] for s in targets]
        print(f'intersection with semi {len(list(set(sources).intersection(set(self.G))))}')
        print(f'intersection with semi {len(list(set(targets).intersection(self.G)))}')
        print(f'length datasets semi {len(list(set(targets)))}')

        sources = list(set(edge_index_test_trans[0].tolist()))
        targets = list(set(edge_index_test_trans[1].tolist()))
        sources = [sch_dataset['publication'].rev_mapping[s] for s in sources]
        targets = [sch_dataset['dataset'].rev_mapping[s] for s in targets]
        print(f'intersection with trans {len(list(set(sources).intersection(set(self.G))))}')
        print(f'intersection with trans {len(list(set(targets).intersection(self.G)))}')
        print(f'length datasets trans {len(list(set(targets)))}')

        # Persistent worker pool: created ONCE per walker instance instead of being
        # spawned/torn down on every call to select_walks_mp (i.e. every mini-batch,
        # every epoch). Re-creating a 40-worker pool that many times was one of the
        # biggest sources of overhead in the training loop.
        #
        # `initializer`/`initargs` push the (constant, per-run) feature tensors and
        # args into each worker EXACTLY ONCE, at pool start-up. This is what lets
        # select_walks_mp() below avoid re-pickling self.G / self.G_conv / features
        # on every mini-batch -- previously the single biggest source of overhead.
        self.n_workers = 10
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.n_workers,
            initializer=_init_worker,
            initargs=(
                self.publications_features,
                self.datasets_features,
                self.args,
                self.core_features,
            ),
        )

    def get_seeds(self):
        return self.seeds

    def reset_seeds(self):
        self.seeds = []

    def __getstate__(self):
        state = self.__dict__.copy()
        # ProcessPoolExecutor holds locks/threads - never picklable. Kept as a
        # safety net; select_walks_mp() no longer sends `self` to workers at all,
        # but this protects against anything else in the codebase pickling the
        # walker instance (e.g. checkpointing).
        state.pop('executor', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.executor = None

    def close(self):
        """Call this once training/inference is done to shut down the pool and
        avoid leaking 40 worker processes for the lifetime of the run."""
        if getattr(self, 'executor', None) is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def create_random_walks(self, seeds_in=False, all=False):
        if all:
            seeds = [node for node in self.G.nodes() if self.is_publication(node) or self.is_dataset(node)]
            self.seeds = seeds
        else:
            if not seeds_in:
                seeds = self.seeds
            else:
                self.seeds = seeds_in
                seeds = seeds_in

        if seeds is not None:
            converted_seeds = [self.rev_mapping[x] for x in self.seeds]
            walks = walker.random_walks(self.G_conv, n_walks=self.all_walks, alpha=self.alpha,
                                        walk_len=self.max_walk_length, start_nodes=converted_seeds, q=0.3, p=1.0)
        else:
            walks = walker.random_walks(self.G_conv, n_walks=self.all_walks, alpha=self.alpha,
                                        walk_len=random.randint(1, self.max_walk_length), p=1.0, q=0.1)
        walks = sorted(walks, key=lambda x: x[0])
        walks_rev = []
        for walk in walks:
            walks_rev.append([self.mapping[p] for p in walk])

        return walks_rev

    def set_seeds(self, seeds=None):
        if seeds is None:
            seeds = [node for node in self.G.nodes() if self.is_publication(node) or self.is_dataset(node)]
            self.seeds = seeds
        else:
            self.seeds = seeds

    def is_publication(self, node):
        return node.startswith('p_')

    def is_dataset(self, node):
        return node.startswith('d_')

    def is_entity(self, node):
        return node.startswith('dbpedia_')

    def is_keyword(self, node):
        return node.startswith('k_')

    def is_topic(self, node):
        return node.startswith('t')

    def is_author(self, node):
        return node.startswith('a_')

    def is_venue(self, node):
        return node.startswith('v_')

    def is_organization(self, node):
        return node.startswith('o_')

    def get_walks_threshold(self, walk_scores):
        return np.percentile(walk_scores, 75)

    def select_walks_mp(self, walks):
        """Select the walks such that
            - the mean cos sim is maximized
            - min num neighbours is reached

        Only the walks each chunk's seeds actually need, plus the seed
        sublists themselves, are shipped to the (already-running) worker
        pool per call. `walks` used to be bound as a fixed `functools.partial`
        argument shared by every chunk -- but a "fixed" partial argument is
        still re-pickled and re-sent on every single dispatched task, so the
        *entire* walks array (O(n_seeds_total)) was being shipped once per
        chunk instead of once total. Now each chunk gets only the slice of
        walks belonging to its own seeds, so the data shipped per mini-batch
        scales with the chunk's own seeds, not with every seed in the batch.
        """

        def chunk_list(main_list, chunk_size):
            for i in range(0, len(main_list), chunk_size):
                yield main_list[i:i + chunk_size]

        selected_seeds_cores = [[] for _ in range(len(self.seeds))]
        selected_seeds_hubs_top = [[] for _ in range(len(self.seeds))]
        selected_seeds_hubs_key = [[] for _ in range(len(self.seeds))]
        selected_seeds_walks = [[] for _ in range(len(self.seeds))]

        first_ind = [w[0] for w in walks]
        self.first_ind = first_ind
        self.walks = walks

        # O(len(walks)) precompute of seed -> starting index, instead of the
        # O(seeds * walks) `list.index()` scan the non-mp select_walks() still does.
        seed_to_start_idx = {}
        for idx, s in enumerate(first_ind):
            if s not in seed_to_start_idx:
                seed_to_start_idx[s] = idx
        self.seed_to_start_idx = seed_to_start_idx

        # Balance chunk size against the number of live workers instead of a
        # fixed 2000: a fixed size can leave most of the pool idle when
        # len(seeds) is small, or create very unbalanced chunks when it's
        # large relative to n_workers.
        n_workers = getattr(self, 'n_workers', self.executor._max_workers)
        chunk_size = max(1, -(-len(self.seeds) // n_workers))
        sublists = list(chunk_list(self.seeds, chunk_size))

        # Build, per chunk, only the walk rows that chunk's own seeds need --
        # this is the actual per-mini-batch payload now, instead of the full
        # `walks` array being reshipped to every chunk.
        tasks_seeds, tasks_walks, tasks_idx = [], [], []
        for sub in sublists:
            local_walks = []
            local_idx = {}
            offset = 0
            for s in sub:
                st = seed_to_start_idx[s]
                block = walks[st: st + self.all_walks]
                local_idx[s] = offset
                local_walks.extend(block)
                offset += len(block)
            tasks_seeds.append(sub)
            tasks_walks.append(local_walks)
            tasks_idx.append(local_idx)

        worker_fn = partial(_process_list, all_walks=self.all_walks)
        try:
            t0 = time.time()

            results = list(self.executor.map(worker_fn, tasks_seeds, tasks_walks, tasks_idx))

            print(
                "executor time:",
                time.time() - t0
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e
        j = 0
        for result in results:
            walk_list, core_list, hubtop_list, hubkey_list = result[0], result[1], result[2], result[3]
            for walk, core, hubtop, hubkey in zip(walk_list, core_list, hubtop_list, hubkey_list):
                selected_seeds_walks[j] = walk
                selected_seeds_cores[j] = core
                selected_seeds_hubs_top[j] = hubtop
                selected_seeds_hubs_key[j] = hubkey
                j += 1

        # Dedup-then-sort in a single pass (previously: sort, dedup via dict,
        # sort again -- the same list was sorted twice).
        def _dedup_sorted(pairs, limit=None):
            best = {}
            for k, v in pairs:
                # keep the highest-value entry per key
                if k not in best or v > best[k][1]:
                    best[k] = (k, v)
            out = sorted(best.values(), key=lambda p: (p[1], p[0]), reverse=True)
            return out[:limit] if limit is not None else out

        final_cores = [_dedup_sorted(s) for s in selected_seeds_cores]
        final_hubs_top = [_dedup_sorted(s, self.args.n_top_hubs) for s in selected_seeds_hubs_top]
        final_hubs_key = [_dedup_sorted(s, self.args.n_keys_hubs) for s in selected_seeds_hubs_key]

        return selected_seeds_walks, final_cores, final_hubs_key, final_hubs_top


    def get_neighbours_vector(self, sel_cores, sel_keys, sel_top):
        """Builds per-seed feature vectors for cores/hubs.

        Previously each of `cores` / `tops` / `keys` was scanned multiple
        times with near-duplicate type filters (e.g. is_publication(c[0])
        applied once to build `publications` and again, separately, to build
        `publications_net`). This does a single pass per list, bucketing into
        both the "plain" and "net" feature variants together.
        """

        publications_vectors = [[] for _ in range(len(self.seeds))]
        publications_net_vectors = [[] for _ in range(len(self.seeds))]
        datasets_vectors = [[] for _ in range(len(self.seeds))]
        datasets_net_vectors = [[] for _ in range(len(self.seeds))]
        authors_vectors = [[] for _ in range(len(self.seeds))]
        venues_vectors = [[] for _ in range(len(self.seeds))]
        keywords_vectors = [[] for _ in range(len(self.seeds))]
        organizations_vectors = [[] for _ in range(len(self.seeds))]
        entities_vectors = [[] for _ in range(len(self.seeds))]
        topics_vectors = [[] for _ in range(len(self.seeds))]

        ret_cores = [[] for _ in range(len(self.seeds))]
        ret_net_cores = [[] for _ in range(len(self.seeds))]
        ret_tops = [[] for _ in range(len(self.seeds))]
        ret_net_tops = [[] for _ in range(len(self.seeds))]
        ret_keys = [[] for _ in range(len(self.seeds))]
        ret_net_keys = [[] for _ in range(len(self.seeds))]
        vectors = []
        vectors_net = []

        # Lazy, cached-on-first-use accessors: `publication`/`dataset` are always
        # present (used unconditionally below), but venue/organization/entity/
        # topic/keyword/author storages may not have `mapping`/`x`/`net_x`
        # populated at all in some dataset variants (e.g. that node type is
        # simply absent for this run). The original code only ever touched
        # these attributes *inside* a filter that matched at least one node of
        # that type, so it never crashed on unused, unpopulated storages.
        # Prefetching them unconditionally at the top of the function (as a
        # previous revision did) broke that and could raise AttributeError
        # even when the type is never actually encountered. This restores the
        # lazy behaviour, while still caching after a successful first access
        # so repeated seeds don't redo the attribute lookup.
        _type_attrs_cache = {}

        def _type_attrs(key):
            cached = _type_attrs_cache.get(key)
            if cached is None:
                storage = self.data[key]
                cached = (storage.mapping, storage.x, storage.net_x)
                _type_attrs_cache[key] = cached
            return cached

        pub_map, pub_x, pub_net_x = self.data['publication'].mapping, self.data['publication'].x, self.data['publication'].net_x
        ds_map, ds_x, ds_net_x = self.data['dataset'].mapping, self.data['dataset'].x, self.data['dataset'].net_x

        for i, seed in enumerate(self.seeds):
            if self.is_publication(seed):
                idx = pub_map[seed]
                vectors.append([seed, seed, 1.0, idx, pub_x[idx]])
                vectors_net.append([seed, seed, 1.0, idx, pub_net_x[idx]])
            elif self.is_dataset(seed):
                idx = ds_map[seed]
                vectors.append([seed, seed, 1.0, idx, ds_x[idx]])
                vectors_net.append([seed, seed, 1.0, idx, ds_net_x[idx]])

            # --- cores: single pass building both plain and net variants ---
            publications, publications_net = [], []
            datasets, datasets_net = [], []
            for c in sel_cores[i]:
                name, weight = c[0], c[1]
                if self.is_publication(name):
                    idx = pub_map[name]
                    publications.append([seed, name, weight, idx, pub_x[idx]])
                    publications_net.append([seed, name, weight, idx, pub_net_x[idx]])
                elif self.is_dataset(name):
                    idx = ds_map[name]
                    datasets.append([seed, name, weight, idx, ds_x[idx]])
                    datasets_net.append([seed, name, weight, idx, ds_net_x[idx]])

            # --- hubs_top: authors / venues / organizations, single pass ---
            authors, authors_net = [], []
            venues, venues_net = [], []
            organizations, organizations_net = [], []
            for c in sel_top[i]:
                name, weight = c[0], c[1]
                if self.is_author(name):
                    auth_map, auth_x, auth_net_x = _type_attrs('author')
                    idx = auth_map[name]
                    authors.append([seed, name, weight, idx, auth_x[idx]])
                    authors_net.append([seed, name, weight, idx, auth_net_x[idx]])
                elif self.is_venue(name):
                    venue_map, venue_x, venue_net_x = _type_attrs('venue')
                    idx = venue_map[name]
                    venues.append([seed, name, weight, idx, venue_x[idx]])
                    venues_net.append([seed, name, weight, idx, venue_net_x[idx]])
                elif self.is_organization(name):
                    org_map, org_x, org_net_x = _type_attrs('organization')
                    idx = org_map[name]
                    organizations.append([seed, name, weight, idx, org_x[idx]])
                    organizations_net.append([seed, name, weight, idx, org_net_x[idx]])

            # --- hubs_key: keywords / entities / topics, single pass ---
            keywords, keywords_net = [], []
            entities, entities_net = [], []
            topics, topics_net = [], []
            for c in sel_keys[i]:
                name, weight = c[0], c[1]
                if self.is_keyword(name):
                    kw_map, kw_x, kw_net_x = _type_attrs('keyword')
                    idx = kw_map[name]
                    keywords.append([seed, name, weight, idx, kw_x[idx]])
                    keywords_net.append([seed, name, weight, idx, kw_net_x[idx]])
                elif self.is_entity(name):
                    ent_map, ent_x, ent_net_x = _type_attrs('entity')
                    idx = ent_map[name]
                    entities.append([seed, name, weight, idx, ent_x[idx]])
                    entities_net.append([seed, name, weight, idx, ent_net_x[idx]])
                elif self.is_topic(name):
                    topic_map, topic_x, topic_net_x = _type_attrs('topic')
                    idx = topic_map[name]
                    topics.append([seed, name, weight, idx, topic_x[idx]])
                    topics_net.append([seed, name, weight, idx, topic_net_x[idx]])

            publications_vectors[i] = publications
            publications_net_vectors[i] = publications_net
            datasets_net_vectors[i] = datasets_net
            datasets_vectors[i] = datasets
            entities_vectors[i] = entities
            topics_vectors[i] = topics
            authors_vectors[i] = authors
            venues_vectors[i] = venues
            organizations_vectors[i] = organizations
            keywords_vectors[i] = keywords

            if self.args.split_cores:
                if self.is_publication(seed):
                    if len(datasets) >= self.args.n_cores:
                        ret_cores[i] = datasets[0:self.args.n_cores]
                        ret_net_cores[i] = datasets_net[0:self.args.n_cores]
                    else:
                        ret_cores[i] = datasets + publications
                        ret_cores[i] = ret_cores[i][0:self.args.n_cores]
                        ret_net_cores[i] = datasets_net + publications_net
                        ret_net_cores[i] = ret_net_cores[i][0:self.args.n_cores]
                elif self.is_dataset(seed):
                    if len(publications) >= self.args.n_cores:
                        ret_cores[i] = publications[0:self.args.n_cores]
                        ret_net_cores[i] = publications_net[0:self.args.n_cores]
                    else:
                        ret_cores[i] = publications + datasets
                        ret_cores[i] = ret_cores[i][0:self.args.n_cores]
                        ret_net_cores[i] = publications_net + datasets_net
                        ret_net_cores[i] = ret_net_cores[i][0:self.args.n_cores]
            else:
                ret_cores[i] = datasets + publications
                ret_net_cores[i] = datasets_net + publications_net
                if self.is_dataset(seed):
                    ret_cores[i] = publications + datasets
                    ret_net_cores[i] = publications_net + datasets_net

                ret_cores[i] = ret_cores[i][:self.args.n_cores]
                ret_net_cores[i] = ret_net_cores[i][:self.args.n_cores]

            ret_keys[i] = topics + entities + keywords
            ret_net_keys[i] = topics_net + entities_net + keywords_net
            if self.args.no_aug:
                ret_keys[i] = keywords
                ret_net_keys[i] = keywords_net
            if self.args.nodes_to_remove == 'entity':
                ret_keys[i] = topics + keywords
                ret_net_keys[i] = topics_net + keywords_net
            elif self.args.nodes_to_remove == 'topic':
                ret_keys[i] = entities + keywords
                ret_net_keys[i] = entities_net + keywords_net

            ret_tops[i] = venues + organizations + authors
            ret_net_tops[i] = venues_net + organizations_net + authors_net

            if len(ret_cores[i]) < self.args.n_cores:
                pad = self.args.n_cores - len(ret_cores[i])
                for _ in range(pad):
                    ret_cores[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.core_dim)])

            elif len(ret_cores[i]) == 0:
                pad = self.args.n_cores
                for _ in range(pad):
                    ret_cores[i].append(vectors[i])

            if len(ret_net_cores[i]) < self.args.n_cores:
                pad = self.args.n_cores - len(ret_net_cores[i])
                for _ in range(pad):
                    ret_net_cores[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.top_dim)])

            elif len(ret_net_cores[i]) == 0:
                pad = self.args.n_cores
                for _ in range(pad):
                    ret_net_cores[i].append(vectors_net[i])

            if len(ret_tops[i]) < self.args.n_top_hubs:
                pad = self.args.n_top_hubs - len(ret_tops[i])
                for _ in range(pad):
                    ret_tops[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.key_dim)])

            elif len(ret_tops[i]) == 0:
                pad = self.args.n_top_hubs
                for _ in range(pad):
                    ret_tops[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.key_dim)])

            if len(ret_net_tops[i]) < self.args.n_top_hubs:
                pad = self.args.n_top_hubs - len(ret_net_tops[i])
                for _ in range(pad):
                    ret_net_tops[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.top_dim)])

            elif len(ret_net_tops[i]) == 0:
                pad = self.args.n_top_hubs
                for _ in range(pad):
                    ret_net_tops[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.top_dim)])

            if len(ret_keys[i]) < self.args.n_keys_hubs:
                pad = self.args.n_keys_hubs - len(ret_keys[i])
                for _ in range(pad):
                    ret_keys[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.key_dim)])

            elif len(ret_keys[i]) == 0:
                pad = self.args.n_keys_hubs
                for _ in range(pad):
                    ret_keys[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.key_dim)])

            if len(ret_net_keys[i]) < self.args.n_keys_hubs:
                pad = self.args.n_keys_hubs - len(ret_net_keys[i])
                for _ in range(pad):
                    ret_net_keys[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.top_dim)])

            elif len(ret_net_keys[i]) == 0:
                pad = self.args.n_keys_hubs
                for _ in range(pad):
                    ret_net_keys[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.top_dim)])

        return vectors, vectors_net, ret_net_cores, ret_cores, ret_net_keys, ret_keys, ret_net_tops, ret_tops, publications_vectors, datasets_vectors, authors_vectors, entities_vectors, topics_vectors, keywords_vectors, venues_vectors, organizations_vectors