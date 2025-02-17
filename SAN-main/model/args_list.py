import argparse

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-dataset',default='mes',type=str,choices=['mes','pubmed_kcore','pubmed'],help='The dataset selected')
	parser.add_argument('-split_type',default='t',type=str,choices=['t','il','if','ih'],help='type of split (select between transductive, inductive light, inductive full, inductive hard')
	parser.add_argument('-path_data',default='datasets/mes/all/final',type=str)
	parser.add_argument('-path_train_data',default='datasets/mes/split_transductive/train',type=str)
	parser.add_argument('-path_vali_data',default='datasets/mes/split_transductive/validation',type=str)
	parser.add_argument('-path_test_data',default='datasets/mes/split_transductive/test',type=str)
	parser.add_argument('-core_sentence_transformer',default='all-MiniLM-L6-v2',type=str,help='The pretrained model to encode core nodes (publications and datasets)')
	parser.add_argument('-key_sentence_transformer',default='whaleloops/phrase-bert',type=str,help='The pretrained model to encode core nodes (publications and datasets)')
	parser.add_argument("-verbose", action="store_true", help="Enable verbose mode.")
	parser.add_argument("-verbose_sampler", action="store_true", help="Enable verbose mode.")
	parser.add_argument("-split_cores", action="store_true", help="Split cores.")
	parser.add_argument('-patience', type=int, default=50, help='Patience. Default is 5.')
	parser.add_argument('-checkpoint', type=str, default='', help='path to checkpoint')
	parser.add_argument('-restart', action="store_true", help="training restart")
	parser.add_argument('-trans', type=float, help="transductive training split ratio,default is 0.8")
	parser.add_argument('-inductive', action='store_true', help="transductive training split ratio,default is 0.8")
	parser.add_argument('-enriched',default='enriched_all', help="consider 4 embeddings concat")
	parser.add_argument('-silent', help="whether to silent the vector")
	parser.add_argument('-iteration',default=0, help="in bootstrap for pubmed data")
	parser.add_argument('-no_metadata',action='store_true', help="whether to silent the vector")
	parser.add_argument('-hetgnn',action='store_true', help="whether to silent the vector")
	parser.add_argument('-bootstrap',type=int, help="whether to silent the vector")
	parser.add_argument('-split',type=float, help="split incomplete metadata",default=0)
	parser.add_argument('-inductive_type', type=str,choices=['light','full'], help="transductive training split ratio,default is 0.8")


	parser.add_argument('-n_cores', type=int, default=5, help='core set size')
	parser.add_argument('-n_keys_hubs', type=int, default=5, help='batch size')
	parser.add_argument('-n_top_hubs', type=int, default=5, help='batch size')

	parser.add_argument('-core_dim', type=int, default=384, help='embedding dimension of core vectors - default 384 dim pretrained embeddings')
	parser.add_argument('-key_dim', type=int, default=768, help='embedding dimension of key vectors - default 768 dim pretrained embeddings')
	parser.add_argument('-top_dim', type=int, default=128, help='embedding dimension of top hub vectors - default 128 dim pretrained embeddings')
	parser.add_argument('-embedding_dim', type=int, default=128, help='final embedding dimension')

	# learning params
	parser.add_argument('-epochs',default=100,type=int,help='The number of epochs used for training the model')
	parser.add_argument('-lr',default=1e-5,type=float,help='The learning rate')
	parser.add_argument('-wd',default=1e-5,type=float,help='weight decay')
	parser.add_argument('-random_seed',default=42,type=int,help='Starting seed')
	parser.add_argument('-batch_size', type=int, default=1024, help='batch size')
	parser.add_argument('-lstm_layers', type=int, default=1, help='lstm layers')
	parser.add_argument('-gru_layers', type=int, default=1, help='gru layers')
	parser.add_argument('-topk', type=int, default=10, help='topk for recommendation downstream task')
	parser.add_argument('-train', action="store_true",help='train the model')
	parser.add_argument('-test', action="store_true",help='test the model')
	parser.add_argument('-lp', action="store_true",help='link prediction results')
	parser.add_argument('-rec', action="store_true",help='recommendation results')
	parser.add_argument('-heads', default=8, type=int,help='attention heads')
	parser.add_argument('-path', type=str,help='the path of the model to test heads')
	parser.add_argument('-f1_threshold', type=float,help='the path of the model to test heads',default=0.5)
	parser.add_argument('-loss', type=str,help='the loss function to minimize',default="crossentropy",choices=["crossentropy","kl","binarycross","hinge"])

	# sampling params
	parser.add_argument('-max_walk_length',default=15,type=int,help='The maximum number of nodes in a single path')
	parser.add_argument('-num_random_walks',default=20,type=float,help='number of walks sampled per node')
	parser.add_argument('-num_selected_walks',default=5,type=float,help='number of walks considered to select neighbors. Must be lower or equal num_random_walks')
	parser.add_argument('-restart_probability',default=0.1,type=float,help='Restart probability')

	# sampling params
	parser.add_argument('-all_aggregation',default='concat',type=str,choices=['mean','concat','lstm','mh-attention','cross-attention','cross-attention_1'],help='all nodes aggregation')
	parser.add_argument('-core_aggregation',default='mh-attention',choices=['cross-attention','lstm','attention','mean_pooling','mh-attention','linear'],type=str,help='core based nodes aggregation')
	parser.add_argument('-key_aggregation',default='mh-attention',type=str,choices=['cross-attention','lstm','attention','mh-attention','linear','cross-attention'],help='keyword based nodes aggregation')
	parser.add_argument('-top_aggregation',default='mh-attention',type=str,choices=['cross-attention','lstm','attention','mh-attention','linear','cross-attention'],help='topology based nodes aggregation')
	parser.add_argument('-keep_core',action="store_true", help="keep core vectors")
	parser.add_argument('-keep_core_key',action="store_true", help="keep core vectors")
	parser.add_argument('-keep_all',action="store_true", help="keep top vectors")
	parser.add_argument('-keep_key',action="store_true", help="keep top vectors")
	parser.add_argument('-keep_tpp',action="store_true", help="keep top vectors")
	parser.add_argument('-rewrite',action="store_true", help="rewrite the paths")

	# training
	parser.add_argument('-eval_batch',action="store_true", help="keep top vectors")
	parser.add_argument('-eval_lr',action="store_true", help="keep top vectors")
	parser.add_argument('-eval_heads',action="store_true", help="keep top vectors")
	parser.add_argument('-eval_neigh',action="store_true", help="keep top vectors")
	parser.add_argument('-eval_aggregation',action="store_true", help="keep top vectors")
	parser.add_argument('-eval_combine_aggregation',action="store_true", help="keep top vectors")
	parser.add_argument('-eval_all',action="store_true", help="keep top vectors")

	args = parser.parse_args()

	return args
