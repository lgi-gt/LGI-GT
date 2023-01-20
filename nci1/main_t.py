
import argparse

parser = argparse.ArgumentParser(description="model training and evaluating") 

parser.add_argument('--info', type=str, default=None) 

# dataset settings 
parser.add_argument('--dataset', type=str, default="NCI1", choices=["PROTEINS", "COLLAB", "PROTEINS_full", 'ogbg-molbbbp']) 
parser.add_argument('--use-node-attr', action='store_true', dest='use_node_attr') 
parser.add_argument('--no-node-attr', action='store_false', dest='use_node_attr') 
parser.set_defaults(use_node_attr=False) 
parser.add_argument('--pre-transform', type=str, default=None, choices=["graph_partition_greedy_modularity", "graph_partition_girvan_newman"]) 
parser.add_argument('--transform', type=str, default="None", choices=["NormalizeFeatures"]) 
parser.add_argument('--clean-cache', action='store_true')
parser.add_argument('--use-public-splits', action='store_true', dest='use_public_splits') 
parser.add_argument('--no-public-splits', action='store_false', dest='use_public_splits') 
parser.set_defaults(use_public_splits=True) # for constant or fixed fold split in 10-fold CV 

# training settings 
parser.add_argument('--use-val', action="store_true", dest='use_val', 
    help="if set then use validation set for choosing test accuracy") 
parser.add_argument('--no-val', action="store_false", dest='use_val', 
    help="if set then directly use test set for choosing test accuracy") 
parser.set_defaults(use_val=True) 

parser.add_argument('--use-val-loss', action="store_true", dest='use_val_loss', 
    help="if set then use validation loss for choosing test accuracy; work only when use_val is True") 
parser.add_argument('--no-val-loss', action="store_false", dest='use_val_loss', 
    help="if set then use validation acc for choosing test accuracy; work only when use_val is True") 
parser.set_defaults(use_val_loss=True) 

parser.add_argument('--same-best-epoch', action='store_true', dest='same_best_epoch', 
    help="if set then take the single same epoch with best average test accuracy of 10-fold") 
parser.add_argument('--no-same-best-epoch', action='store_false', dest='same_best_epoch', 
    help="if set then take the average of best test accuracy of every fold"
        "work only when use_val is False") 
parser.set_defaults(same_best_epoch=False) 

parser.add_argument('--early-stop', action='store_true', dest='early_stop', 
    help="trigger early stop on or off") 
parser.add_argument('--no-early-stop', action='store_false', dest='early_stop', 
    help='turn off early stop') 
parser.set_defaults(early_stop=False) 

# parser.add_argument('--epochs', type=int, default=300) 
# parser.add_argument('--patience', type=int, default=500) 
# parser.add_argument('--batch-size', type=int, default=64) 
# parser.add_argument('--lr', type=float, default=0.0001) 
# parser.add_argument('--weight-decay', type=float, default=0.) 
# parser.add_argument('--lr-decay-factor', type=float, default=0.5) 
# parser.add_argument('--lr-decay-step', type=int, default=50) 

parser.add_argument('--epochs', type=int, default=100) 
parser.add_argument('--scheduler', type=str, default='cosine') 
parser.add_argument('--warmup', type=int, default=0) 
parser.add_argument('--patience', type=int, default=25) 
parser.add_argument('--batch-size', type=int, default=128) 
parser.add_argument('--lr', type=float, default=0.0001) 
parser.add_argument('--weight-decay', type=float, default=0.0001) 

parser.add_argument('--clustering', action="store_true", dest='clustering')  
parser.add_argument('--no-clustering', action="store_false", dest='clustering')  
parser.set_defaults(clustering=True) 

parser.add_argument('--tlayer-dropout', type=float, default=0.1) 

parser.add_argument('--seeds', type=int, default=12344) 
# 42, 2333, 23333, 12138, 666, 886, 314159, 271828, 2020, 2022 

parser.add_argument('--device', type=int, choices=[0, 1, 2, 3], default=0) 

parser.add_argument('--save-state', action='store_true') 

# model settings 
# parser.add_argument('--config', type=str, default="configs/config.yaml") 


args = parser.parse_args() 


import sys 
# sys.path.append("..") 

import os 
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':16:8' 
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device) 
os.environ['PYTHONHASHSEED'] = str(args.seeds) 

import random 
import numpy as np 

import torch 

from train_evaluate import cross_validation_gt, cross_validation, record_basic_info, \
                            hold_out, hold_out_gt  

def set_random_seed(seed): 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 

#################################################################### 

set_random_seed(args.seeds) 

args.cmd_str = ' '.join(sys.argv) 

# dataset = load_dataset(args.dataset, 
#             pre_transform=args.pre_transform, 
#             transform=args.transform, 
#             use_node_attr=args.use_node_attr, 
#             clean_cache=args.clean_cache) 

from torch_geometric.datasets import TUDataset 
from torch_geometric.data import Data 
def add_complete_edge_index(data: Data): 
    n = data.num_nodes 
    index = torch.arange(n) 
    data.complete_edge_index = torch.vstack((index.repeat_interleave(n), index.repeat(n))) 
    return data 


from t import GraphTransformer 
from gt_nci1_diy import GraphTransformer_DIY  

result_logs = [] 

for run in range(20): 
    dataset = TUDataset('.', 'NCI1') #, transform=add_complete_edge_index) 
    model = GraphTransformer( 
                dataset.num_features, 128, dataset.num_classes, 
                layer_sequence=['SelfAtt', 'SelfAtt', 'SelfAtt'], #'SelfAtt', 'SelfAtt'], 
                gconv_dropout=0., 
                between_dropout=0., 
                tlayer_dropout=args.tlayer_dropout, 
                clustering=args.clustering, 
                layer_norm=True, 
                readout='mean') 
    # model = GraphTransformer_DIY( 
    #             dataset.num_features, 128, dataset.num_classes, 
    #             gconv_dropout=0., 
    #             between_dropout=0., 
    #             tlayer_dropout=0.1, 
    #             layer_norm=True, 
    #             readout='mean') 

    # cross_validation(model, dataset, args) 
    result_logs.append(hold_out_gt(model, dataset, args) ) 
    # hold_out(model, dataset, args) 

import time 
current_time = time.localtime() 
current_time_log = time.strftime("%Y-%m-%d %H:%M:%S", current_time) 

runs_result_fp = open('exp_result/NCI1-20runs.txt', 'a') 

record_basic_info(runs_result_fp, current_time_log, args) 
for log in result_logs: 
    runs_result_fp.write(log) 
runs_result_fp.close() 
