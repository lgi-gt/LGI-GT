
import argparse

parser = argparse.ArgumentParser(description="model training and evaluating") 

parser.add_argument('--info', type=str, default=None) 

# dataset settings 
parser.add_argument('--dataset', type=str, default="ogbg-molbbbp", choices=["PROTEINS", "COLLAB", "PROTEINS_full", 'ogbg-molbbbp']) 
parser.add_argument('--use-node-attr', action='store_true', dest='use_node_attr') 
parser.add_argument('--no-node-attr', action='store_false', dest='use_node_attr') 
parser.set_defaults(use_node_attr=False) 
parser.add_argument('--pre-transform', type=str, default=None, choices=["graph_partition_greedy_modularity", "graph_partition_girvan_newman"]) 
parser.add_argument('--transform', type=str, default=None, choices=["NormalizeFeatures"]) 
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
parser.add_argument('--patience', type=int, default=0) 

# parser.add_argument('--epochs', type=int, default=200) 
# parser.add_argument('--batch-size', type=int, default=32) 
# parser.add_argument('--lr', type=float, default=0.001) 
# parser.add_argument('--weight-decay', type=float, default=0.) 
# parser.add_argument('--lr-decay-factor', type=float, default=0.5) 
# parser.add_argument('--lr-decay-step', type=int, default=50) 

parser.add_argument('--scheduler-type', type=str, default='cosine') 
parser.add_argument('--epochs', type=int, default=200) 
parser.add_argument('--warmup', type=int, default=10) 
parser.add_argument('--batch-size', type=int, default=32) 
parser.add_argument('--lr', type=float, default=0.0001) 
parser.add_argument('--weight-decay', type=float, default=1e-4) 


parser.add_argument('--seeds', type=int, default=42) 
# 42, 2333, 23333, 12138, 666, 886, 314159, 271828, 2020, 2022 

parser.add_argument('--device', type=int, choices=[0, 1, 2, 3], default=0) 

parser.add_argument('--save-state', action='store_true') 

parser.add_argument('--transformer-dropout', type=float, default=0.1) 

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

from train_evaluate import hold_out, cross_validation, gt_hold_out, gt_cross_validation 
from ogb.graphproppred import PygGraphPropPredDataset 

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

dataset = PygGraphPropPredDataset(name='ogbg-molbbbp', root='.') 

from g import GraphTransformer 

model = GraphTransformer( 
    dataset.num_features, 128, dataset.num_tasks, 
    layer_sequence=['SelfAtt', 'SelfAtt', 'SelfAtt', 'SelfAtt'], 
    dropout_ratio=args.transformer_dropout, 
    layer_norm=True) 

# hold_out(model, dataset, args) 
gt_hold_out(model, dataset, args) 
