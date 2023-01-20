
import argparse

parser = argparse.ArgumentParser(description="model training and evaluating") 

parser.add_argument('--info', type=str, default=None) 

# dataset settings 
parser.add_argument('--dataset', type=str, default="ogbg-moltox21", choices=["ogbg-moltox21"]) 

# training settings 
parser.add_argument('--use-val-loss', action="store_true", dest='use_val_loss', 
    help="if set then use validation loss for choosing test accuracy; work only when use_val is True") 
parser.add_argument('--no-val-loss', action="store_false", dest='use_val_loss', 
    help="if set then use validation acc for choosing test accuracy; work only when use_val is True") 
parser.set_defaults(use_val_loss=False) 

# parser.add_argument('--epochs', type=int, default=100) 
# parser.add_argument('--patience', type=int, default=500) 
# parser.add_argument('--batch-size', type=int, default=32) 
# parser.add_argument('--lr', type=float, default=0.001) 
# parser.add_argument('--weight-decay', type=float, default=0.) 
# parser.add_argument('--lr-decay-factor', type=float, default=1.0) 
# parser.add_argument('--lr-decay-step', type=int, default=50) 

parser.add_argument('--epochs', type=int, default=100) 
parser.add_argument('--scheduler', type=str, default='linear') 
parser.add_argument('--warmup', type=int, default=10) 
parser.add_argument('--patience', type=int, default=25) 
parser.add_argument('--batch-size', type=int, default=128) 
parser.add_argument('--lr', type=float, default=0.0001) 
parser.add_argument('--weight-decay', type=float, default=0.0001) 

parser.add_argument('--clustering', action="store_true", dest='clustering') 
parser.add_argument('--no-clustering', action="store_false", dest='clustering') 
parser.set_defaults(clustering=True) 

parser.add_argument('--masked-attention', action="store_true", dest='masked_attention') 
parser.add_argument('--no-masked-attention', action="store_false", dest='masked_attention') 
parser.set_defaults(masked_attention=False) 

parser.add_argument('--gconv-dim', type=int, default=128) 
parser.add_argument('--tlayer-dim', type=int, default=128) 

parser.add_argument('--gconv-dropout', type=float, default=0) 
parser.add_argument('--tlayer-dropout', type=float, default=0) 

parser.add_argument('--num-layers', type=int, default=4) 
parser.add_argument('--num-heads', type=int, default=4) 

parser.add_argument('--skip-connection', type=str, default='none', choices=['none', 'long', 'short']) 

parser.add_argument('--readout', type=str, default='mean', choices=['mean', 'add', 'cls'])

parser.add_argument('--seeds', type=int, default=0) 
# 42, 2333, 23333, 12138, 666, 886, 314159, 271828, 2020, 2022 

parser.add_argument('--device', type=int, choices=[0, 1, 2, 3], default=0) 

parser.add_argument('--save-state', action='store_true') 


args = parser.parse_args() 


import sys 
# sys.path.append("..") 

import os 
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':16:8' 
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device) 
os.environ['PYTHONHASHSEED'] = str(args.seeds) 


import time 
import random 
import numpy as np 

import torch 

from train_evaluate import record_basic_info, \
                            hold_out, gt_hold_out  
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

vl = sys.argv 
if '--info' in vl: 
    info_index = vl.index('--info') 
    vl.pop(info_index) 
    vl.pop(info_index) 

args.cmd_str = ' '.join(vl) 

from parallel import GraphTransformer 

dataset = PygGraphPropPredDataset(name='ogbg-moltox21', root='.') 
model = GraphTransformer( 
            dataset.num_features, dataset.num_tasks, 
            gconv_dim = args.gconv_dim, 
            tlayer_dim = args.tlayer_dim, 
            num_layers = args.num_layers, 
            num_heads=args.num_heads, 
            gconv_dropout=args.gconv_dropout, 
            tlayer_dropout=args.tlayer_dropout, 
            clustering=args.clustering, 
            masked_attention=args.masked_attention, 
            layer_norm=True, 
            skip_connection=args.skip_connection, 
            readout=args.readout) 

gt_hold_out(model, dataset, args) 
# hold_out(model, dataset, args) 
