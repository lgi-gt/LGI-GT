
import argparse 

parser = argparse.ArgumentParser(description="model training and evaluating") 

parser.add_argument('--info', type=str, default=None) 
parser.add_argument('--baseline', type=str, default='SAT') 
# parser.add_argument('--baseline', type=str, default='GPS') 

# dataset settings 
parser.add_argument('--dataset', type=str, default="CLUSTER", choices=["CLUSTER"]) 

# training settings 
parser.add_argument('--use-val-loss', action="store_true", dest='use_val_loss', 
    help="if set then use validation loss for choosing test accuracy; work only when use_val is True") 
parser.add_argument('--no-val-loss', action="store_false", dest='use_val_loss', 
    help="if set then use validation acc for choosing test accuracy; work only when use_val is True") 
parser.set_defaults(use_val_loss=False) 

parser.add_argument('--use-cpu', action="store_true") 

parser.add_argument('--max_freqs', type=int, default=16) 
parser.add_argument('--eigvec_norm', type=str, default='L2') 

parser.add_argument('--num_rw_steps', type=int, default=6) 

parser.add_argument('--num_classes', type=int, default=6) 

# parser.add_argument('--epochs', type=int, default=100) 
# parser.add_argument('--patience', type=int, default=500) 
# parser.add_argument('--batch-size', type=int, default=32) 
# parser.add_argument('--lr', type=float, default=0.001) 
# parser.add_argument('--weight-decay', type=float, default=0.) 
# parser.add_argument('--lr-decay-factor', type=float, default=1.0) 
# parser.add_argument('--lr-decay-step', type=int, default=50) 

parser.add_argument('--epochs', type=int, default=100) 
parser.add_argument('--scheduler', type=str, default='cosine', choices=['linear', 'cosine', 'none']) 
parser.add_argument('--warmup', type=int, default=5) 
parser.add_argument('--batch-size', type=int, default=32) 
parser.add_argument('--lr', type=float, default=0.001) 
parser.add_argument('--weight-decay', type=float, default=1e-5) 

parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)') 

#### model settings 
parser.add_argument('--clustering', action="store_true", dest='clustering') 
parser.add_argument('--no-clustering', action="store_false", dest='clustering') 
# parser.set_defaults(clustering=True) 
parser.set_defaults(clustering=False) 

parser.add_argument('--masked-attention', action="store_true", dest='masked_attention') 
parser.add_argument('--no-masked-attention', action="store_false", dest='masked_attention') 
# parser.set_defaults(masked_attention=False) 
parser.set_defaults(masked_attention=True) 

parser.add_argument('--gconv-dim', type=int, default=48) 
parser.add_argument('--dim-pe', type=int, default=16) 
parser.add_argument('--tlayer-dim', type=int, default=48) 

parser.add_argument('--gconv-dropout', type=float, default=0) 
parser.add_argument('--attn-dropout', type=float, default=0.5) 
parser.add_argument('--tlayer-dropout', type=float, default=0.1) 
parser.add_argument('--classifier-head-dropout', type=float, default=0.3) 

parser.add_argument('--num-layers', type=int, default=16) 
parser.add_argument('--num-heads', type=int, default=8) 

parser.add_argument('--skip-connection', type=str, default='short', choices=['none', 'long', 'short']) 

# other settings 
parser.add_argument('--seeds', type=int, default=2023) 
# 42, 2333, 23333, 12138, 666, 886, 314159, 271828, 2020, 2022 

parser.add_argument('--device', type=int, choices=[0, 1, 2, 3], default=1) 

parser.add_argument('--save-state', action='store_true') 

parser.add_argument('--n', type=int, default=1) 
parser.add_argument('--m', type=int, default=1) 

args = parser.parse_args() 

import sys 
# sys.path.insert(0, "..") 

import os 
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':16:8' 
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device) 
# os.environ['PYTHONHASHSEED'] = str(args.seeds) 


import random 
import numpy as np 

import torch 

from train_evaluate import hold_out 

from torch_geometric.datasets import GNNBenchmarkDataset 


def set_random_seed(seed): 
    # torch.backends.cudnn.deterministic = True 
    # torch.backends.cudnn.benchmark = False 
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    
#################################################################### 

set_random_seed(args.seeds) 

vl = sys.argv 
if '--info' in vl: 
    info_index = vl.index('--info') 
    vl.pop(info_index) 
    vl.pop(info_index) 

args.cmd_str = ' '.join(vl) 


import importlib 

module = importlib.import_module("lgi_gt_" + str(args.n) + str(args.m) )

from utils import compute_posenc_stats 
from functools import partial 

from torch_geometric.transforms import Compose 

def add_zeros(data): 
    z = data.edge_index.new_zeros(data.edge_index.shape[1]) 
    data.edge_attr = z 
    return data 

# dataset initilization 
train_dataset = GNNBenchmarkDataset(root='.', name='CLUSTER', split='train', 
                    pre_transform=Compose([ 
                                    add_zeros, 
                                    partial( 
                                        compute_posenc_stats, 
                                        pe_types=['RWSE'], 
                                        times=list(range(1, args.num_rw_steps + 1)), 
                                        max_freqs=args.max_freqs, 
                                        eigvec_norm=args.eigvec_norm
                                        )
                                    ]) ) 
val_dataset = GNNBenchmarkDataset(root='.', name='CLUSTER', split='val', 
                    pre_transform=Compose([ 
                                    add_zeros, 
                                    partial( 
                                        compute_posenc_stats, 
                                        pe_types=['RWSE'], 
                                        times=list(range(1, args.num_rw_steps + 1)), 
                                        max_freqs=args.max_freqs, 
                                        eigvec_norm=args.eigvec_norm
                                        )
                                    ]) ) 
test_dataset = GNNBenchmarkDataset(root='.', name='CLUSTER', split='test', 
                    pre_transform=Compose([ 
                                    add_zeros, 
                                    partial( 
                                        compute_posenc_stats, 
                                        pe_types=['RWSE'], 
                                        times=list(range(1, args.num_rw_steps + 1)), 
                                        max_freqs=args.max_freqs, 
                                        eigvec_norm=args.eigvec_norm
                                        )
                                    ]) ) 

# model construction 
model = module.LGI_GT( 
            in_dim=train_dataset.num_features, 
            out_dim=train_dataset.num_classes, 
            gconv_dim = args.gconv_dim, 
            tlayer_dim = args.tlayer_dim, 
            pese_type='RWSE', 
            dim_pe=args.dim_pe, 
            num_rw_steps=args.num_rw_steps, 
            max_freqs=args.max_freqs, 
            # layer_sequence=['SelfAtt', 'SelfAtt', 'SelfAtt', 'SelfAtt'], #'SelfAtt'], 
            num_layers = args.num_layers, 
            num_heads=args.num_heads, 
            gconv_dropout=args.gconv_dropout, 
            attn_dropout=args.attn_dropout, 
            tlayer_dropout=args.tlayer_dropout, 
            clustering=args.clustering, 
            masked_attention=args.masked_attention, 
            layer_norm=True, 
            skip_connection=args.skip_connection) 

# train and evaluate 
hold_out(model, train_dataset, val_dataset, test_dataset, args) 
