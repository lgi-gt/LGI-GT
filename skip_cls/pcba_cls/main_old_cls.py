
import argparse

parser = argparse.ArgumentParser(description="model training and evaluating") 

parser.add_argument('--info', type=str, default=None) 
# parser.add_argument('--baseline', type=str, default='SAT') 
parser.add_argument('--baseline', type=str, default='GPS') 

# dataset settings 
parser.add_argument('--dataset', type=str, default="ogbg-molpcba", choices=["ogbg-molpcba"]) 

# training settings 
parser.add_argument('--use-val-loss', action="store_true", dest='use_val_loss', 
    help="if set then use validation loss for choosing test accuracy; work only when use_val is True") 
parser.add_argument('--no-val-loss', action="store_false", dest='use_val_loss', 
    help="if set then use validation acc for choosing test accuracy; work only when use_val is True") 
parser.set_defaults(use_val_loss=False) 

parser.add_argument('--use-cpu', action="store_true") 

parser.add_argument('--epochs', type=int, default=100) 
parser.add_argument('--scheduler', type=str, default='linear', choices=['none', 'linear', 'cosine']) 
parser.add_argument('--warmup', type=int, default=10) 
parser.add_argument('--batch-size', type=int, default=256) 
parser.add_argument('--lr', type=float, default=0.0002) 
parser.add_argument('--weight-decay', type=float, default=1e-4) 

parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)') 

# model settings 
parser.add_argument('--clustering', action="store_true", dest='clustering') 
parser.add_argument('--no-clustering', action="store_false", dest='clustering') 
# parser.set_defaults(clustering=True) 
parser.set_defaults(clustering=False) 

parser.add_argument('--masked-attention', action="store_true", dest='masked_attention') 
parser.add_argument('--no-masked-attention', action="store_false", dest='masked_attention') 
# parser.set_defaults(masked_attention=False) 
parser.set_defaults(masked_attention=True) 

parser.add_argument('--gconv-dim', type=int, default=384) 
parser.add_argument('--tlayer-dim', type=int, default=384) 

parser.add_argument('--local-attn-dropout', type=float, default=0.0) 
parser.add_argument('--local-ffn-dropout', type=float, default=0.3) 

parser.add_argument('--global-attn-dropout', type=float, default=0.3) 
parser.add_argument('--global-ffn-dropout', type=float, default=0.3) 

parser.add_argument('--num-layers', type=int, default=5) 
parser.add_argument('--num-heads', type=int, default=8) 

parser.add_argument('--skip-connection', type=str, default='none', choices=['none', 'long', 'short']) 

parser.add_argument('--norm', type=str, default='ln', choices=['ln', 'bn']) 
parser.add_argument('--readout', type=str, default='cls', choices=['mean', 'add', 'cls']) 

# other settings 
parser.add_argument('--seeds', type=int, default=2023) 
# 42, 2333, 23333, 12138, 666, 886, 314159, 271828, 2020, 2022 

parser.add_argument('--device', type=int, choices=[0, 1, 2, 3, 4, 5], default=1) 

parser.add_argument('--save-state', action='store_true') 


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

from ogb.graphproppred import PygGraphPropPredDataset 

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

from lgi_gt_old_cls import LGI_GT  

dataset = PygGraphPropPredDataset(name='ogbg-molpcba', root='.') 

model = LGI_GT( 
            out_dim = dataset.num_tasks, 
            gconv_dim = args.gconv_dim, 
            tlayer_dim = args.tlayer_dim, 
            num_layers = args.num_layers, 
            num_heads=args.num_heads, 
            local_attn_dropout=args.local_attn_dropout, 
            global_attn_dropout=args.global_attn_dropout, 
            local_ffn_dropout=args.local_ffn_dropout, 
            global_ffn_dropout=args.global_ffn_dropout, 
            clustering=args.clustering, 
            masked_attention=args.masked_attention, 
            norm=args.norm, 
            skip_connection=args.skip_connection, 
            readout=args.readout)

hold_out(model, dataset, args) 