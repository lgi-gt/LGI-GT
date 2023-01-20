
import argparse

parser = argparse.ArgumentParser(description="model training and evaluating") 

parser.add_argument('--info', type=str, default=None) 
parser.add_argument('--baseline', type=str, default='GraphTrans') 

# dataset settings 
parser.add_argument('--dataset', type=str, default="ogbg-code2", choices=["ogbg-code2"]) 

parser.add_argument('--max_seq_len', type=int, default=5,
                        help='maximum sequence length to predict (default: 5)')
parser.add_argument('--num_vocab', type=int, default=5000,
                        help='the number of vocabulary used for sequence prediction (default: 5000)')

parser.add_argument("--max_input_len", type=int, default=1000, help="The max input length of transformer input") 

# training settings 
parser.add_argument('--use-val-loss', action="store_true", dest='use_val_loss', 
    help="if set then use validation loss for choosing test accuracy; work only when use_val is True") 
parser.add_argument('--no-val-loss', action="store_false", dest='use_val_loss', 
    help="if set then use validation acc for choosing test accuracy; work only when use_val is True") 
parser.set_defaults(use_val_loss=False) 

parser.add_argument('--use-cpu', action="store_true") 

parser.add_argument('--epochs', type=int, default=30) 
parser.add_argument('--scheduler', type=str, default=None, choices=['linear', 'cosine']) 
parser.add_argument('--warmup', type=int, default=0) 
parser.add_argument('--batch-size', type=int, default=32) 
parser.add_argument('--lr', type=float, default=0.0001) 
parser.add_argument('--weight-decay', type=float, default=1e-6) 

parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')

parser.add_argument('--clustering', action="store_true", dest='clustering') 
parser.add_argument('--no-clustering', action="store_false", dest='clustering') 
# parser.set_defaults(clustering=True) 
parser.set_defaults(clustering=False) 

parser.add_argument('--masked-attention', action="store_true", dest='masked_attention') 
parser.add_argument('--no-masked-attention', action="store_false", dest='masked_attention') 
# parser.set_defaults(masked_attention=False) 
parser.set_defaults(masked_attention=True) 

parser.add_argument('--gconv-dim', type=int, default=128) 
parser.add_argument('--tlayer-dim', type=int, default=128) 

parser.add_argument('--gconv-dropout', type=float, default=0) 
parser.add_argument('--attn-dropout', type=float, default=0) 
parser.add_argument('--tlayer-dropout', type=float, default=0) 

parser.add_argument('--num-layers', type=int, default=4) 
parser.add_argument('--num-heads', type=int, default=4) 

parser.add_argument('--skip-connection', type=str, default='none', choices=['none', 'long', 'short']) 

parser.add_argument('--readout', type=str, default='mean', choices=['mean', 'add', 'cls'])

# other settings 
parser.add_argument('--seeds', type=int, default=0) 
# 42, 2333, 23333, 12138, 666, 886, 314159, 271828, 2020, 2022 

parser.add_argument('--device', type=int, choices=[0, 1, 2, 3], default=0) 

parser.add_argument('--save-state', action='store_true') 

parser.add_argument('--segment_pooling', type=str, default='mean', choices=['mean', 'max', 'sum'])


args = parser.parse_args() 

import sys 
# sys.path.insert(0, "..") 

import os 
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':16:8' 
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device) 
# os.environ['PYTHONHASHSEED'] = str(args.seeds) 


import time 
import random 
import numpy as np 

import torch 

from torch_geometric import transforms 

from train_evaluate import hold_out 

from ogb.graphproppred import PygGraphPropPredDataset 

### importing utils
from utils import get_vocab_mapping
### for data transform
from utils import augment_edge, encode_y_to_arr 

# for data pre-processing 
from utils import segment 

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

from lgi_gt_segment import LGI_GT_segment  

dataset = PygGraphPropPredDataset(name='ogbg-code2', root='.') 

# pre-processing 
seq_len_list = np.array([len(seq) for seq in dataset.data.y])
print('Target seqence less or equal to {} is {}%.'.format(args.max_seq_len, np.sum(seq_len_list <= args.max_seq_len) / len(seq_len_list))) 

split_idx = dataset.get_idx_split() 
vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], args.num_vocab)
dataset.transform = transforms.Compose([augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, args.max_seq_len),  segment]) 
# pre-processing done 

model = LGI_GT_segment( 
            os.path.abspath(dataset.root), 
            num_vocab = len(vocab2idx), 
            max_seq_len=args.max_seq_len, 
            max_input_len=args.max_input_len, 
            gconv_dim = args.gconv_dim, 
            tlayer_dim = args.tlayer_dim, 
            num_layers = args.num_layers, 
            num_heads=args.num_heads, 
            gconv_dropout=args.gconv_dropout, 
            attn_dropout=args.attn_dropout, 
            tlayer_dropout=args.tlayer_dropout, 
            clustering=args.clustering, 
            masked_attention=args.masked_attention, 
            layer_norm=True, 
            skip_connection=args.skip_connection, 
            readout=args.readout, 
            segment_pooling=args.segment_pooling) 

hold_out(model, dataset, args, idx2vocab) 
