
"""
segment graph into subgraphs (if node number > 1000)
CLS from different subgraphs (same graph) will be global pooled 
"""

from typing import Union  

import os 

import pandas as pd

import torch 
from torch import Tensor 

from torch import nn 

import torch.nn.functional as F 

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool 
from torch_geometric.data import Batch, Data 

from utils import ASTNodeEncoder 

from gconv import GCNConv_CODE2 
from tlayer import GraphTransformerEncoderLayer 


class LGI_GT_segment(nn.Module): 

    def __init__(
        self, 
        dataset_root: str, 
        num_vocab: int, 
        max_seq_len: int, 
        max_input_len: int, 
        gconv_dim: int, 
        tlayer_dim: int, 
        num_layers: int = 4, 
        num_heads: int = 4, 
        gconv_dropout: float = 0., 
        attn_dropout: float = 0., 
        tlayer_dropout: float = 0., 
        clustering: bool = True, 
        masked_attention: bool = False, 
        layer_norm: bool = False, 
        skip_connection: str = 'none', 
        readout: str = 'mean', 
        segment_pooling: str = 'mean'  
    ):
        super().__init__()
        self.gconv_dim = gconv_dim 
        self.tlayer_dim = tlayer_dim
        self.num_vocab = num_vocab
        self.max_seq_len = max_seq_len 
        self.num_layers = num_layers 
        self.num_heads = num_heads 
        self.gconv_dropout = gconv_dropout 
        self.tlayer_dropout = tlayer_dropout 
        self.layer_norm = layer_norm 
        self.skip_connection = skip_connection 
        self.readout = readout 
        if readout == 'mean': 
            self.readout_fn = global_mean_pool 
        elif readout == 'add': 
            self.readout_fn = global_add_pool 
        elif readout == 'cls': 
            self.CLS = nn.Parameter(torch.randn(1, tlayer_dim)) 
        else: 
            pass 
            
        if segment_pooling == 'mean': 
            self.segment_pooling_fn = global_mean_pool
        elif segment_pooling == 'max': 
            self.segment_pooling_fn = global_max_pool
        elif segment_pooling == 'sum': 
            self.segment_pooling_fn = global_add_pool

        nodetypes_mapping = pd.read_csv(os.path.join(dataset_root, 'mapping', 'typeidx2type.csv.gz'))
        nodeattributes_mapping = pd.read_csv(os.path.join(dataset_root, 'mapping', 'attridx2attr.csv.gz'))

        self.node_encoder = ASTNodeEncoder( 
            gconv_dim, 
            num_nodetypes = len(nodetypes_mapping['type']), 
            num_nodeattributes = len(nodeattributes_mapping['attr']), 
            max_depth = 20
            )

        self.edge_encoder = nn.Linear(2, 128) 

        self.graph_pred_linear_list = nn.ModuleList() 
        for i in range(max_seq_len):
            self.graph_pred_linear_list.append(nn.Linear(tlayer_dim, num_vocab)) 
        
        self.gconvs = nn.ModuleList() 

        self.middle_linear = nn.ModuleList() 
        self.middle_ln = nn.ModuleList() 

        self.tlayers = nn.ModuleList() 
        for i in range(num_layers): 
            self.gconvs.append(GCNConv_CODE2(gconv_dim, 128)) 
            self.middle_linear.append(nn.Linear(gconv_dim, gconv_dim)) 
            self.middle_ln.append(nn.LayerNorm(gconv_dim)) 
            self.tlayers.append( 
                GraphTransformerEncoderLayer( 
                    tlayer_dim, 
                    num_heads, 
                    attn_dropout_ratio=attn_dropout, 
                    dropout_ratio=self.tlayer_dropout, 
                    clustering=clustering, 
                    masked_attention=masked_attention, 
                    layer_norm=layer_norm, 
                    max_input_len=max_input_len)) 

    def reset_parameters(self): 
        pass # no need if initilize or construct a new instance every time before used 

    def forward(self, data: Union[Data, Batch]) -> Tensor: 
        x, edge_index, batch = data.x, data.edge_index, data.batch 
        edge_attr = data.edge_attr 
        node_depth = data.node_depth 

        # use node_segment_batch as batch 
        node_segment_batch = data.node_segment_batch # every 1000 nodes come into 1 segment as 1 subgraph
        subgraphs_to_graph_batch = data.subgraphs_to_graph_batch # which subgraph belongs to which graph, used for cls readout
        num_segments = subgraphs_to_graph_batch.shape[0] # how many subgraphs after segmenting 

        x = self.node_encoder(x, node_depth.view(-1,)) 
        edge_attr = self.edge_encoder(edge_attr) 
        
        if self.readout == 'cls': 
            batch_CLS = self.CLS.expand(num_segments, 1, -1) 
        else: 
            batch_CLS = None 

        x = torch.cat((x, batch_CLS[:,0,:]), dim=0) 
        out = 0 
        for i in range(self.num_layers): 

            x = self.gconvs[i](x, edge_index, edge_attr) 
            x = F.dropout(x, p=self.gconv_dropout, training=self.training) 

            x = self.middle_linear[i](x) 
            x = self.middle_ln[i](x) 
            x = x.relu() 
            x = F.dropout(x, p=self.gconv_dropout, training=self.training) 

            if self.readout == 'cls': 
                batch_CLS = x[-num_segments:].unsqueeze(1) 
                x = x[:-num_segments] 

            graph = (x, node_segment_batch, batch_CLS) 
            x = self.tlayers[i](graph) 

            # if self.readout == 'cls': 
            #     batch_CLS = x[-num_segments:].unsqueeze(1) 
            #     x = x[:-num_segments] 

            if self.skip_connection == 'none': 
                # like JK=last, I would rather call it "no skip connection"
                out = x # only last layer to output, consequently: identical 
            elif self.skip_connection == 'long': 
                # like JK=add, I would rather call it "summation long skip connection to the output"
                out = out + x # long skip connection to output: hierarchical 
            elif self.skip_connection == 'short': 
                # normal residual, I would rather call it "summantion short skip connection in every layer"
                out = out + x 
                x = out 

        if self.readout == 'cls': 
            # batch_CLS = x[-num_segments:].unsqueeze(1) 
            # out = batch_CLS.squeeze(1) 
            out = x[-num_segments:] 
            out = self.segment_pooling_fn(out, subgraphs_to_graph_batch) 
        else: 
            out = self.readout_fn(out, batch) 
        
        pred_list = [] 
        for i in range(self.max_seq_len): 
            pred_list.append(self.graph_pred_linear_list[i](out)) 

        return pred_list 

