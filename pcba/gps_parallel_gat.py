

from typing import List, Optional, Tuple, Type, Union 


import torch 
from torch import Tensor 

from torch import nn 
import torch_geometric.nn as gnn 

import torch.nn.functional as F 

from torch_geometric.nn import global_add_pool, global_mean_pool 
from torch_geometric.data import Batch, Data 
from torch_geometric.utils import to_dense_batch 

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder 

from tlayer import GraphTransformerEncoderLayer 


class GraphTransformer(nn.Module): 

    def __init__(
        self, 
        out_dim: int, 
        gconv_dim: int, 
        tlayer_dim: int, 
        num_layers: int = 4, 
        num_heads: int = 4, 
        gat_attn_dropout: float = 0., 
        gconv_dropout: float = 0., 
        tlayer_attn_dropout: float = 0., 
        tlayer_dropout: float = 0., 
        clustering: bool = True, 
        masked_attention: bool = False, 
        layer_norm: bool = False, 
        skip_connection: str = 'none', 
        readout: str = 'mean' 
    ): 
        super().__init__() 
        self.gconv_dim = gconv_dim 
        self.tlayer_dim = tlayer_dim
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

        self.node_encoder = AtomEncoder(gconv_dim) 
        self.edge_encoder = BondEncoder(gconv_dim)  

        self.lin2 = nn.Linear(tlayer_dim, out_dim) 
        
        self.gconvs1 = nn.ModuleList() 

        self.sas = nn.ModuleList() 
        self.lns1 = nn.ModuleList() 
        self.lns2 = nn.ModuleList() 
        self.ff_linear1 = nn.ModuleList() 
        self.ff_linear2 = nn.ModuleList() 

        self.middle_linear = nn.ModuleList() 
        self.middle_ln = nn.ModuleList() 

        self.tlayers = nn.ModuleList() 
        for i in range(num_layers): 
            self.gconvs1.append(gnn.GATConv(gconv_dim, gconv_dim // 8, 
                                    heads=8, 
                                    edge_dim=gconv_dim, 
                                    dropout=gat_attn_dropout)) 
            self.middle_linear.append(nn.Linear(gconv_dim, gconv_dim)) 
            self.middle_ln.append(nn.LayerNorm(gconv_dim)) 
            self.sas.append(nn.MultiheadAttention(tlayer_dim, num_heads, 
                            dropout=tlayer_attn_dropout)) 
            self.lns2.append(nn.LayerNorm(gconv_dim)) 
            self.lns1.append(nn.LayerNorm(gconv_dim)) 
            self.ff_linear1.append(nn.Linear(tlayer_dim, 2 * tlayer_dim)) 
            self.ff_linear2.append(nn.Linear(2 * tlayer_dim, tlayer_dim)) 
            # self.tlayers.append( 
            #     GraphTransformerEncoderLayer( 
            #         tlayer_dim, 
            #         num_heads, 
            #         global_attn_dropout_ratio=tlayer_attn_dropout, 
            #         global_ffn_dropout_ratio=self.tlayer_dropout, 
            #         clustering=clustering, 
            #         masked_attention=masked_attention)) 

    def reset_parameters(self): 
        pass # no need if initilize or construct a new instance every time before used 

    def forward(self, data: Union[Data, Batch]) -> Tensor: 
        x, edge_index, batch = data.x, data.edge_index, data.batch 
        edge_attr = data.edge_attr 

        edge_attr = self.edge_encoder(edge_attr) 
        x = self.node_encoder(x) 

        # if self.readout == 'cls': 
        #     batch_CLS = self.CLS.expand(data.num_graphs, 1, -1) 
        # else: 
        #     batch_CLS = None 

        for i in range(self.num_layers): 
            x0 = x 

            x = self.gconvs1[i](x, edge_index, edge_attr) 
            x = self.middle_linear[i](x) 
            x = self.middle_ln[i](x) 
            x = x.relu() 
            x = F.dropout(x, p=self.gconv_dropout, training=self.training) 
            x1 = x 

            x = x0 
            x_dense, mask = to_dense_batch(x, batch) 
            x_dense = x_dense.transpose(1, 0) 
            x = self.sas[i](x_dense, x_dense, x_dense, 
                            attn_mask=None, 
                            key_padding_mask=~mask, 
                            need_weights=False)[0] 
            x = x.transpose(1, 0) 
            x = x[mask] 
            x = F.dropout(x, p=self.tlayer_dropout, training=self.training)
            x2 = x 

            x = x1 + x2 
            x = self.lns1[i](x) 

            ff_in = x 

            x = self.ff_linear1[i](x) 
            x = F.relu(x) 
            x = F.dropout(x, p=self.tlayer_dropout, training=self.training) 
            x = self.ff_linear2[i](x) 
            x = F.dropout(x, p=self.tlayer_dropout, training=self.training) 

            x = x + ff_in 
            x = self.lns2[i](x) 

        # if self.readout == 'cls': 
        #     if self.skip_connection == 'short': 
        #         out = batch_CLS.squeeze(1) 
        #     else: 
        #         out = out.squeeze(1) 
        # else: 
        #     out = self.readout_fn(out, batch) 

        out = self.readout_fn(x, batch) 

        out = self.lin2(out) 

        return out 

