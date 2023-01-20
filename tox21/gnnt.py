from typing import Union  

import torch 
from torch import Tensor  

from torch import nn 

import torch.nn.functional as F 
from torch_geometric.nn import global_add_pool, global_mean_pool 
from torch_geometric.data import Batch, Data 
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder 

from gconv import GINConv_OGB 
from tlayer import GraphTransformerEncoderLayer 

class GraphTransformer(nn.Module): 

    def __init__(
        self,
        in_channels: int, 
        out_channels: int,
        gconv_dim: int, 
        tlayer_dim: int, 
        num_layers: int = 4, 
        num_heads: int = 4, 
        gconv_dropout: float = 0., 
        tlayer_dropout: float = 0., 
        clustering: bool = True, 
        masked_attention: bool = False, 
        layer_norm: bool = False, 
        skip_connection: str = 'none', 
        readout: str = 'mean' 
    ):
        super().__init__()
        self.in_channels = in_channels 
        self.gconv_dim = gconv_dim 
        self.tlayer_dim = tlayer_dim
        self.out_channels = out_channels
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

        self.node_embedding = AtomEncoder(gconv_dim) 
        self.edge_embedding = BondEncoder(gconv_dim) 

        self.lin1 = nn.Linear(in_channels, gconv_dim) 
        self.lin2 = nn.Linear(gconv_dim, out_channels) 
        
        self.gconvs1 = nn.ModuleList() 
        self.lns1 = nn.ModuleList() 
        self.gconvs2 = nn.ModuleList() 
        self.lns2 = nn.ModuleList() 

        self.tlayers = nn.ModuleList() 
        for i in range(num_layers): 
            self.gconvs1.append(GINConv_OGB(gconv_dim)) 
            self.lns1.append(nn.LayerNorm(gconv_dim)) 
            self.gconvs2.append(GINConv_OGB(gconv_dim)) 
            self.lns2.append(nn.LayerNorm(gconv_dim)) 
            self.tlayers.append( 
                GraphTransformerEncoderLayer( 
                    tlayer_dim, 
                    num_heads, 
                    dropout_ratio=self.tlayer_dropout, 
                    clustering=clustering, 
                    masked_attention=masked_attention, 
                    layer_norm=layer_norm)) 

    def reset_parameters(self): 
        pass # no need if initilize or construct a new instance every time before used 

    def forward(self, data: Union[Data, Batch]) -> Tensor: 
        x, edge_index, batch = data.x, data.edge_index, data.batch 
        edge_attr = data.edge_attr 

        x = self.node_embedding(x) 
        edge_attr = self.edge_embedding(edge_attr) 
        
        if self.readout == 'cls': 
            batch_CLS = self.CLS.expand(data.num_graphs, 1, -1) 
        else: 
            batch_CLS = None 

        out = 0 
        for i in range(self.num_layers): 

            x = self.gconvs1[i](x, edge_index, edge_attr) 
            x = self.lns1[i](x) 
            x = x.relu() 
            x = F.dropout(x, p=self.gconv_dropout, training=self.training) 

            x = self.gconvs2[i](x, edge_index, edge_attr) 
            x = self.lns2[i](x) 
            x = x.relu() 
            x = F.dropout(x, p=self.gconv_dropout, training=self.training) 
        
        for i in range(self.num_layers): 
            
            graph = (x, batch, batch_CLS) 
            x = self.tlayers[i](graph) 
            
            if self.readout == 'cls': 
                batch_CLS = x[-data.num_graphs:].unsqueeze(1) 
                x = x[:-data.num_graphs] 

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
            out = batch_CLS.squeeze(1) 
        else: 
            out = self.readout_fn(out, batch) 
        
        out = self.lin2(out) 

        return out  

