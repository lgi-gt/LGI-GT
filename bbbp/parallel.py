
from typing import List 

import torch 
from torch import Tensor, hinge_embedding_loss 
from torch.nn import LayerNorm, Linear, Sequential, \
            ReLU, Tanh, Sigmoid, LeakyReLU, Dropout, \
            Module, ModuleList, BatchNorm1d    

from torch import nn 
import torch.nn.functional as F 
from torch_geometric.nn import global_add_pool, global_mean_pool 
from torch_geometric.data import Data 
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder 

from gconv import GINConv_OGB 
from tlayer import GraphTransformerEncoderLayer 

class GraphTransformer(Module): 

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        layer_sequence: List[str] = ['SelfAtt'],
        num_heads: int = 4, 
        dropout_ratio: float = 0.,
        layer_norm: bool = False, 
        readout: str = 'mean' 
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.layer_sequence = layer_sequence
        self.num_heads = num_heads 
        self.dropout_ratio = dropout_ratio 
        self.layer_norm = layer_norm 
        if readout == 'mean': 
            self.readout = global_mean_pool 
        elif readout == 'add': 
            self.readout = global_add_pool 
        elif readout == 'cls': 
            self.CLS = nn.Parameter(torch.randn(1, hidden_channels)) 
            self.readout = self.CLS_pool 
        else: 
            pass 

        self.lin1 = Linear(in_channels, hidden_channels) 
        self.lin2 = Linear(hidden_channels, out_channels) 

        self.node_embedding = AtomEncoder(hidden_channels) 
        self.edge_embedding = BondEncoder(hidden_channels) 

        self.gconvs = ModuleList() 
        self.pre_lns = ModuleList() 

        self.tlayers = ModuleList() 
        for i, pool_type in enumerate(layer_sequence): 
            if pool_type == 'SelfAtt': 
                self.gconvs.append(GINConv_OGB(hidden_channels)) 
                self.pre_lns.append(LayerNorm(hidden_channels)) 
                self.tlayers.append( 
                    GraphTransformerEncoderLayer( 
                        hidden_channels, 
                        num_heads, 
                        dropout_ratio=dropout_ratio, 
                        layer_norm=layer_norm)) 

    def reset_parameters(self): 
        pass 

    def CLS_pool(self, x, batch): 
        #TODO 
        pass 

    def forward(self, data: Data) -> Tensor: 
        x, edge_index, batch = data.x, data.edge_index, data.batch 
        x = self.node_embedding(x) 

        edge_attr = data.edge_attr 
        edge_attr = self.edge_embedding(edge_attr) 
        out = 0 
        for i, name in enumerate(self.layer_sequence): 
            x0 = x 
            x = self.gconvs[i](x, edge_index, edge_attr) 
            x = self.pre_lns[i](x) 
            x = F.dropout(x, p=0.5, training=self.training) 
        
            graph = (x0, edge_index, batch)  
            x0 = self.tlayers[i](graph)  
            
            x = x + x0 
            out = out + x # long skip connection to output 

        out = self.readout(out, batch) 
        out = self.lin2(out) 

        return out  

