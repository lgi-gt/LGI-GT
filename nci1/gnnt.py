import math 
from typing import List, Optional, Tuple, Type 

import torch 
from torch import Tensor  
from torch.nn import LayerNorm, Linear, Sequential, \
            ReLU, Tanh, Sigmoid, LeakyReLU, Dropout, \
            Module, ModuleList, BatchNorm1d    

from torch import nn 
import torch_geometric.nn as gnn 

import torch.nn.functional as F 

from torch_geometric.utils import to_dense_batch 

from torch_geometric.nn import global_add_pool, global_mean_pool 

from torch_geometric.data import Data 

class MultiHeadAttention(Module): 

    def __init__(self, hidden_dim: int, num_heads: int, 
            dropout_ratio: float = 0.0, clustering: bool = True): 

        super().__init__() 
        self.hidden_dim = hidden_dim 
        self.num_heads = num_heads 
        self.dropout = Dropout(dropout_ratio) 
        self.clustering = clustering 

        self.linear_q = Linear(hidden_dim, hidden_dim) 
        self.linear_k = Linear(hidden_dim, hidden_dim) 
        self.linear_v = Linear(hidden_dim, hidden_dim) 

        self.linear_attn_out = Linear(hidden_dim, hidden_dim) 

    def reset_parameters(self):
        self.linear_q.reset_parameters()
        self.linear_k.reset_parameters()
        self.linear_v.reset_parameters() 
        self.linear_attn_out.reset_parameters()

    def forward(
        self,
        x_q: Tensor,
        x_k: Tensor, 
        x_v: Tensor, 
        mask: Tensor 
    ) -> Tensor: # B x N x F -> B x N x F 

        Q = self.linear_q(x_q) 
        K = self.linear_k(x_k) 
        V = self.linear_v(x_v) 

        dim_split = self.hidden_dim // self.num_heads
        Q_heads = torch.cat(Q.split(dim_split, 2), dim=0)
        K_heads = torch.cat(K.split(dim_split, 2), dim=0)
        V_heads = torch.cat(V.split(dim_split, 2), dim=0)
        
        attention_score = Q_heads.bmm(K_heads.transpose(1, 2)) 
        attention_score = attention_score / math.sqrt(self.hidden_dim) 

        inf_mask = (~mask).unsqueeze(1).to(dtype=torch.float) * -1e9 
        # inf_mask = (~mask).unsqueeze(-1).to(dtype=torch.float) * -1e9
        inf_mask = torch.cat([inf_mask for _ in range(self.num_heads)], 0) 

        # A = torch.softmax(attention_score, 1) # clustering, no inf mask 
        # A = torch.softmax(inf_mask + attention_score, 1) 

        # A = torch.softmax(inf_mask + attention_score, -1) # no clusering 
        # A = torch.softmax(attention_score, -1) # no clusering 

        if self.clustering: 
            A = torch.softmax(attention_score, 1) 
        else: 
            A = torch.softmax(attention_score, -1) 
            
        A = self.dropout(A) 
        out = torch.cat((A.bmm(V_heads)).split(Q.size(0), 0), 2) 
        out = self.linear_attn_out(out) 
        
        return out


class NodeSelfAttention(MultiHeadAttention): 

    def __init__(self, hidden_dim: int, num_heads: int, 
            dropout_ratio: float = 0., clustering: bool = True):
        super().__init__(hidden_dim, num_heads, dropout_ratio, clustering)

    def forward(
        self,
        x: Tensor, 
        mask: Tensor 
    ) -> Tensor: 

        return super().forward(x, x, x, mask) 
        

class GraphTransformerEncoderLayer(Module): 
    
    def __init__(self, hidden_dim, num_heads: int, 
            dropout_ratio: float = 0.0, clustering: bool = True, layer_norm: bool = False): 
        
        super().__init__() 
        self.hidden_dim = hidden_dim 
        self.num_heads = num_heads 
        self.layer_norm = layer_norm 
        self.dropout = Dropout(dropout_ratio) 
        
        self.node_self_attention = NodeSelfAttention( 
            hidden_dim, num_heads, dropout_ratio, clustering) 
        
        self.linear_out1 = Linear(hidden_dim, 2*hidden_dim) 
        self.linear_out2 = Linear(2*hidden_dim, hidden_dim) 

        if layer_norm: 
            # self.ln0 = BatchNorm1d(hidden_dim) 
            # self.ln1 = BatchNorm1d(hidden_dim) 
            self.ln0 = LayerNorm(hidden_dim) 
            self.ln1 = LayerNorm(hidden_dim) 

    def reset_parameters(self): 
        self.node_self_attention.reset_parameters() 
        self.linear_out1.reset_parameters() 
        self.linear_out2.reset_parameters() 
        if self.layer_norm:
            self.ln0.reset_parameters()
            self.ln1.reset_parameters() 
        
    def forward(self, graph: Optional[Tuple[Tensor, Tensor, Tensor]]): 
        
        # Sparse -> Dense 
        x, edge_index, batch = graph 
        x_dense, mask = to_dense_batch(x, batch) 
        
        # Dense given to Node Self Attention  

        attention_out = self.node_self_attention(x_dense, mask) 
        
        attention_out = self.dropout(attention_out) 
        attention_out = attention_out + x_dense 

        # Dense -> Sparse 
        attention_out = attention_out[mask] 

        if self.layer_norm: 
            attention_out = self.ln0(attention_out) 
        
        out = self.linear_out1(attention_out) 
        out = torch.relu(out) 
        out = self.dropout(out) 
        out = self.linear_out2(out) 
        out = self.dropout(out) 

        out = out + attention_out 

        if self.layer_norm: 
            out = self.ln1(out) 

        return out 


class GraphTransformer(Module): 

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        layer_sequence: List[str] = ['SelfAtt'],
        num_heads: int = 4, 
        gconv_dropout: float = 0., 
        between_dropout: float = 0., 
        tlayer_dropout: float = 0., 
        clustering: bool = True, 
        layer_norm: bool = False, 
        readout: str = 'mean' 
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.layer_sequence = layer_sequence
        self.num_heads = num_heads 
        self.gconv_dropout = gconv_dropout 
        self.between_dropout = between_dropout 
        self.tlayer_dropout = tlayer_dropout 
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

        self.gconvs = ModuleList() 
        self.within_linears = ModuleList() 
        self.within_bns = ModuleList() 
        self.middle_linears = ModuleList() 
        self.bns = ModuleList() 
        self.pre_lns = ModuleList() 

        self.tlayers = ModuleList() 
        for i, layer_type in enumerate(layer_sequence): 
            if layer_type == 'SelfAtt': 
                self.gconvs.append(gnn.GCNConv(hidden_channels, hidden_channels)) 
                self.bns.append(nn.BatchNorm1d(hidden_channels)) 

                self.within_linears.append(nn.Linear(hidden_channels, hidden_channels)) 
                self.within_bns.append(nn.BatchNorm1d(hidden_channels)) 

                self.gconvs.append(gnn.GCNConv(hidden_channels, hidden_channels)) 
                self.bns.append(nn.BatchNorm1d(hidden_channels)) 

                self.middle_linears.append(Linear(hidden_channels, hidden_channels)) 
                self.pre_lns.append(LayerNorm(hidden_channels)) 
                self.tlayers.append( 
                    GraphTransformerEncoderLayer( 
                        hidden_channels, 
                        num_heads, 
                        dropout_ratio=self.tlayer_dropout, 
                        clustering=clustering, 
                        layer_norm=layer_norm)) 

    def reset_parameters(self): 
        self.lin1.reset_parameters() 
        self.lin2.reset_parameters() 
        for layer in self.tlayers: 
            layer.reset_parameters() 
    
    def CLS_pool(self, x, batch): 
        #TODO 
        pass 

    def forward(self, data: Data) -> Tensor: 
        x, edge_index, batch = data.x, data.edge_index, data.batch 
        x = self.lin1(x) 

        out = 0 
        for i, name in enumerate(self.layer_sequence): 

            x = self.gconvs[2*i](x, edge_index) 
            x = self.bns[2*i](x) 

            x = x.relu() 
            x = F.dropout(x, p=self.gconv_dropout, training=self.training) 

            x = self.within_linears[i](x) 
            x = self.within_bns[i](x) 
            x = x.relu() 

            x = self.gconvs[2*i+1](x, edge_index) 
            x = self.bns[2*i+1](x) 

            x = x.relu() 
            x = F.dropout(x, p=self.gconv_dropout, training=self.training) 

        for i, name in enumerate(self.layer_sequence): 
            x = self.middle_linears[i](x) 
            x = self.pre_lns[i](x) 
            x = x.relu() 
            x = F.dropout(x, p=self.between_dropout, training=self.training) 
            
            graph = (x, edge_index, batch) 
            x = self.tlayers[i](graph) 

            out = x # only last layer to output, consequently: identical  

        out = self.readout(out, batch) 
        out = self.lin2(out) 

        return out  


