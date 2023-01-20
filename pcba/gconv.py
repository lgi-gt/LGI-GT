

import math 
from typing import Optional 

import torch 

from torch import nn 
import torch_geometric.nn as gnn 

import torch.nn.functional as F 

from torch_geometric.utils import softmax


class EELA(gnn.MessagePassing): 
    def __init__(self, hidden_dim: int, edge_input_dim: int, num_heads: int,
                local_attn_dropout_ratio: float = 0.0, 
                local_ffn_dropout_ratio: float = 0.0): 
        
        super().__init__(aggr='add', node_dim=0) 

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads 
        self.local_attn_dropout_ratio = local_attn_dropout_ratio 

        self.linear_dst = nn.Linear(hidden_dim, hidden_dim) 
        self.linear_src_edge = nn.Linear(2 * hidden_dim, hidden_dim) 

        self.ffn = nn.Sequential( 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(local_ffn_dropout_ratio), 
        ) 

    def forward(self, x, edge_index, edge_attr): 
        local_out = self.propagate(edge_index, x=x, edge_attr=edge_attr) 
        local_out = local_out.view(-1, self.hidden_dim) 
        x = self.ffn(local_out) 

        return x 


    def message(self, x_i, x_j, edge_attr, edge_index_i, size_i: Optional[int]): 
        H, C = self.num_heads, self.hidden_dim // self.num_heads 

        x_dst = self.linear_dst(x_i).view(-1, H, C) 
        m_src = self.linear_src_edge(torch.cat([x_j, edge_attr], dim=-1)).view(-1, H, C) 

        alpha = (x_dst * m_src).sum(dim=-1) / math.sqrt(C) 

        alpha = F.leaky_relu(alpha, 0.2) 
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i) 
        alpha = F.dropout(alpha, p=self.local_attn_dropout_ratio, training=self.training) 

        return m_src * alpha.unsqueeze(-1) 

