
import math 
from typing import Optional, Tuple 

import torch 
from torch import Tensor 
from torch.nn import LayerNorm, Linear, Dropout, Module

from torch_geometric.utils import to_dense_batch 


class MultiHeadAttention(Module): 

    def __init__(self, hidden_dim: int, num_heads: int, 
            dropout_ratio: float = 0.):

        super().__init__() 
        self.hidden_dim = hidden_dim   
        self.num_heads = num_heads 
        self.dropout = Dropout(dropout_ratio) 

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

        # inf_mask = (~mask).unsqueeze(1).to(dtype=torch.float) * -1e9 
        inf_mask = (~mask).unsqueeze(-1).to(dtype=torch.float) * -1e9 

        dim_split = self.hidden_dim // self.num_heads
        Q_heads = torch.cat(Q.split(dim_split, 2), dim=0)
        K_heads = torch.cat(K.split(dim_split, 2), dim=0)
        V_heads = torch.cat(V.split(dim_split, 2), dim=0)

        if inf_mask is not None: 
            inf_mask = torch.cat([inf_mask for _ in range(self.num_heads)], 0) 
            attention_score = Q_heads.bmm(K_heads.transpose(1, 2)) 
            attention_score = attention_score / math.sqrt(self.hidden_dim) 

            A = torch.softmax(attention_score, 1) # clustering, no infinity mask 
            # A = torch.softmax(inf_mask + attention_score, 1) # clustering 
            # A = torch.softmax(inf_mask + attention_score, -1) # no clustering 
            # A = torch.softmax(attention_score, -1) # no clustering, no infinity mask 
        # else: 
        #     A = torch.softmax( 
        #         Q_heads.bmm(K_heads.transpose(1, 2)) / math.sqrt(self.hidden_dim), 1)

        A = self.dropout(A) 
        out = torch.cat((A.bmm(V_heads)).split(Q.size(0), 0), 2) 
        out = self.linear_attn_out(out) 
        
        return out


class NodeSelfAttention(MultiHeadAttention): 

    def __init__(self, hidden_dim: int, num_heads: int, 
            dropout_ratio: float = 0.):
        super().__init__(hidden_dim, num_heads, dropout_ratio)

    def forward(
        self,
        x: Tensor, 
        mask: Tensor 
    ) -> Tensor: 

        return super().forward(x, x, x, mask) 
        

class GraphTransformerEncoderLayer(Module): 
    
    def __init__(self, hidden_dim, num_heads: int, 
            dropout_ratio: float = 0.0, layer_norm: bool = False): 
        
        super().__init__() 
        self.hidden_dim = hidden_dim 
        self.num_heads = num_heads 
        self.layer_norm = layer_norm 
        self.dropout = Dropout(dropout_ratio) 
        
        self.node_self_attention = NodeSelfAttention( 
            hidden_dim, num_heads, dropout_ratio) 
        
        self.linear_out1 = Linear(hidden_dim, hidden_dim) 
        self.linear_out2 = Linear(hidden_dim, hidden_dim) 

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

        # Dense -> Sparse 
        out = out[mask] 
        return out 