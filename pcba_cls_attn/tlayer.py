
import math 

import torch 
from torch import Tensor 

from torch import nn 
from torch_geometric.utils import to_dense_batch 


class GraphTransformerEncoderLayer(nn.Module): 

    def __init__(self, hidden_dim: int, num_heads: int, 
            global_attn_dropout_ratio: float = 0.0, 
            global_ffn_dropout_ratio: float = 0.0, 
            clustering: bool = True, 
            masked_attention: bool = False, 
            norm: str = 'ln'): 

        super().__init__() 

        self.hidden_dim = hidden_dim 
        self.num_heads = num_heads 
        self.global_attn_dropout = nn.Dropout(global_attn_dropout_ratio) 
        self.clustering = clustering 
        self.masked_attention = masked_attention 

        self.linear_Q = nn.Linear(hidden_dim, hidden_dim) 
        self.linear_K = nn.Linear(hidden_dim, hidden_dim) 
        self.linear_V = nn.Linear(hidden_dim, hidden_dim) 

        self.ATTN_OUT = nn.Linear(hidden_dim, hidden_dim) 
        self.attn_out = nn.Linear(hidden_dim, hidden_dim) 

        if norm == 'ln': 
            # self.norm1 = nn.LayerNorm(hidden_dim) 
            # self.norm2 = nn.LayerNorm(hidden_dim) 
            self.NORM1 = nn.LayerNorm(hidden_dim) 
            self.NORM2 = nn.LayerNorm(hidden_dim) 
        else: 
            # self.norm1 = nn.BatchNorm1d(hidden_dim) 
            # self.norm2 = nn.BatchNorm1d(hidden_dim) 
            self.NORM1 = nn.BatchNorm1d(hidden_dim) 
            self.NORM2 = nn.BatchNorm1d(hidden_dim) 
        
        self.dropout_global = nn.Dropout(global_ffn_dropout_ratio) 
        self.FFN = nn.Sequential( 
            nn.Linear(hidden_dim, 2*hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(global_ffn_dropout_ratio), 
            nn.Linear(2*hidden_dim, hidden_dim), 
            nn.Dropout(global_ffn_dropout_ratio) 
        ) 

    def reset_parameters(self):
        pass 

    def forward(
        self,
        graph, 
        batch_CLS 
    ) -> Tensor: # B x N x F -> B x N x F 

        x, batch = graph 

        # Sparse -> Dense 
        x_dense, mask = to_dense_batch(x, batch) 
        attention_mask = mask 

        if batch_CLS is not None: 
            x_dense = torch.cat([x_dense, batch_CLS], dim=1) 
            CLS_mask = torch.full((batch_CLS.shape[0], 1), True, device=batch_CLS.device) 
            attention_mask = torch.cat([mask, CLS_mask], dim=1) 

        Q = self.linear_Q(x_dense) 
        K = self.linear_K(x_dense) 
        V = self.linear_V(x_dense) 

        dim_split = self.hidden_dim // self.num_heads 
        Q_heads = torch.cat(Q.split(dim_split, 2), dim=0) 
        K_heads = torch.cat(K.split(dim_split, 2), dim=0) 
        V_heads = torch.cat(V.split(dim_split, 2), dim=0) 
        
        attention_score = Q_heads.bmm(K_heads.transpose(1, 2)) 
        # of size (B x H, N, N) if batch_CLS is not None else (B x H, N+1, N+1) 

        attention_score = attention_score / math.sqrt(self.hidden_dim // self.num_heads) 

        if self.clustering: 
            if self.masked_attention:
                inf_mask = (~attention_mask).unsqueeze(-1).to(dtype=torch.float) * -1e9 
                inf_mask = torch.cat([inf_mask for _ in range(self.num_heads)], 0) 
                A = torch.softmax(attention_score + inf_mask, 1) 
            else: 
                A = torch.softmax(attention_score, 1) 
        else: 
            if self.masked_attention: 
                inf_mask = (~attention_mask).unsqueeze(1).to(dtype=torch.float) * -1e9
                inf_mask = torch.cat([inf_mask for _ in range(self.num_heads)], 0) 
                A = torch.softmax(attention_score + inf_mask, -1) 
            else: 
                A = torch.softmax(attention_score, -1) 
        
        cls_attn = A[:, -1, :].clone().detach() # 8 heads 
            
        A = self.global_attn_dropout(A) 
        global_out = torch.cat((A.bmm(V_heads)).split(Q.size(0), 0), 2) # (B, N+1, F) 

        attention_out = self.ATTN_OUT(global_out) 
        attention_out = self.dropout_global(attention_out) 
        attention_out = attention_out + x_dense 

        ######### change to sparse 
        if batch_CLS is not None: 
            batch_CLS = attention_out[:, -1, :] 
            attention_out = attention_out[:, :-1, :] 
        
        attention_out = attention_out[mask] 

        if batch_CLS is not None: 
            attention_out = torch.cat((attention_out, batch_CLS), dim=0) 

        attention_out = self.NORM1(attention_out) 
        
        out = self.FFN(attention_out) + attention_out 
        out = self.NORM2(out) 

        return out, cls_attn 

