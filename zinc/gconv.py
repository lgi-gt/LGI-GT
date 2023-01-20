

from torch import nn 
import torch_geometric.nn as gnn 

import torch.nn.functional as F 


class GINConv_ZINC(gnn.MessagePassing): 
    def __init__(self, emb_dim, num_layers=2, norm='bn'): 

        super().__init__(aggr = "add")

        self.mlp = nn.Sequential( 
            nn.Linear(emb_dim, 2*emb_dim), 
            nn.ReLU(), 

            nn.Linear(2*emb_dim, emb_dim) 
        ) 
        self.edge_linear = nn.Linear(emb_dim, emb_dim) 

    def forward(self, x, edge_index, edge_attr): 
        edge_embedding = self.edge_linear(edge_attr) 
        out = self.mlp(x + self.propagate(edge_index, x=x, edge_attr=edge_embedding)) 

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr) 

    def update(self, aggr_out):
        return aggr_out

