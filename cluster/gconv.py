

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

import torch.nn.functional as F 

from torch import nn 



class GCNConv_SBMs(MessagePassing): 
    def __init__(self, emb_dim): 
        super().__init__(aggr='add')

        self.linear = nn.Linear(emb_dim, emb_dim)
        self.root_emb = nn.Embedding(1, emb_dim)

        self.edge_encoder = nn.Linear(emb_dim, emb_dim) 

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out 
