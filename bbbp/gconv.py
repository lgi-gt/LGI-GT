
from torch_geometric.nn import MessagePassing

import torch.nn.functional as F 

from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, \
                        Dropout, LayerNorm    

from torch_geometric.nn.inits import reset 


class GINConv_OGB(MessagePassing):  
    def __init__(self, emb_dim): 
        super().__init__(aggr = "add") 

        self.nn = Sequential(
            Linear(emb_dim, emb_dim), 
            LayerNorm(emb_dim), 
            ReLU(), 
            Linear(emb_dim, emb_dim) )  

    def reset_parameters(self): 
        reset(self.nn) 

    def forward(self, x, edge_index, edge_attr):
        out = self.nn(x + self.propagate(edge_index, x=x, edge_attr=edge_attr)) 
    
        return out

    def message(self, x_j, edge_attr): 
        m = x_j + edge_attr 
        m = F.relu(m) 
        return m 
        
    def update(self, aggr_out):
        return aggr_out

