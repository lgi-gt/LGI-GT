
import torch 
from torch import Tensor  

from torch import nn 
import torch_geometric.nn as gnn 

import torch.nn.functional as F 

from torch_geometric.nn import global_add_pool, global_mean_pool 

from torch_geometric.data import Data 

from tlayer import GraphTransformerEncoderLayer 


class LGI_GT(nn.Module): 

    def __init__(
        self,
        in_channels: int, 
        out_channels: int,
        gconv_dim: int, 
        tlayer_dim: int,
        num_layers: int = 3, 
        num_heads: int = 4, 
        gconv_dropout: float = 0., 
        between_dropout: float = 0., 
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
        self.between_dropout = between_dropout 
        self.tlayer_dropout = tlayer_dropout 
        self.layer_norm = layer_norm 
        self.skip_connection = skip_connection 
        if readout == 'mean': 
            self.readout = global_mean_pool 
        elif readout == 'add': 
            self.readout = global_add_pool 
        elif readout == 'cls': 
            self.CLS = nn.Parameter(torch.randn(1, tlayer_dim)) 
            self.readout = self.CLS_pool 
        else: 
            pass 

        self.lin1 = nn.Linear(in_channels, gconv_dim) 
        self.gconv2tlayer = nn.ModuleList() 
        self.tlayer2gconv = nn.ModuleList() 
        self.lin2 = nn.Linear(gconv_dim, out_channels) 

        self.gconvs1 = nn.ModuleList() 
        self.bns1 = nn.ModuleList() 
        self.bns2 = nn.ModuleList() 
        self.gconvs2 = nn.ModuleList() 
        self.pre_lns = nn.ModuleList() 

        self.tlayers = nn.ModuleList() 

        self.pre_lns.append(nn.LayerNorm(tlayer_dim)) 
        self.pre_lns.append(nn.LayerNorm(tlayer_dim)) 
        for i in range(4): 
            mlp1 = nn.Sequential( 
                    nn.Linear(gconv_dim, gconv_dim), 
                    nn.BatchNorm1d(gconv_dim), 
                    nn.ReLU(), 
                    nn.Linear(gconv_dim, gconv_dim) 
                    ) 
            self.gconvs1.append(gnn.GINConv(mlp1)) 
            self.bns1.append(nn.BatchNorm1d(gconv_dim)) 
            mlp2 = nn.Sequential( 
                    nn.Linear(gconv_dim, gconv_dim), 
                    nn.BatchNorm1d(gconv_dim), 
                    nn.ReLU(), 
                    nn.Linear(gconv_dim, gconv_dim) 
                    )  
            self.gconvs2.append(gnn.GINConv(mlp2)) 
            self.bns2.append(nn.BatchNorm1d(gconv_dim)) 
            
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

    def CLS_pool(self, x, batch): 
        #TODO 
        pass 

    def forward(self, data: Data) -> Tensor: 
        x, edge_index, batch = data.x, data.edge_index, data.batch 
        x = self.lin1(x) 

        out = 0 
        # 4 gconv -> 2 tlayer -> 4 gconv -> 2 tlayer 

        for i in range(4): 
            x = self.gconvs1[i](x, edge_index) 

            x = self.bns1[i](x) 

            out = out + x 
            x = out 

            x = x.relu() 


            x = F.dropout(x, p=self.gconv_dropout, training=self.training) 
        
        graph = (x, edge_index, batch) 
        x = self.tlayers[0](graph) 

        out = out + x 
        x = out 

        graph = (x, edge_index, batch) 
        x = self.tlayers[1](graph) 

        out = out + x 
        x = out 

        for i in range(4):
            x = self.gconvs2[i](x, edge_index) 

            x = self.bns2[i](x) 

            out = out + x 
            x = out 

            x = x.relu() 
            x = F.dropout(x, p=self.gconv_dropout, training=self.training) 

        graph = (x, edge_index, batch) 
        x = self.tlayers[2](graph) 

        out = out + x 
        x = out 

        graph = (x, edge_index, batch) 
        x = self.tlayers[3](graph) 

        out = out + x 
        x = out 
        
        out = self.readout(out, batch) 

        out = self.lin2(out) 

        return out  

