

from typing import Union 

import torch 
from torch import Tensor 

from torch import nn 

from torch_geometric.nn import global_add_pool, global_mean_pool 
from torch_geometric.data import Batch, Data 

from gconv import GINConv_ZINC 
from tlayer import GraphTransformerEncoderLayer 
from pese import RW_StructuralEncoder 


class LGI_GT(nn.Module): 

    def __init__(
        self, 
        node_num_types: int, 
        edge_num_types: int, 
        dim_pe: int, 
        num_rw_steps: int, 
        out_dim: int, 
        gconv_dim: int, 
        tlayer_dim: int, 
        num_layers: int = 4, 
        num_heads: int = 4, 
        gconv_dropout: float = 0., 
        attn_dropout: float = 0., 
        tlayer_dropout: float = 0., 
        classifier_head_dropout: float = 0., 
        clustering: bool = True, 
        masked_attention: bool = False, 
        layer_norm: bool = False, 
        skip_connection: str = 'none', 
        readout: str = 'mean' 
    ):
        super().__init__()
        self.gconv_dim = gconv_dim 
        self.tlayer_dim = tlayer_dim
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

        self.node_encoder = nn.Embedding(node_num_types, gconv_dim - dim_pe) 
        # self.node_encoder = nn.Embedding(node_num_types, gconv_dim) 
        self.se_encoder = RW_StructuralEncoder(dim_pe, num_rw_steps, 
                                model_type='linear', 
                                n_layers=2, 
                                norm_type='bn') 
        self.edge_encoder = nn.Embedding(edge_num_types, gconv_dim) 

        self.lin2 = nn.Sequential( 
            nn.Linear(tlayer_dim, 8 * tlayer_dim), 
            nn.ReLU(), 
            
            nn.Linear(8 * tlayer_dim, 8 * tlayer_dim), 
            nn.ReLU(), 

            nn.Linear(8 * tlayer_dim, out_dim) 
        ) 
        
        self.gconvs = nn.ModuleList() 
        self.bns1 = nn.ModuleList() 

        self.tlayers = nn.ModuleList() 
        for i in range(num_layers): 
            self.gconvs.append(GINConv_ZINC(gconv_dim)) 
            self.bns1.append(nn.BatchNorm1d(gconv_dim)) 
            self.tlayers.append( 
                GraphTransformerEncoderLayer( 
                    tlayer_dim, 
                    num_heads, 
                    attn_dropout_ratio=attn_dropout, 
                    dropout_ratio=self.tlayer_dropout, 
                    clustering=clustering, 
                    masked_attention=masked_attention, 
                    layer_norm=layer_norm)) 

    def reset_parameters(self): 
        pass # no need if initilize or construct a new instance every time before used 

    def forward(self, data: Union[Data, Batch]) -> Tensor: 
        x, edge_index, batch = data.x, data.edge_index, data.batch 
        rw = data.pestat_RWSE 
        edge_attr = data.edge_attr 

        x = x.squeeze(-1) 
        x = self.node_encoder(x) 
        rwse = self.se_encoder(rw) 
        x = torch.cat((x, rwse), dim=1) 
        
        edge_attr = self.edge_encoder(data.edge_attr) 

        out = x 
        for i in range(self.num_layers): 

            x = self.gconvs[i](x, edge_index, edge_attr) 
            x = out + x 
            x = self.bns1[i](x) 
            out = x 

            graph = (x, batch, None) 
            x = self.tlayers[i](graph) 

            x = out + x 
            out = x 

        out = self.readout_fn(out, batch) 
        out = self.lin2(out) 

        return out 

