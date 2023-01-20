
from typing import Union 

import torch 
from torch import Tensor 

from torch import nn 

from torch_geometric.nn import global_add_pool, global_mean_pool 
from torch_geometric.data import Batch, Data 

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder 

from gconv import EELA 
from tlayer import GraphTransformerEncoderLayer 


class LGI_GT(nn.Module): 

    def __init__(
        self, 
        out_dim: int, 
        gconv_dim: int, 
        tlayer_dim: int, 
        num_layers: int = 4, 
        num_heads: int = 4, 
        local_attn_dropout: float = 0., 
        global_attn_dropout: float = 0., 
        local_ffn_dropout: float = 0., 
        global_ffn_dropout: float = 0., 
        clustering: bool = True, 
        masked_attention: bool = False, 
        norm: str = 'ln', 
        skip_connection: str = 'none', 
        readout: str = 'mean' 
    ):
        super().__init__()
        self.gconv_dim = gconv_dim 
        self.tlayer_dim = tlayer_dim
        self.num_layers = num_layers 
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

        self.node_encoder = AtomEncoder(gconv_dim) 
        self.edge_encoder = BondEncoder(gconv_dim) 

        self.lin2 = nn.Linear(tlayer_dim, out_dim) 
        
        self.gconvs = nn.ModuleList() 

        self.tlayers = nn.ModuleList() 
        for i in range(self.num_layers): 
            self.gconvs.append(EELA(gconv_dim, gconv_dim, num_heads, 
                                local_attn_dropout_ratio=local_attn_dropout, 
                                local_ffn_dropout_ratio=local_ffn_dropout)) 
            self.tlayers.append( 
                GraphTransformerEncoderLayer( 
                    tlayer_dim, 
                    num_heads, 
                    global_attn_dropout_ratio=global_attn_dropout, 
                    global_ffn_dropout_ratio=global_ffn_dropout, 
                    clustering=clustering, 
                    masked_attention=masked_attention, 
                    norm=norm)) 

    def reset_parameters(self): 
        pass # no need if initilize or construct a new instance every time before used 

    def forward(self, data: Union[Data, Batch]) -> Tensor: 
        x, edge_index, batch = data.x, data.edge_index, data.batch 
        edge_attr = data.edge_attr 

        x = self.node_encoder(x) 
        edge_attr = self.edge_encoder(edge_attr) 

        if self.readout == 'cls': 
            batch_CLS = self.CLS.expand(data.num_graphs, 1, -1)  
        else: 
            batch_CLS = None 

        out = 0 
        for i in range(self.num_layers): 
            x = self.gconvs[i](x, edge_index, edge_attr) 

            graph = (x, batch) 
            x, cls_attn = self.tlayers[i](graph, batch_CLS) 

            if self.readout == 'cls': 
                batch_CLS = x[-data.num_graphs:].unsqueeze(1) 
                x = x[:-data.num_graphs] 
            
            x = x + out 
            out = x 
            
        if self.readout == 'cls': 
            out = batch_CLS.squeeze(1) 
        else: 
            out = self.readout_fn(out, batch) 

        out = self.lin2(out) 

        return out 
    
    
    def get_clf_attn(self, data, num_layer): 
        if num_layer > self.num_layers: 
            return None 

        x, edge_index, batch = data.x, data.edge_index, data.batch 
        edge_attr = data.edge_attr 

        x = self.node_encoder(x) 
        edge_attr = self.edge_encoder(edge_attr) 

        if self.readout == 'cls': 
            batch_CLS = self.CLS.expand(data.num_graphs, 1, -1)  
        else: 
            batch_CLS = None 

        out = 0 
        for i in range(self.num_layers): 
            x = self.gconvs[i](x, edge_index, edge_attr) 

            graph = (x, batch) 
            x, cls_attn = self.tlayers[i](graph, batch_CLS) 

            if i + 1 == num_layer: 
                return cls_attn 

            if self.readout == 'cls': 
                batch_CLS = x[-data.num_graphs:].unsqueeze(1) 
                x = x[:-data.num_graphs] 
            
            x = x + out 
            out = x 

        return None 
