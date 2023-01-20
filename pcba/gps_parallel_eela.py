
from typing import Union 

import torch 
from torch import Tensor 

from torch import nn 
import torch.nn.functional as F 

from torch_geometric.nn import global_add_pool, global_mean_pool 
from torch_geometric.data import Batch, Data 
from torch_geometric.utils import to_dense_batch 


from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder 

from gconv import EELA 
from tlayer import GraphTransformerEncoderLayer 


class GraphTransformer(nn.Module): 

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
        self.global_ffn_dropout = global_ffn_dropout 
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

        self.sas = nn.ModuleList() 
        self.lns1 = nn.ModuleList() 
        self.lns2 = nn.ModuleList() 
        self.bns1 = nn.ModuleList() 
        self.bns2 = nn.ModuleList() 
        self.ff_linear1 = nn.ModuleList() 
        self.ff_linear2 = nn.ModuleList() 

        self.tlayers = nn.ModuleList() 
        for i in range(self.num_layers): 
            self.gconvs.append(EELA(gconv_dim, gconv_dim, num_heads, 
                                local_attn_dropout_ratio=local_attn_dropout, 
                                local_ffn_dropout_ratio=local_ffn_dropout)) 
            self.sas.append(nn.MultiheadAttention(tlayer_dim, num_heads, 
                            dropout=global_attn_dropout)) 
            self.lns1.append(nn.LayerNorm(gconv_dim)) 
            self.lns2.append(nn.LayerNorm(gconv_dim)) 
            self.bns1.append(nn.BatchNorm1d(gconv_dim)) 
            self.bns2.append(nn.BatchNorm1d(gconv_dim)) 
            self.ff_linear1.append(nn.Linear(tlayer_dim, 2 * tlayer_dim)) 
            self.ff_linear2.append(nn.Linear(2 * tlayer_dim, tlayer_dim)) 
            # self.tlayers.append( 
            #     GraphTransformerEncoderLayer( 
            #         tlayer_dim, 
            #         num_heads, 
            #         global_attn_dropout_ratio=global_attn_dropout, 
            #         global_ffn_dropout_ratio=global_ffn_dropout, 
            #         clustering=clustering, 
            #         masked_attention=masked_attention, 
            #         norm=norm)) 

    def reset_parameters(self): 
        pass # no need if initilize or construct a new instance every time before used 

    def forward(self, data: Union[Data, Batch]) -> Tensor: 
        x, edge_index, batch = data.x, data.edge_index, data.batch 
        edge_attr = data.edge_attr 

        x = self.node_encoder(x) 
        edge_attr = self.edge_encoder(edge_attr) 

        # if self.readout == 'cls': 
        #     batch_CLS = self.CLS.expand(data.num_graphs, 1, -1)  
        # else: 
        #     batch_CLS = None 

        for i in range(self.num_layers): 
            x0 = x 

            x = self.gconvs[i](x, edge_index, edge_attr) + x 
            x1 = x 

            x = x0 
            x_dense, mask = to_dense_batch(x, batch) 

            x_dense = x_dense.transpose(1, 0) 
            x = self.sas[i](x_dense, x_dense, x_dense, 
                            attn_mask=None, 
                            key_padding_mask=~mask, 
                            need_weights=False)[0] 
            x = x.transpose(1, 0) 
            x = x[mask] 
            x = F.dropout(x, p=self.global_ffn_dropout, training=self.training) 
            x2 = x 

            x = x1 + x2 
            x = self.lns1[i](x) 

            ff_in = x 

            x = self.ff_linear1[i](x) 
            x = F.relu(x) 
            x = F.dropout(x, p=self.global_ffn_dropout, training=self.training) 
            x = self.ff_linear2[i](x) 
            x = F.dropout(x, p=self.global_ffn_dropout, training=self.training) 

            x = x + ff_in 
            x = self.lns2[i](x) 
            
        # if self.readout == 'cls': 
        #     out = batch_CLS.squeeze(1) 
        # else: 
        #     out = self.readout_fn(out, batch) 

        out = self.readout_fn(x, batch) 
        out = self.lin2(out) 

        return out 

