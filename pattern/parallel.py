

from typing import List, Optional, Tuple, Type, Union

from torch import Tensor 
from torch import nn 
import torch.nn.functional as F 

from torch_geometric.data import Batch, Data 
from torch_geometric.utils import to_dense_batch 

from gconv import GCNConv_SBMs 
from tlayer import GraphTransformerEncoderLayer 
from pese import RW_StructuralEncoder, LapPENodeEncoder 

class GraphTransformer(nn.Module): 

    def __init__(
        self, 
        in_dim: int, 
        out_dim: str, 
        gconv_dim: int, 
        tlayer_dim: int, 
        pese_type: str, 
        dim_pe: int, 
        num_rw_steps: int, 
        max_freqs: int, 
        num_layers: int = 4, 
        num_heads: int = 4, 
        gconv_dropout: float = 0., 
        attn_dropout: float = 0., 
        tlayer_dropout: float = 0., 
        classifier_head_dropout: float = 0., 
        clustering: bool = True, 
        masked_attention: bool = False, 
        layer_norm: bool = False, 
        skip_connection: str = 'none' 
    ):
        super().__init__()
        self.gconv_dim = gconv_dim 
        self.tlayer_dim = tlayer_dim 
        self.pese_type = pese_type 
        self.num_layers = num_layers 
        self.num_heads = num_heads 
        self.gconv_dropout = gconv_dropout 
        self.tlayer_dropout = tlayer_dropout 
        self.layer_norm = layer_norm 
        self.skip_connection = skip_connection 
        
        if pese_type == 'RWSE': 
            # self.node_encoder = nn.Embedding(in_dim, gconv_dim - dim_pe) 
            self.node_encoder = nn.Linear(in_dim, gconv_dim - dim_pe) 
            self.se_encoder = RW_StructuralEncoder(dim_pe, num_rw_steps, 
                                    model_type='linear', 
                                    n_layers=2, 
                                    norm_type='bn') 
        elif pese_type == 'LapPE': 
            self.node_se_encoder = LapPENodeEncoder(in_dim, gconv_dim, dim_pe, max_freqs) 

        self.edge_encoder = nn.Embedding(1, gconv_dim) 

        self.lin2 = nn.Linear(tlayer_dim, out_dim) 
        
        self.gconvs = nn.ModuleList() 
        self.sas = nn.ModuleList() 
        self.bns1 = nn.ModuleList() 
        self.bns2 = nn.ModuleList() 
        self.bns3 = nn.ModuleList() 
        self.ff_linear1 = nn.ModuleList() 
        self.ff_linear2 = nn.ModuleList() 

        self.tlayers = nn.ModuleList() 
        for i in range(num_layers): 
            self.gconvs.append(GCNConv_SBMs(gconv_dim)) 
            self.bns1.append(nn.BatchNorm1d(gconv_dim)) 
            self.bns2.append(nn.BatchNorm1d(gconv_dim)) 
            self.bns3.append(nn.BatchNorm1d(tlayer_dim)) 
            self.sas.append(nn.MultiheadAttention(tlayer_dim, num_heads, 
                            dropout=attn_dropout)) 
            self.ff_linear1.append(nn.Linear(tlayer_dim, 2 * tlayer_dim)) 
            self.ff_linear2.append(nn.Linear(2 * tlayer_dim, tlayer_dim)) 
            # self.tlayers.append( 
            #     GraphTransformerEncoderLayer( 
            #         tlayer_dim, 
            #         num_heads, 
            #         attn_dropout_ratio=attn_dropout, 
            #         dropout_ratio=self.tlayer_dropout, 
            #         clustering=clustering, 
            #         masked_attention=masked_attention, 
            #         layer_norm=layer_norm)) 

    def reset_parameters(self): 
        pass # no need if initilize or construct a new instance every time before used 

    def forward(self, data: Union[Data, Batch]) -> Tensor: 
        if self.pese_type == 'RWSE': 
            x = data.x.squeeze(-1) 
            h = self.node_encoder(x) 
            data.x = h 
            data = self.se_encoder(data) 
        elif self.pese_type == 'LapPE': 
            data = self.node_se_encoder(data) 

        x, edge_index, batch = data.x, data.edge_index, data.batch 
        edge_attr = data.edge_attr 
        edge_attr = self.edge_encoder(edge_attr) 

        for i in range(self.num_layers): 
            x0 = x 

            x = self.gconvs[i](x, edge_index, edge_attr) 
            x = x + x0 
            x = self.bns1[i](x) 
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
            # graph = (x, batch) 
            # x = self.tlayers[i](graph) 

            x = F.dropout(x, p=self.tlayer_dropout, training=self.training) 
            x2 = x 

            x = x1 + x2 
            x = self.bns2[i](x) 

            ff_in = x 

            x = self.ff_linear1[i](x) 
            x = F.relu(x) 
            x = F.dropout(x, p=self.tlayer_dropout, training=self.training) 
            x = self.ff_linear2[i](x) 
            x = F.dropout(x, p=self.tlayer_dropout, training=self.training) 

            x = x + ff_in 
            x = self.bns3[i](x) 

        out = self.lin2(x) 

        return out 

