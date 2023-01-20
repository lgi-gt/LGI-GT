


from typing import Union
from torch import Tensor 
from torch import nn 

from torch_geometric.data import Batch, Data 

from gconv import GCNConv_SBMs 
from tlayer import GraphTransformerEncoderLayer 

from pese import RW_StructuralEncoder, LapPENodeEncoder 


class LGI_GT(nn.Module): 

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
        self.bns1 = nn.ModuleList() 

        self.tlayers = nn.ModuleList() 
        for i in range(num_layers): 
            self.gconvs.append(GCNConv_SBMs(gconv_dim)) 
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

        out = x 
        for i in range(self.num_layers // 4): 

            x = self.gconvs[4*i](x, edge_index, edge_attr) 
            x = self.gconvs[4*i+1](x, edge_index, edge_attr) 
            x = self.gconvs[4*i+2](x, edge_index, edge_attr) 
            x = self.gconvs[4*i+3](x, edge_index, edge_attr) 
            x = out + x 
            x = self.bns1[i](x) 
            out = x 

            graph = (x, batch) 
            x = self.tlayers[i](graph) 

            if self.skip_connection == 'none': 
                # like JK=last, I would rather call it "no skip connection"
                out = x # only last layer to output, consequently: identical 
            elif self.skip_connection == 'long': 
                # like JK=add, I would rather call it "summation long skip connection to the output"
                out = out + x  # long skip connection to output: hierarchical 
            elif self.skip_connection == 'short': 
                # normal residual, I would rather call it "summantion short skip connection in every layer"
                x = x + out 
                out = x 

        out = self.lin2(out) 

        return out 


