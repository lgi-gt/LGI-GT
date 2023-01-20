
import torch 

from torch import nn 


class RW_StructuralEncoder(nn.Module): 
    """ 
    adapted from GPS code 
    """

    def __init__(self, dim_pe, num_rw_steps, model_type='linear', n_layers=1, norm_type='bn'):
        super().__init__()

        if norm_type == 'bn': 
            self.raw_norm = nn.BatchNorm1d(num_rw_steps)
        else:
            self.raw_norm = None

        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(num_rw_steps, dim_pe))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(num_rw_steps, 2 * dim_pe))
                layers.append(nn.ReLU())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(nn.ReLU())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                # layers.append(nn.ReLU()) 
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(num_rw_steps, dim_pe)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def forward(self, rw):
        if self.raw_norm:
            rw = self.raw_norm(rw)
        rw = self.pe_encoder(rw)  # (Num nodes) x dim_pe

        return rw 


class LapPENodeEncoder(torch.nn.Module):
    """
    adapted from GPS code 
    """

    def __init__(self, dim_in, dim_emb, dim_pe, max_freqs, n_layers=2, post_n_layers=0, norm_type='none', expand_x=True):
        super().__init__()

        if dim_emb - dim_pe < 1:
            raise ValueError(f"LapPE size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x

        # Initial projection of eigenvalue and the node's eigenvector value
        self.linear_A = nn.Linear(2, dim_pe)
        if norm_type == 'bn':
            self.raw_norm = nn.BatchNorm1d(max_freqs)
        else:
            self.raw_norm = None

        # DeepSet model for LapPE
        layers = []
        if n_layers == 1:
            layers.append(nn.ReLU())
        else:
            self.linear_A = nn.Linear(2, 2 * dim_pe)
            layers.append(nn.ReLU())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(2 * dim_pe, dim_pe))
            layers.append(nn.ReLU())
        self.pe_encoder = nn.Sequential(*layers)

        self.post_mlp = None
        if post_n_layers > 0:
            # MLP to apply post pooling
            layers = []
            if post_n_layers == 1:
                layers.append(nn.Linear(dim_pe, dim_pe))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(dim_pe, 2 * dim_pe))
                layers.append(nn.ReLU())
                for _ in range(post_n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(nn.ReLU())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(nn.ReLU())
            self.post_mlp = nn.Sequential(*layers)


    def forward(self, batch):
        if not (hasattr(batch, 'EigVals') and hasattr(batch, 'EigVecs')):
            raise ValueError("Precomputed eigen values and vectors are "
                             f"required for {self.__class__.__name__}; "
                             "set config 'posenc_LapPE.enable' to True")
        EigVals = batch.EigVals
        EigVecs = batch.EigVecs

        if self.training:
            sign_flip = torch.rand(EigVecs.size(1), device=EigVecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            EigVecs = EigVecs * sign_flip.unsqueeze(0)

        pos_enc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2) # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors) x 2

        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 2
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.linear_A(pos_enc)  # (Num nodes) x (Num Eigenvectors) x dim_pe

        # PE encoder: a Transformer or DeepSet model
        pos_enc = self.pe_encoder(pos_enc)

        # Remove masked sequences; must clone before overwriting masked elements
        pos_enc = pos_enc.clone().masked_fill_(empty_mask[:, :, 0].unsqueeze(2),
                                               0.)

        # Sum pooling
        pos_enc = torch.sum(pos_enc, 1, keepdim=False)  # (Num nodes) x dim_pe

        # MLP post pooling
        if self.post_mlp is not None:
            pos_enc = self.post_mlp(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1) 
        return batch
