import torch
import torch.nn as nn

# feature extractor
from diffusion_net.layers import DiffusionNet

# maps block
from utils import get_batch_mask


class RegularizedFMNet(nn.Module):
    """Compute the functional map matrix representation for batched inputs."""

    def __init__(self, lambda_=1e-3, resolvant_gamma=0.5):
        super().__init__()
        self.lambda_ = lambda_
        self.resolvant_gamma = resolvant_gamma

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        # Ensure the inputs are compatible with batch processing
        # The squeeze(0) and unsqueeze(0) operations are removed since we're now handling batches natively

        # Perform batch matrix multiplication
        F_hat = torch.bmm(evecs_trans_x, feat_x)
        G_hat = torch.bmm(evecs_trans_y, feat_y)
        A, B = F_hat, G_hat

        # Compute masks for the regularization term
        D12 = get_batch_mask(evals_x, evals_y, self.resolvant_gamma, feat_x.device)
        D21 = get_batch_mask(evals_y, evals_x, self.resolvant_gamma, feat_y.device)

        A_t, B_t = A.transpose(1, 2), B.transpose(1, 2)
        A_A_t, B_B_t = torch.bmm(A, A_t), torch.bmm(B, B_t)
        B_A_t, A_B_t = torch.bmm(B, A_t), torch.bmm(A, B_t)

        # Process each dimension of evals_x and evals_y in a batch-friendly manner
        C12 = self.compute_C(A_A_t, B_A_t, D12, self.lambda_)
        C21 = self.compute_C(B_B_t, A_B_t, D21, self.lambda_)

        return [C12, C21]

    @staticmethod
    def compute_C(AA_t, BA_t, D, lambda_):
        C_i = []
        for i in range(AA_t.size(1)):
            D_i = torch.stack([torch.diag(D[bs, i, :]) for bs in range(D.size(0))])
            C = torch.bmm(torch.inverse(AA_t + lambda_ * D_i), BA_t[:, i, :].unsqueeze(2))
            C_i.append(C.transpose(1, 2))
        C = torch.cat(C_i, dim=1)

        return C


class RegularizedCFMNet(nn.Module):
    """Compute the complex functional map matrix representation."""

    def __init__(self, lambda_=1e-3, resolvant_gamma=0.5):
        super().__init__()
        self.lambda_ = lambda_
        self.resolvant_gamma = resolvant_gamma

    def forward(self, feat_x, feat_y, spec_grad_x, spec_grad_y, cevals_x, cevals_y):
        # Ensure complex data type
        cty = torch.complex128

        # Use `bmm` for batch-wise matrix multiplication
        F_hat = torch.bmm(spec_grad_x, feat_x.type(cty))
        G_hat = torch.bmm(spec_grad_y, feat_y.type(cty))
        A, B = F_hat, G_hat

        if self.lambda_ == 0:
            Q = torch.bmm(B, torch.pinverse(A))
            return Q

        D = get_batch_mask(cevals_x, cevals_y, self.resolvant_gamma, feat_x.device)

        A_t = torch.conj(A.transpose(1, 2))
        A_A_t = torch.bmm(A, A_t)
        B_A_t = torch.bmm(B, A_t)

        Q_i = []
        for i in range(cevals_x.size(1)):
            D_i = torch.stack([torch.diag(D[bs, i, :]) for bs in range(cevals_x.size(0))])
            Q = torch.bmm(torch.inverse(A_A_t + self.lambda_ * D_i), torch.conj(B_A_t[:, i, :].unsqueeze(2)))
            Q_i.append(torch.conj(Q.transpose(1, 2)))
        Q = torch.cat(Q_i, dim=1)

        return Q


class DQFMNet(nn.Module):
    """
    Compilation of the global model :
    - diffusion net as feature extractor
    - fmap + q-fmap
    - unsupervised loss
    """

    def __init__(self, C_in, C_width, N_block, n_feat, mlp_hidden_dims, lambda_, resolvant_gamma, n_fmap, n_cfmap,
                 robust):
        super().__init__()

        # feature extractor #
        with_grad = True
        self.C_in = C_in

        self.feature_extractor = DiffusionNet(
            C_in=C_in,
            C_out=n_feat,
            C_width=C_width,
            N_block=N_block,
            dropout=True,
            mlp_hidden_dims=mlp_hidden_dims,
            with_gradient_features=with_grad,
            with_gradient_rotations=with_grad,
        )

        # regularized fmap
        self.fmreg_net = RegularizedFMNet(lambda_=lambda_,
                                          resolvant_gamma=resolvant_gamma)
        self.cfmreg_net = RegularizedCFMNet(lambda_=lambda_,
                                            resolvant_gamma=resolvant_gamma)

        # parameters
        self.n_fmap = n_fmap
        self.n_cfmap = n_cfmap
        self.robust = robust

    def forward(self, batch):
        # Extract shape1 and shape2 data from batch dictionary
        shape1_data = batch["shape1"]
        shape2_data = batch["shape2"]

        # Get features
        if self.C_in == 3:
            features1, features2 = shape1_data["xyz"], shape2_data["xyz"]
        else:
            features1, features2 = shape1_data["wks"], shape2_data["wks"]

        # Feature extraction
        feat1 = self.feature_extractor(features1, shape1_data["mass"], L=shape1_data["L"],
                                       evals=shape1_data["evals"], evecs=shape1_data["evecs"],
                                       gradX=shape1_data["gradX"], gradY=shape1_data["gradY"],
                                       faces=shape1_data["faces"])

        feat2 = self.feature_extractor(features2, shape2_data["mass"], L=shape2_data["L"],
                                       evals=shape2_data["evals"], evecs=shape2_data["evecs"],
                                       gradX=shape2_data["gradX"], gradY=shape2_data["gradY"],
                                       faces=shape2_data["faces"])

        # Prepare data for regularized fmap
        evecs_trans1, evecs_trans2 = shape1_data["evecs_trans"], shape2_data["evecs_trans"]

        evals1, evals2 = shape1_data["evals"][:, :self.n_fmap], shape2_data["evals"][:, :self.n_fmap]
        evecs1, evecs2 = shape1_data["evecs"][:, :, :self.n_fmap], shape2_data["evecs"][:, :, :self.n_fmap]

        #
        C12_pred, C21_pred = self.fmreg_net(feat1, feat2, evals1, evals2, evecs_trans1, evecs_trans2)
        #

        # if we don't have complex spectral info we just return C
        if self.n_cfmap == 0:
            return C12_pred, C21_pred, None, feat1, feat2, evecs_trans1, evecs_trans2, evecs1, evecs2

        # Prepare data for cfmap prediction
        spec_grad1 = shape1_data["spec_grad"][:, :self.n_cfmap]
        spec_grad2 = shape2_data["spec_grad"][:, :self.n_cfmap]

        cevals1, cevals2 = shape1_data["cevals"][:, :self.n_fmap], shape2_data["cevals"][:, :self.n_fmap]

        # Get cfmap prediction
        cfeat1, cfeat2 = feat1, feat2  # network features
        Q_pred = self.cfmreg_net(cfeat1, cfeat2,
                                 spec_grad1, spec_grad2,
                                 cevals1, cevals2)

        return C12_pred, C21_pred, Q_pred, feat1, feat2, evecs_trans1, evecs_trans2, evecs1, evecs2
