import random
import numpy as np
import torch
import torch.nn as nn


class FrobeniusLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        loss = torch.sum(torch.abs(a - b) ** 2, axis=(1, 2))
        return torch.mean(loss)


class DQFMLoss(nn.Module):
    def __init__(self, w_gt=False, w_ortho=1, w_Qortho=1, w_bij=1, w_res=1, w_rank=-0.01):
        super().__init__()

        # loss HP
        self.w_gt = w_gt
        self.w_ortho = w_ortho
        self.w_Qortho = w_Qortho
        self.w_bij = w_bij
        self.w_res = w_res
        self.w_rank = w_rank
        # frob loss function
        self.frob_loss = FrobeniusLoss()

        # different losses
        self.gt_loss = 0
        self.gt_old_loss = 0
        self.ortho_loss = 0
        self.Qortho_loss = 0
        self.bij_loss = 0
        self.res_loss = 0
        self.rank_loss = 0

    def forward(self, C12_gt, C21_gt, C12, C21, C12_new, C21_new, Q12, feat1, feat2, evecs_trans1, evecs_trans2):
        loss = 0

        # gt loss (if we train on ground-truth then return directly)
        self.gt_old_loss = (self.frob_loss(C12, C12_gt) / self.frob_loss(0, C12_gt) + self.frob_loss(C21,
                                                                                                     C21_gt) / self.frob_loss(
            0, C21_gt)) / 2
        self.gt_loss = (self.frob_loss(C12_new, C12_gt) / self.frob_loss(0, C12_gt) + self.frob_loss(C21_new,
                                                                                                     C21_gt) / self.frob_loss(
            0, C21_gt)) / 2
        if self.w_gt:
            loss = self.gt_loss
            return loss

        # fmap ortho loss
        if self.w_ortho > 0:
            I = torch.eye(C12.shape[1]).unsqueeze(0).to(C12.device)
            CCt12 = C12 @ C12.transpose(1, 2)
            CCt21 = C21 @ C21.transpose(1, 2)

            CCt12_new = C12_new @ C12_new.transpose(1, 2)
            CCt21_new = C21_new @ C21_new.transpose(1, 2)
            self.ortho_loss = (self.frob_loss(CCt12, I) + self.frob_loss(CCt21, I) +
                               self.frob_loss(CCt12_new, I) + self.frob_loss(CCt21_new, I)) * self.w_ortho / 2
            # self.ortho_loss = (self.frob_loss(CCt12, I) + self.frob_loss(CCt21, I)) * self.w_ortho
            loss += self.ortho_loss

        # fmap bij loss
        if self.w_bij > 0:
            I = torch.eye(C12.shape[1]).unsqueeze(0).to(C12.device)
            self.bij_loss = (self.frob_loss(torch.bmm(C12, C21), I) + self.frob_loss(torch.bmm(C21, C12),
                                                                                     I)) * self.w_bij
            # loss += self.bij_loss

        if self.w_res > 0:
            self.res_loss = self.frob_loss(C12, C12_new) + self.frob_loss(C21, C21_new)
            self.res_loss *= self.w_res
            loss += self.res_loss

        if self.w_rank < 0:
            F_hat = torch.bmm(evecs_trans1, feat1)
            G_hat = torch.bmm(evecs_trans2, feat2)
            F = F_hat @ F_hat.transpose(1, 2)
            G = G_hat @ G_hat.transpose(1, 2)
            I = torch.eye(F.shape[1]).unsqueeze(0).to(F.device)
            rank_pen = 0
            for i in range(F_hat.shape[0]):
                rank_pen += F_hat[i].norm(p='nuc') + G_hat[i].norm(p='nuc')
            self.rank_loss = rank_pen
            # self.rank_loss = self.frob_loss(F+G, 2*I)
            self.rank_loss *= self.w_rank
            # loss += self.rank_loss

        # qfmap ortho loss
        if Q12 is not None and self.w_Qortho > 0:
            I = torch.eye(Q12.shape[1]).unsqueeze(0).to(Q12.device)
            CCt = Q12 @ torch.conj(Q12.transpose(1, 2))
            self.Qortho_loss = self.frob_loss(CCt, I) * self.w_Qortho
            loss += self.Qortho_loss

        return [loss, self.gt_old_loss, self.gt_loss, self.ortho_loss, self.bij_loss, self.res_loss, self.rank_loss]


def get_mask(evals1, evals2, gamma=0.5, device="cpu"):
    scaling_factor = max(torch.max(evals1), torch.max(evals2))
    evals1, evals2 = evals1.to(device) / scaling_factor, evals2.to(device) / scaling_factor
    evals_gamma1, evals_gamma2 = (evals1 ** gamma)[None, :], (evals2 ** gamma)[:, None]

    M_re = evals_gamma2 / (evals_gamma2.square() + 1) - evals_gamma1 / (evals_gamma1.square() + 1)
    M_im = 1 / (evals_gamma2.square() + 1) - 1 / (evals_gamma1.square() + 1)
    return M_re.square() + M_im.square()


def farthest_point_sample(xyz, ratio):
    xyz = xyz.t().unsqueeze(0)
    npoint = int(ratio * xyz.shape[1])
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids[0]


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def nn_interpolate(desc, xyz, dists, idx, idf):
    xyz = xyz.unsqueeze(0)
    B, N, _ = xyz.shape
    mask = torch.from_numpy(np.isin(idx.numpy(), idf.numpy())).int()
    mask = torch.argsort(mask, dim=-1, descending=True)[:, :, :3]
    dists, idx = torch.gather(dists, 2, mask), torch.gather(idx, 2, mask)
    transl = torch.arange(dists.size(1))
    transl[idf.flatten()] = torch.arange(idf.flatten().size(0))
    shape = idx.shape
    idx = transl[idx.flatten()].reshape(shape)
    dists, idx = dists.to(desc.device), idx.to(desc.device)

    dist_recip = 1.0 / (dists + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    interpolated_points = torch.sum(index_points(desc, idx) * weight.view(B, N, 3, 1), dim=2)

    return interpolated_points


def euler_angles_to_rotation_matrix(theta):
    R_x = torch.tensor(
        [[1, 0, 0], [0, torch.cos(theta[0]), -torch.sin(theta[0])], [0, torch.sin(theta[0]), torch.cos(theta[0])]])
    R_y = torch.tensor(
        [[torch.cos(theta[1]), 0, torch.sin(theta[1])], [0, 1, 0], [-torch.sin(theta[1]), 0, torch.cos(theta[1])]])
    R_z = torch.tensor(
        [[torch.cos(theta[2]), -torch.sin(theta[2]), 0], [torch.sin(theta[2]), torch.cos(theta[2]), 0], [0, 0, 1]])

    matrices = [R_x, R_y, R_z]

    R = torch.mm(matrices[2], torch.mm(matrices[1], matrices[0]))
    return R


def get_random_rotation(x, y, z):
    thetas = torch.zeros(3, dtype=torch.float)
    degree_angles = [x, y, z]
    for axis_ind, deg_angle in enumerate(degree_angles):
        rand_deg_angle = random.random() * 2 * deg_angle - deg_angle
        rand_radian_angle = float(rand_deg_angle * np.pi) / 180.0
        thetas[axis_ind] = rand_radian_angle

    return euler_angles_to_rotation_matrix(thetas)


def data_augmentation(verts, rot_x=0, rot_y=90, rot_z=0, std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1):
    # random rotation
    rotation_matrix = get_random_rotation(rot_x, rot_y, rot_z).to(verts.device)
    verts = verts @ rotation_matrix.T

    # random noise
    noise = std * torch.randn(verts.shape).to(verts.device)
    noise = noise.clamp(-noise_clip, noise_clip)
    verts += noise

    # random scaling
    scales = [scale_min, scale_max]
    scale = scales[0] + torch.rand((3,)) * (scales[1] - scales[0])
    verts = verts * scale.to(verts.device)

    return verts


def augment_batch(data, rot_x=0, rot_y=90, rot_z=0, std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1):
    data["shape1"]["xyz"] = data_augmentation(data["shape1"]["xyz"], rot_x, rot_y, rot_z, std, noise_clip, scale_min,
                                              scale_max)
    data["shape2"]["xyz"] = data_augmentation(data["shape2"]["xyz"], rot_x, rot_y, rot_z, std, noise_clip, scale_min,
                                              scale_max)

    return data


def data_augmentation_sym(shape):
    """
    we symmetrise the shape which results in conjugation of complex info
    """
    shape["gradY"] = -shape["gradY"]  # gradients get conjugated

    # so should complex data (to double check)
    shape["cevecs"] = torch.conj(shape["cevecs"])
    shape["spec_grad"] = torch.conj(shape["spec_grad"])
    if "vts_sym" in shape:
        shape["vts"] = shape["vts_sym"]


def augment_batch_sym(data, rand=True):
    """
    if rand = False : (test time with sym only) we symmetrize the shape
    if rand = True  : with a probability of 0.5 we symmetrize the shape
    """
    # print(data["shape1"]["gradY"][0,0])
    if not rand or random.randint(0, 1) == 1:
        # print("sym")
        data_augmentation_sym(data["shape1"])
    # print(data["shape1"]["gradY"][0,0], data["shape2"]["gradY"][0,0])
    return data


def auto_WKS(evals, evects, num_E, scaled=True):
    """
    Compute WKS with an automatic choice of scale and energy

    Parameters
    ------------------------
    evals       : (K,) array of  K eigenvalues
    evects      : (N,K) array with K eigenvectors
    landmarks   : (p,) If not None, indices of landmarks to compute.
    num_E       : (int) number values of e to use
    Output
    ------------------------
    WKS or lm_WKS : (N,num_E) or (N,p*num_E)  array where each column is the WKS for a given e
                    and possibly for some landmarks
    """
    abs_ev = sorted(np.abs(evals))

    e_min, e_max = np.log(abs_ev[1]), np.log(abs_ev[-1])
    sigma = 7 * (e_max - e_min) / num_E

    e_min += 2 * sigma
    e_max -= 2 * sigma

    energy_list = np.linspace(e_min, e_max, num_E)

    return WKS(abs_ev, evects, energy_list, sigma, scaled=scaled)


def WKS(evals, evects, energy_list, sigma, scaled=False):
    """
    Returns the Wave Kernel Signature for some energy values.

    Parameters
    ------------------------
    evects      : (N,K) array with the K eigenvectors of the Laplace Beltrami operator
    evals       : (K,) array of the K corresponding eigenvalues
    energy_list : (num_E,) values of e to use
    sigma       : (float) [positive] standard deviation to use
    scaled      : (bool) Whether to scale each energy level

    Output
    ------------------------
    WKS : (N,num_E) array where each column is the WKS for a given e
    """
    assert sigma > 0, f"Sigma should be positive ! Given value : {sigma}"

    evals = np.asarray(evals).flatten()
    indices = np.where(evals > 1e-5)[0].flatten()
    evals = evals[indices]
    evects = evects[:, indices]

    e_list = np.asarray(energy_list)
    coefs = np.exp(-np.square(e_list[:, None] - np.log(np.abs(evals))[None, :]) / (2 * sigma ** 2))  # (num_E,K)

    weighted_evects = evects[None, :, :] * coefs[:, None, :]  # (num_E,N,K)

    natural_WKS = np.einsum('tnk,nk->nt', weighted_evects, evects)  # (N,num_E)

    if scaled:
        inv_scaling = coefs.sum(1)  # (num_E)
        return (1 / inv_scaling)[None, :] * natural_WKS

    else:
        return natural_WKS


def read_geodist(mat):
    # get geodist matrix
    if 'Gamma' in mat:
        G_s = mat['Gamma']
    elif 'G' in mat:
        G_s = mat['G']
    else:
        raise NotImplementedError('no geodist file found or not under name "G" or "Gamma"')

    # get square of mesh area
    if 'SQRarea' in mat:
        SQ_s = mat['SQRarea'][0]
        # print("from mat:", SQ_s)
    else:
        SQ_s = 1

    return G_s, SQ_s


def shape_to_device(dict_shape, device):
    names_to_device = ["xyz", "faces", "mass", "evals", "evecs", "evecs_trans", "gradX", "gradY",
                       "cevecs", "cevals", "spec_grad", "wks", "xyz_unscaled", "faces_unscaled"]
    for k, v in dict_shape.items():
        if "shape" in k:
            for name in names_to_device:
                try:
                    if v[name] is not None:
                        v[name] = v[name].to(device)  # .float()
                except KeyError:
                    pass  # Skip if the key does not exist
            dict_shape[k] = v
        else:
            try:
                dict_shape[k] = v.to(device)
            except AttributeError:
                pass  # Skip if the value does not have a .to(device) method

    return dict_shape


def get_batch_mask(evals1, evals2, gamma=0.5, device="cpu"):
    scaling_factor = max(torch.max(evals1), torch.max(evals2))
    evals1, evals2 = evals1.to(device) / scaling_factor, evals2.to(device) / scaling_factor
    evals_gamma1, evals_gamma2 = evals1.pow(gamma)[:, None, :], evals2.pow(gamma)[:, :, None]

    M_re = evals_gamma2 / (evals_gamma2.square() + 1) - evals_gamma1 / (evals_gamma1.square() + 1)
    M_im = 1 / (evals_gamma2.square() + 1) - 1 / (evals_gamma1.square() + 1)
    return M_re.square() + M_im.square()
