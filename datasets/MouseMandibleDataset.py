import os.path as osp
from os import listdir as osls
from os.path import isfile, join
from itertools import permutations
from random import shuffle
from tqdm import tqdm

import numpy as np

import igl

import shutil

import torch
from torch_geometric.data import extract_zip, Dataset


import potpourri3d as pp3d
from diffusion_net import compute_operators
import Tools.mesh as qm
from Tools.utils import op_cpl
from utils import auto_WKS, farthest_point_sample, square_distance


class MouseMandibleDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, wks_eig=None, k_eig=None, n_fmap=None, n_cfmap=None):
        self.k_eig = k_eig
        self.wks_eig = wks_eig
        self.n_cfmap = n_cfmap
        self.n_fmap = n_fmap
        super(MouseMandibleDataset, self).__init__(root, transform, pre_transform)

        corresp = list(permutations(self.processed_file_names, 2))
        shuffle(corresp)
        self.source_filenames, self.target_filenames = zip(*corresp)

    @property
    def raw_file_names(self):
        return 'mandible.zip'

    @property
    def processed_file_names(self):
        extract_zip(self.raw_paths[0], self.raw_dir, log=False)

        off_path = osp.join(self.raw_dir, 'mandible_ply')
        file_names = [f[:-3] + 'pt' for f in osls(off_path) if isfile(osp.join(off_path, f)) and f.endswith('.ply')]

        # shutil.rmtree(osp.join(self.raw_dir, 'cuboid_off'))

        return file_names

    def process(self):
        extract_zip(self.raw_paths[0], self.raw_dir, log=False)

        ply_path = osp.join(self.raw_dir, 'mandible_ply')

        locations = {f[:-4]: join(ply_path, f) for f in osls(ply_path) if isfile(join(ply_path, f))}
        flawed_files = []
        for key, value in tqdm(locations.items(), ascii=True):
            print(key)
            shape = {}
            # verts, faces = read_off(value)

            verts_unscaled, faces_unscaled = pp3d.read_mesh(value)
            verts, faces = center_and_scale(verts_unscaled, faces_unscaled, scale=True)
            # Scale and center here if necessary

            shape['name'] = key
            shape['xyz'] = torch.tensor(np.ascontiguousarray(verts)).float()
            shape['faces'] = torch.tensor(np.ascontiguousarray(faces))

            shape['xyz_unscaled'] = torch.tensor(np.ascontiguousarray(verts_unscaled)).float()
            shape['faces_unscaled'] = torch.tensor(np.ascontiguousarray(faces_unscaled))

            try:
                (shape['frames'],
                 shape['mass'],
                 shape['L'],
                 shape['evals'],
                 shape['evecs'],
                 shape['gradX'],
                 shape['gradY']) = compute_operators(shape['xyz'], shape['faces'],
                                                     k_eig=self.k_eig, normals=None)
            except ValueError:
                flawed_files.append(key)
                print(key, 'is not manifold, skipping.')

                continue

            shape['wks'] = auto_WKS(shape['evals'], shape['evecs'], num_E=self.wks_eig).float()

            shape['evecs_trans'] = shape['evecs'].t()[:self.n_fmap] @ torch.diag(shape['mass'])

            try:
                # load mesh and compute complex laplacian and gradient operators
                mesh_for_Q = qm.mesh(value, normalized=True, spectral=0, complex_spectral=self.n_cfmap,
                                     spectral_folder=self.root)
                shape['mesh'] = mesh_for_Q
            except ValueError:
                flawed_files.append(key)
                print(key, 'is not manifold, skipping.')

                continue
            if self.n_cfmap > 0:
                mesh_for_Q.grad_vert_op()
                mesh_for_Q.grad_vc = op_cpl(mesh_for_Q.gradv.T).T

                shape['cevecs'] = torch.tensor(mesh_for_Q.ceig)
                shape['cevals'] = torch.tensor(mesh_for_Q.cvals)
                shape['spec_grad'] = torch.tensor(np.linalg.pinv(mesh_for_Q.ceig) @ mesh_for_Q.grad_vc)

            fn = key + '.pt'
            torch.save(shape, osp.join(self.processed_dir, fn))

        print(flawed_files)
        shutil.rmtree(osp.join(self.raw_dir, 'mandible_ply'))
        print('Preprocessed')

    def len(self):
        return len(self.source_filenames)

    def get(self, idx):
        data_src = torch.load(osp.join(self.processed_dir, self.source_filenames[idx]))
        data_tar = torch.load(osp.join(self.processed_dir, self.target_filenames[idx]))

        del data_src['mesh']
        del data_tar['mesh']

        # Compute fmap
        evec_1, evec_2 = data_src["evecs"][:, :self.n_fmap], data_tar["evecs"][:, :self.n_fmap]

        C12_gt = torch.zeros_like(torch.pinverse(evec_2[:1000]) @ evec_1[:1000])
        C21_gt = torch.zeros_like(torch.pinverse(evec_1[:1000]) @ evec_2[:1000])

        return {"shape1": data_src, "shape2": data_tar, 'C12_gt': C12_gt, 'C21_gt': C21_gt}


def center_and_scale(v, f, scale=False):
    v -= np.mean(v, axis=0)
    if not scale:
        return v, f
    else:
        area = np.sum(igl.doublearea(v, f)) / 2
        print('area was', area)
        v /= np.sqrt(area)

        return v, f
