
import numpy
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

class MaskMatDataset(Dataset):
    def __init__(self, split,
                 datafile,
                 group='mix',
                #  obs_ratio=0.9, # mat will be 10% censored
                 flip_majority=False,
                 pos_majority_cats=None,
                 val_split=None, # 'train' or 'val' further dividing the original train set
                 val_split_ratio=0.80,
                 mask_type=dict(),
                 center_binary_phenos=True,
                 mask_distrib='0',
                 boot=None,
                 index_col='FID',
            ):

        # self.obs_ratio = obs_ratio
        self.split = split
        self.center_binary_phenos = center_binary_phenos
        self.mask_type = mask_type
        self.mask_distrib = mask_distrib

        if self.mask_distrib == 'unif':
            print('Loading uniform mask')

        print('Random Masking:', 'ON' if len(mask_type) else 'OFF')
        print('Mask Type     :', mask_type)

        sep = ',' if 'csv' in datafile else '\t'
        mat = pd.read_csv(datafile, sep=sep, index_col=index_col)

        data_dir = os.path.dirname(datafile)
        cols_file_path = os.path.join(data_dir, 'metadata.csv')
        catsmat = pd.read_csv(cols_file_path)
        self.cont_cats = catsmat[catsmat['isbinary'] == False]['cats'].values.tolist()
        self.binary_cats = catsmat[catsmat['isbinary'] == True]['cats'].values.tolist()

        if val_split is not None:
            val_split_ind = int(mat.shape[0] * val_split_ratio)
            # assert split == 'train'
            # we could get a subset of the test too (for quick metrics)
            if val_split == 'train':
                mat = mat.iloc[:val_split_ind,:]
            elif val_split == 'val':
                mat = mat.iloc[val_split_ind:,:]
            else:
                assert False

        self.mat = mat
        self.vmat = mat.values

        print('N             :', self.mat.shape[0])

    def __getitem__(self, idx):
        phenoRow = self.vmat[idx,].copy() # np.array(self.mat.iloc[idx,].values.tolist())
        if self.center_binary_phenos:
            phenoRow[-len(self.binary_cats):] -= 0.5
        missing_inds = np.isnan(phenoRow)
        # existing_inds = ~missing_inds

        masked_inds = np.zeros(len(phenoRow)).astype(bool)

        if 'copy_mask' in self.mask_type:
            assert self.split != 'test'
            # get an empty starting mask (no copy mask)
            masked_inds = np.zeros(len(phenoRow), dtype=bool)

            # grab one random from population
            if np.random.rand() < self.mask_type['copy_mask']:
                rnd_ind = np.random.randint(self.mat.shape[0])
                masked_inds |= np.isnan(self.vmat[rnd_ind,]) # np.isnan(self.mat.iloc[rnd_ind,].values)

        # existing nans set to zero
        phenoRow[missing_inds] = 0

        return phenoRow, missing_inds, masked_inds #, eval_inds

    def __len__(self):
        return self.mat.shape[0]