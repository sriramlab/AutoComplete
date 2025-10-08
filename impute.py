#%%
# %load_ext autoreload
# %autoreload 2
#%%
PROJECT_ROOT = '/home/ulzee'
import pandas as pd
import numpy
numpy.random.seed(0)
import torch
torch.manual_seed(0)
import numpy as np
import torch.nn as nn
import os, sys
import torch.optim as optim
import json
from torch.optim.lr_scheduler import StepLR
from time import time
import torch.nn.functional as F
sys.path.append(f'{PROJECT_ROOT}/imp/scripts')
from dataset import MaskMatDataset
# %%
MODEL_NAME = 'deepcoder'
DFILE = sys.argv[1]
DEVICE = sys.argv[2]
LOAD_MODEL_NAME = sys.argv[3]
IMPUTED_SAVE_NAME = sys.argv[4]
INDEX_COL = 'FID'
#%%
parseFloat = lambda raw: float(raw[0] + '.'+raw[1:])

# if PHASE == 'val':
testset = MaskMatDataset(
    'test',
    datafile=DFILE,
    group='mix',
    mask_type={},
    val_split=None,
    boot=None,
    index_col=INDEX_COL)

dsets = dict(
    test=testset,
)
print('Data shape:', testset.mat.shape)

dataloaders = {
    'test': torch.utils.data.DataLoader(
        dsets['test'],
        batch_size=2048,
        shuffle=False)
}
#%%
print('Loading model:', LOAD_MODEL_NAME)
core = torch.load(LOAD_MODEL_NAME, map_location=torch.device('cpu'))
model = core.to(DEVICE)
print(core)
#%%
CONT_BINARY_SPLIT = len(dsets['test'].cont_cats) # ~55
print('cont/binary: %d/%d' % (CONT_BINARY_SPLIT, len(dsets['test'].binary_cats)))

vmat = dataloaders['test'].dataset.vmat
obs_pos = ~np.isnan(vmat)
#%%
cont_crit = nn.MSELoss()
binary_crit = nn.BCEWithLogitsLoss()
model.eval()
dset = dataloaders['test']
preds_ls = []
ep_hist = dict(test=[])
for bi, batch in enumerate(dset):
    pheno, nan_inds, _ = batch
    pheno = pheno.float()
    existing_inds = ~nan_inds
    masked_pheno = pheno.clone().detach()
    score_inds = existing_inds

    score_inds = score_inds.to(DEVICE)
    masked_pheno = masked_pheno.to(DEVICE)
    pheno = pheno.float().to(DEVICE)

    with torch.no_grad():
        yhat = model(masked_pheno)
    sind = CONT_BINARY_SPLIT
    yhat = torch.cat([yhat[:, :sind], torch.sigmoid(yhat[:, sind:])], dim=1)
    preds_ls += [yhat.cpu().numpy()]

    print(f'\r{bi}/{len(dset)}', end='')
print()
#%%
pmat = np.concatenate(preds_ls)
template = testset.mat.copy()
template.isna().sum()
tmat = template.values
tmat[np.isnan(tmat)] = pmat[np.isnan(tmat)]
# %%
template[:] = tmat
sep = ',' if 'csv' in IMPUTED_SAVE_NAME else '\t'
template.to_csv(IMPUTED_SAVE_NAME, sep=sep)
print(IMPUTED_SAVE_NAME)
# %%
