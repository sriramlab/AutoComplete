#%%

import pandas as pd
import numpy
import torch
numpy.random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(4)
import numpy as np
import torch.nn as nn
import os, sys
import torch.optim as optim
import json
from torch.optim.lr_scheduler import StepLR
from time import time
import torch.nn.functional as F
from dataset import MaskMatDataset
# %%
MODEL_NAME = 'deepcoder'
#%%
# TAGS = 'RANDMASK05_COPYMASK05_WIDTH10_DEPTH1_MULT0_BATCH2048_RELU_SGDLR01M09_VAL'
#%%
DATA_FILE = sys.argv[1]
MODEL_NAME = sys.argv[2]
DEVICE = sys.argv[3]
COPYMASK_OBS = float(sys.argv[4])
SAVE_MODEL_NAME = sys.argv[5]
BATCH_SIZE=2048
SEED = 0
BOOT = None
EPOCHS = 200
COPYMASK = True
#%%

if SEED is not None:
    numpy.random.seed(SEED)
    torch.manual_seed(SEED)

parseFloat = lambda raw: float(raw[0] + '.'+raw[1:])

sgd_momentum = 0.9
LR = 0.1


print('Config:')
print(MODEL_NAME)
print('EP', EPOCHS)
print('LR', LR)
print('MOM', sgd_momentum)
# print('TAGS', TAGS)
print('BATCH', BATCH_SIZE)
# print('RANDOM MASK', RANDOM_MASK, RANDOM_MASK_RATIO)
print('COPYMASK', COPYMASK, COPYMASK_OBS)
print('DEVICE', DEVICE)
print('SEED', SEED)
print('BOOT', BOOT)

#%%
def template(split, valsplit, doboot):
    mask_type = {} #'copymask'
    if split != 'test' and valsplit != 'val':
        if COPYMASK:
            mask_type['copy_mask'] = COPYMASK_OBS
    else:
        mask_type['fixed_copy_mask'] = True
    return MaskMatDataset(
        split,
        datafile=DATA_FILE,
        group='mix',
        mask_type=mask_type,
        val_split=valsplit,
        val_split_ratio=0.8,
        boot=doboot)

dsets = dict(
    train=template('train', 'train', doboot=BOOT),
    test=template('train', 'val', doboot=None), # this will be our "test" for training
)

dataloaders = {x: torch.utils.data.DataLoader(
    dsets[x],
    batch_size=BATCH_SIZE,
    shuffle=x=='train', num_workers=0) for x in ['train', 'test']}
#%%
print('# Cats:', len(dsets['train'].cont_cats), len(dsets['train'].binary_cats))
#%%
pccats = [c for c in dsets['train'].cont_cats if 'PC' in c]
NPCS = len(pccats)
#%%
from nets import DeepCoder
lookup = dict(
    deepcoder=DeepCoder,
)
print('Saving model to:', SAVE_MODEL_NAME)
#%%
from torch.optim.lr_scheduler import ReduceLROnPlateau
pheno_dim = len(dsets['train'][0][0])
core = lookup[MODEL_NAME](indim=pheno_dim)
model = core.to(DEVICE)
print(core)
print('Weight hash:', next(core.net[0].parameters())[0, :5].detach().cpu().numpy())

cont_crit = nn.MSELoss()
binary_crit = nn.BCEWithLogitsLoss()

cont_crit_flat = nn.MSELoss(reduction='sum')
binary_crit_flat = nn.BCEWithLogitsLoss(reduction='sum')
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

# learning rate schedulers
scheduler = ReduceLROnPlateau(optimizer, factor=0.5, threshold=1e-8, patience=5)

def get_lr():
    for param_group in optimizer.param_groups:
        return param_group['lr']
print('starting lr', get_lr())
#%%
CONT_BINARY_SPLIT = len(dsets['train'].cont_cats) # ~55
print('cont/binary: %d/%d' % (CONT_BINARY_SPLIT, len(dsets['train'].binary_cats)))
#%%
hist = dict(
    train=list(), test=list(), train_imp=list(), test_imp=list(), lr=list(),
)
train_start_time = time()
best_test_loss = None
for ep in range(EPOCHS):
    for phase in (['train', 'test']):
        model.train() if phase == 'train' else model.eval()

        t_ep = time()
        ep_hist = { k: list() for k in hist.keys() }
        dset = dataloaders[phase]
        for bi, batch in enumerate(dset):
            pheno, nan_inds, masked_inds = batch
            pheno = pheno.float()
            existing_inds = ~nan_inds
            masked_pheno = pheno.clone().detach()
            masked_pheno[masked_inds] = 0
            score_inds = existing_inds

            # eval_inds = eval_inds.to(DEVICE)
            existing_inds = existing_inds.to(DEVICE)
            masked_inds = masked_inds.to(DEVICE)
            score_inds = score_inds.to(DEVICE)
            masked_pheno = masked_pheno.to(DEVICE)
            pheno = pheno.float().to(DEVICE)

            # pheno - original pheno matrix (nans are zerod)
            # nan_inds - inds that were nan in original pheno
            # existing_inds - opposite of nan_inds
            # score_inds - randomly picked inds that were NOT NAN in the original pheno

            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                yhat = model(masked_pheno)
                sind = CONT_BINARY_SPLIT

                l_cont = cont_crit_flat((yhat*existing_inds)[:,:sind], (pheno*existing_inds)[:, :sind])
                l_binary = binary_crit_flat(
                    (yhat*existing_inds)[:, sind:],
                    ((pheno+0.5)*existing_inds)[:, sind:])
                loss = (l_cont/ existing_inds[:, :sind].sum() + l_binary/ existing_inds[:, sind:].sum())
                # loss = l_cont + l_binary

                if phase == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    optimizer.step()

                ep_hist[f'{phase}'] += [loss.item()]

            print(f'\r[E{ep+1} {phase} {bi+1}/{len(dset)}] - L%.4f %.1fs LR:{get_lr()}   ' % (
                np.mean(ep_hist[phase]),
                (time() - t_ep)
            ), end='')
        print()

        hist[phase] += [ep_hist[phase]]
        hist[f'{phase}_imp'] += [ep_hist[f'{phase}_imp']]
        hist['lr'] += [get_lr()]

    scheduler.step(np.mean(hist['test'][-1]))
    with open(SAVE_MODEL_NAME + '.json', 'w') as fl:
        json.dump(hist, fl)

    if MODEL_NAME == 'stacked':
        if ep > 1:
            L0 = np.mean(hist['test'][-1])
            L1 = np.mean(hist['test'][-2])
            Ldiff = L1 - L0
            if abs(Ldiff) < 10e-6: # TODO: adjust
                # reset LR
                for param_group in optimizer.param_groups:
                    param_group['lr'] = LR
                # progress stack
                all_trained = model.progress_stack()
                if all_trained:
                    break
        continue


    current_loss = hist['test'][-1]
    if best_test_loss == None or best_test_loss > current_loss:
        # save if loss improved
        best_test_loss = current_loss
        # if BENCH_POP is None:
        print('save!')
        torch.save(core, SAVE_MODEL_NAME)
    # if ~converged, end early

    # if starting to overfit, stop early
    if ep > 25 and ep % 5 == 0:
        Lrecent = np.mean(hist['test'][-1])
        Lmin = np.min([np.mean(h) for h in hist['test']])
        if Lrecent > Lmin*2:

            print('Early stopping', Lrecent, '>', Lmin, '(x2)')
            break

    if np.isnan(np.mean(hist['train'][-1])):
        print('Training NaN')
        break
    # break
#%%
print('done')
# %%
