#%%
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from time import time
import json
import argparse

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ac import AutoComplete
from dataset import CopymaskDataset

#%%
class args:
    data_file = 'datasets/phenotypes/data_fit.csv'
    id_name = 'ID'
    lr = 0.1
    batch_size = 2048
    val_split = 0.8
    device = 'cuda:0'
    epochs = 200
    momentum = 0.9
    impute_using_saved = 'datasets/phenotypes/model.pth'
    output = None
    encoding_ratio = 1
    depth = 1
    impute_data_file = None
    copymask_amount = 0.3
#%%
parser = argparse.ArgumentParser(description='AutoComplete')
parser.add_argument('data_file', type=str, help='CSV file where rows are samples and columns correspond to features.')
parser.add_argument('--id_name', type=str, default='ID', help='Column in CSV file which is the identifier for the samples.')
parser.add_argument('--output', nargs='?', type=str, help='The imputed version of the data will be saved as this file. ' +\
    'If not specified the imputed data will be saved as `imputed_{data_file}` in the same folder as the `data_file`.')

parser.add_argument('--copymask_amount', nargs='?', type=float, default=0.3, help='Probability that a sample will be copy-masked. A range from 10%%~50%% is recommemded.')
parser.add_argument('--batch_size', nargs='?', type=int, default=2048, help='Batch size for fitting the model.')
parser.add_argument('--epochs', nargs='?', type=int, default=200, help='Number of epochs.')
parser.add_argument('--lr', nargs='?', type=float, default=0.1, help='Learning rate for fitting the model. A starting LR between 2~0.1 is recommended.')
parser.add_argument('--momentum', nargs='?', type=float, default=0.9, help='Momentum for SGD optimizer (default is recommended).')
parser.add_argument('--val_split', nargs='?', type=float, default=0.8, help='Amount of data to use as a validation split. The validation split is monitored for convergeance.')
parser.add_argument('--device', nargs='?', type=str, default='cpu:0', help='Device available for torch (use cpu:0 if no GPU available).')
parser.add_argument('--encoding_ratio', nargs='?', type=float, default=1,
    help='Size of the centermost encoding dimension as a ratio of # of input features; ' + \
    'eg. `0.5` would force an encoding by half.')
parser.add_argument('--depth', nargs='?', type=int, default=1, help='# of fully connected layers between input and centermost deep layer; ' + \
    'the # of layers beteen the centermost layer and the output layer will be defined equally.')

parser.add_argument('--impute_using_saved', nargs='?', type=str, help='Load trained weights from a saved .pth file to ' + \
    'impute the data without going through model training.')
parser.add_argument('--impute_data_file', nargs='?', type=str, help='CSV file where rows are samples and columns correspond to features.')

args = parser.parse_args()

#%%
fparts = args.data_file.split('/')
save_folder = '/'.join(fparts[:-1]) + '/'
save_model_name = save_folder + 'model.pth'

if args.output:
    save_table_name = args.output
else:
    save_table_name = save_folder + f'imputed_{fparts[-1]}'
if not args.impute_using_saved: print('Saving model to:', save_model_name)
print('Saving imputed table to:', save_table_name)
#%%
tab = pd.read_csv(args.data_file).set_index(args.id_name)
print(f'Dataset size:', tab.shape[0])
#%%
# detect binary phenotypes
ncats = tab.nunique()
binary_features = tab.columns[ncats == 2]
contin_features = tab.columns[~(ncats == 2)]
feature_ord = list(contin_features) + list(binary_features)
print(f'Features loaded: contin={len(contin_features)}, binary={len(binary_features)}')
CONT_BINARY_SPLIT = len(contin_features)
# %%
# keep a validation set
val_ind = int(tab.shape[0]*args.val_split)
splits = ['train', 'val', 'final']
dsets = dict(
    train=tab[feature_ord].iloc[:val_ind, :],
    val=tab[feature_ord].iloc[val_ind:, :],
    final=tab[feature_ord],
)
# %%
train_stats = dict(
    mean=dsets['train'].mean().values,
    std=dsets['train'].std().values,
)
#%%
normd_dsets = {
    split: (dsets[split].values - train_stats['mean'])/train_stats['std'] \
        for split in splits }
# %%
dataloaders = {
    split: torch.utils.data.DataLoader(
        CopymaskDataset(mat, split, copymask_amount=args.copymask_amount),
        batch_size=args.batch_size,
        shuffle=split=='train', num_workers=0) \
            for split, mat in normd_dsets.items() }
#%%
feature_dim = dsets['train'].shape[1]
core = AutoComplete(
        indim=feature_dim,
        width=1/args.encoding_ratio,
        n_depth=args.depth,
    )
model = core.to(args.device)
print(core)
#%%
if not args.impute_using_saved:
    cont_crit = nn.MSELoss()
    binary_crit = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, threshold=1e-10, patience=5)

    def get_lr():
        for param_group in optimizer.param_groups:
            return param_group['lr']
    print('starting lr', get_lr())

    hist = dict(
        train=list(),
        val=list(),
        lr=list(),
    )
    best_test_loss = None
    for ep in range(args.epochs):
        for phase in (['train', 'val']):
            model.train() if phase == 'train' else model.eval()

            t_ep = time()
            ep_hist = dict(total=list(), binary=list())
            dset = dataloaders[phase]

            for bi, batch in enumerate(dset):
                datarow, nan_inds, train_inds = batch
                datarow = datarow.float()
                masked_data = datarow.clone().detach()
                masked_data[train_inds] = 0

                existing_inds = ~nan_inds
                score_inds = existing_inds
                score_inds = score_inds.to(args.device)
                masked_data = masked_data.to(args.device)
                datarow = datarow.to(args.device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    yhat = model(masked_data)
                    sind = CONT_BINARY_SPLIT

                    l_cont, l_binary = torch.zeros(1), torch.zeros(1)
                    if len(contin_features) != 0:
                        l_cont = cont_crit((yhat*score_inds)[:,:sind], (datarow*score_inds)[:, :sind])
                    if len(binary_features) != 0:
                        binarized = (((datarow)*score_inds)[:, sind:] > 0.5).float()
                        l_binary = binary_crit(
                            (yhat*score_inds)[:, sind:],
                            binarized)
                    loss = l_cont + l_binary

                    ep_hist['total'] += [loss.item()]
                    ep_hist['binary'] += [l_binary.item()]
                    if np.isnan(loss.item()):
                        print(yhat.isnan().sum())
                        print(l_cont.item())
                        print(l_binary.item())

                if phase == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    optimizer.step()

                print(f'\r[E{ep+1} {phase} {bi+1}/{len(dset)}] - L%.4f (%.4f %.4f) %.1fs LR:{get_lr()}   ' % (
                    np.mean(ep_hist['total']), l_cont.item(), l_binary.item(),(time() - t_ep)
                ), end='')

            print()

            hist[phase] += [ep_hist['total']]
            hist['lr'] += [get_lr()]

        scheduler.step(np.mean(hist['val'][-1]))

        with open(save_model_name + '.json', 'w') as fl:
            json.dump(hist, fl)

        # save if loss improved
        current_loss = hist['val'][-1]
        if best_test_loss == None or best_test_loss > current_loss:
            best_test_loss = current_loss
            torch.save(core, save_model_name)
            print('saved')

        # if starting to overfit, stop early
        if ep > 50:
            loss_1 = np.mean(hist['val'][-1])
            loss_50 = np.mean(hist['val'][-50])
            if loss_1 > loss_50*2:

                print('Early stopping', loss_1, '>', loss_50, '(x2)')
                break

        if np.isnan(np.mean(hist['val'][-1])):
            print('Training NaN, exiting...')
            break
#%%
model.eval()
dset = dataloaders['final']

if args.impute_using_saved:
    model = torch.load(args.impute_using_saved)
    model = model.to(args.device)

if args.impute_data_file:
    imptab = pd.read_csv(args.impute_data_file).set_index(args.id_name)[feature_ord]
    print(f'(impute) Dataset size:', imptab.shape[0])
    mat_imptab = (imptab.values - train_stats['mean'])/train_stats['std']
    dset = torch.utils.data.DataLoader(
        CopymaskDataset(mat_imptab, 'final'),
        batch_size=args.batch_size,
        shuffle=False, num_workers=0)
    if not args.output:
        impute_fparts = args.impute_data_file.split('/')
        save_table_name = save_folder + f'imputed_{impute_fparts[-1]}'

preds_ls = []
for bi, batch in enumerate(dset):
    datarow, _, masked_inds = batch
    datarow = datarow.float().to(args.device)

    with torch.no_grad():
        yhat = model(datarow)
    sind = CONT_BINARY_SPLIT
    yhat = torch.cat([yhat[:, :sind], torch.sigmoid(yhat[:, sind:])], dim=1)

    preds_ls += [yhat.cpu().numpy()]
    print(f'\r{bi}/{len(dset)}', end='')
print()
#%%
pmat = np.concatenate(preds_ls)
pmat[:, :CONT_BINARY_SPLIT] = (pmat[:,:CONT_BINARY_SPLIT] * train_stats['std'][:CONT_BINARY_SPLIT]) \
    + train_stats['mean'][:CONT_BINARY_SPLIT]
#%%
template = tab.copy() if not args.impute_data_file else imptab.copy()
tmat = template.values
tmat[np.isnan(tmat)] = pmat[np.isnan(tmat)]
template[:] = tmat
template
# %%
template.to_csv(save_table_name)
#%%
print('done')
#%%
