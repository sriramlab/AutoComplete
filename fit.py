#%%
import pandas as pd
from time import time
import json
import argparse
import sys
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
parser.add_argument('--output', type=str, help='The imputed version of the data will be saved as this file. ' +\
    'If not specified the imputed data will be saved as `imputed_{data_file}` in the same folder as the `data_file`.')

parser.add_argument('--save_model_path', type=str, help='A location to save the imputation model weights. Will default to file_name.pth if not set.', default=None)

parser.add_argument('--copymask_amount', type=float, default=0.3, help='Probability that a sample will be copy-masked. A range from 10%%~50%% is recommemded.')
parser.add_argument('--batch_size', type=int, default=2048, help='Batch size for fitting the model.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs.')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for fitting the model. A starting LR between 2~0.1 is recommended.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer (default is recommended).')
parser.add_argument('--val_split', type=float, default=0.8, help='Amount of data to use as a validation split. The validation split is monitored for convergeance.')
parser.add_argument('--device', type=str, default='cpu:0', help='Device available for torch (use cpu:0 if no GPU available).')
parser.add_argument('--encoding_ratio', type=float, default=1,
    help='Size of the centermost encoding dimension as a ratio of # of input features; ' + \
    'eg. `0.5` would force an encoding by half.')
parser.add_argument('--depth', type=int, default=1, help='# of fully connected layers between input and centermost deep layer; ' + \
    'the # of layers beteen the centermost layer and the output layer will be defined equally.')

parser.add_argument('--save_imputed', help='Will save an imputed version of the matrix immediately after fitting it.', action='store_true', default=False)
parser.add_argument('--impute_using_saved', type=str, help='Load trained weights from a saved .pth file to ' + \
    'impute the data without going through model training.')
parser.add_argument('--impute_data_file', type=str, help='CSV file where rows are samples and columns correspond to features.')
parser.add_argument('--seed', type=int, help='A specific seed to use. Can be used to instantiate multiple imputations.', default=-1)
parser.add_argument('--bootstrap', help='Flag to specify whether the dataset should be bootstrapped for the purpose of fitting.', default=False, action='store_true')
parser.add_argument('--multiple', type=int, help='If set, this script will save a list of commands which can be run (either in sequence or in parallel) to save mulitple imputations', default=-1)
parser.add_argument('--quality', help='Applies to the fitting procedure. If set, this script will compute a variance ratio metric and a r^2 metric for each feature to roughly inform the quality of imputation', default=False, action='store_true')
parser.add_argument('--simulate_missing', help='Specifies the %% of original data to be simulated as missing for r^2 computation.', default=0.01, type=float)

args = parser.parse_args()
#%%
if args.multiple != -1:
    print('Saving commands for multiple imputations based on the current configs.')
    configs = sys.argv[1:]
    mi = configs.index('--multiple')
    configs.pop(mi)
    configs.pop(mi)
    with open('multiple_imputation.sh', 'w') as fl:
        fl.write('\n'.join([
            'python fit.py ' + ' '.join(configs) + f' --seed {m} --bootstrap --save_imputed'
            for m in range(args.multiple)]))
    exit()
#%%
fparts = args.data_file.split('/')
save_folder = '/'.join(fparts[:-1]) + '/'
filename = args.data_file.split('/')[-1].replace('.csv', '')
save_model_path = save_folder + filename

if args.output:
    save_table_name = args.output
else:
    save_table_name = save_folder + f'imputed_{filename}'

if args.seed != -1:
    save_table_name += f'_seed{args.seed}'
    save_model_path += f'_seed{args.seed}'
if args.bootstrap:
    save_table_name += f'_bootstrap'
    save_model_path += f'_bootstrap'

save_model_path += '.pth'
save_table_name += '.csv'

if args.save_model_path is not None:
    save_model_path = args.save_model_path

if not args.impute_using_saved:
    print('Saving model to:', save_model_path)
if args.impute_using_saved or args.save_imputed:
    print('Saving imputed table to:', save_table_name)
#%%
import torch
import random
import numpy as np

if args.seed != -1:
    print(f'Using seed: {args.seed}')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ac import AutoComplete
from dataset import CopymaskDataset
#%%
tab = pd.read_csv(args.data_file).set_index(args.id_name)
print(f'Dataset size:', tab.shape[0])
#%%
if args.bootstrap:
    print('Bootstrap mode')
    ix = list(range(len(tab)))
    ix = np.random.choice(ix, size=len(tab), replace=True)
    tab = tab.iloc[ix]
    print('First few ids are:')
    for i in tab.index[:5]:
        print(' ', i)

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
#%%
if not args.impute_using_saved:
    cont_crit = nn.MSELoss()
    binary_crit = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, threshold=1e-10, patience=20)

    def get_lr():
        for param_group in optimizer.param_groups:
            return param_group['lr']

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

        with open(save_model_path + '.json', 'w') as fl:
            json.dump(hist, fl)

        # save if loss improved
        current_loss = hist['val'][-1]
        if best_test_loss == None or best_test_loss > current_loss:
            best_test_loss = current_loss
            torch.save(core, save_model_path)
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

if args.impute_using_saved:
    print(f'Loading specified weights: {args.impute_using_saved}')
    model = torch.load(args.impute_using_saved)

if (args.save_imputed or args.quality) and not args.impute_using_saved:
    print('Loading last best checkpoint')
    model = torch.load(save_model_path)

if args.impute_data_file or args.save_imputed or args.quality:
    model = model.to(args.device)
    model.eval()

    impute_mat = args.impute_data_file if args.impute_data_file else args.data_file
    imptab = pd.read_csv(impute_mat).set_index(args.id_name)[feature_ord]
    print(f'(impute) Dataset size:', imptab.shape[0])

    mat_imptab = (imptab.values - train_stats['mean'])/train_stats['std']
    dset = torch.utils.data.DataLoader(
        CopymaskDataset(mat_imptab, 'final'),
        batch_size=args.batch_size,
        shuffle=False, num_workers=0)

    preds_ls = []

    sim_missing = imptab.values.copy()
    print('Starting # observed values:', (~np.isnan(sim_missing)).sum())
    target_missing_sim = (~np.isnan(sim_missing)).sum() * (1 - args.simulate_missing)
    while target_missing_sim < (~np.isnan(sim_missing)).sum():
        samplesA = np.random.choice(range(len(sim_missing)), size=len(imptab)//100)
        samplesB = np.random.choice(range(len(sim_missing)), size=len(imptab)//100)
        # print(np.isnan(sim_missing[samplesB]).sum())
        patch = sim_missing[samplesA]
        patch[np.isnan(sim_missing[samplesB])] = np.nan
        sim_missing[samplesA] = patch
        print(f'\r Simulating missing values: {target_missing_sim} < { (~np.isnan(sim_missing)).sum()}', end='')
    sim_missing = np.isnan(sim_missing)
    print()

    for bi, batch in enumerate(dset):
        datarow, _, masked_inds = batch
        datarow = datarow.float().to(args.device)

        if args.quality:
            sim_mask = sim_missing[bi*args.batch_size:(bi+1)*args.batch_size]
            datarow[sim_mask] = 0

        with torch.no_grad():
            yhat = model(datarow)
        sind = CONT_BINARY_SPLIT
        yhat = torch.cat([yhat[:, :sind], torch.sigmoid(yhat[:, sind:])], dim=1)

        preds_ls += [yhat.cpu().numpy()]
        print(f'\rImputing: {bi}/{len(dset)}', end='')

    pmat = np.concatenate(preds_ls)
    print()

    if args.quality:
        print('=================================================')
        print('Imputation Quality:')
        qdf = dict(feature=[], info=[], r2=[], quality=[])
        for pi, feature in enumerate(imptab.columns):
            mfrac = imptab[feature].isna().sum() / len(imptab)
            dxstr = '(no missing values)'
            var_info = None
            simr2 = 0
            flag = 'NOM'
            if mfrac > 0:

                var_imp = pmat[:, pi][imptab[feature].isna()].var()
                var_obs = imptab[feature][~imptab[feature].isna()].values.var()
                var_info = var_imp / var_obs

                vsim = sim_missing[:, pi] & ~imptab[feature].isna()
                nsim = vsim.sum()
                simr2 = np.corrcoef(pmat[:, pi][vsim], imptab[feature].values[vsim])[0, 1]**2

                Nobs = np.sum(~imptab[feature].isna())
                Neff = int(simr2 * np.sum(imptab[feature].isna()) + Nobs)
                eff_fold = Neff / Nobs



                if mfrac < 0.1:
                    flag = 'LOM'
                else:
                    if var_info >=0.2 and simr2 < 0.2:
                        flag = 'LOR'
                    elif var_info < 0.2 and simr2 >= 0.2:
                        flag = 'LOV'
                    elif var_info >= 0.2 and simr2 >= 0.2:
                        flag = 'QOK'
                    else:
                        flag = 'LOQ'

                dxstr = f'var_info={var_info:.2f} r2={simr2:.2f} effective=x{eff_fold:.1f}'
            qdf['feature'] += [feature]
            qdf['info'] += [var_info]
            qdf['r2'] += [simr2]
            qdf['quality'] += [flag]
            print(f'{flag} missing={mfrac*100:.1f}% {dxstr}', feature)
        print('=================================================')
        qdf = pd.DataFrame(qdf)
        qdf.to_csv(save_model_path.replace('.pth', '_quality.csv'), index=False)

    if args.impute_data_file or args.save_imputed:
        pmat[:, :CONT_BINARY_SPLIT] = (pmat[:,:CONT_BINARY_SPLIT] * train_stats['std'][:CONT_BINARY_SPLIT]) \
            + train_stats['mean'][:CONT_BINARY_SPLIT]
        template = imptab.copy()
        tmat = template.values
        tmat[np.isnan(tmat)] = pmat[np.isnan(tmat)]
        template[:] = tmat
        template

        template.to_csv(save_table_name)

print('done')
