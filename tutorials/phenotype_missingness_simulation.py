#%%
import pandas as pd
import numpy as np
#%%
simulate_missing = 0.01
#%%
droot = 'datasets/phenotypes'
#%%
db = pd.read_csv(f'{droot}/data.csv', index_col=False).set_index('ID')
db
#%%
vmat = db.values
#%%
obs_level = lambda: (vmat.shape[0]*vmat.shape[1]) - np.sum(np.isnan(vmat))
otarget = obs_level() * (1-simulate_missing)
mcopy = 100
obs_level(), otarget
#%%
while obs_level() > otarget:
    randpos = np.random.randint(0, len(db), size=mcopy)
    maskpos = np.isnan(vmat[randpos, :])
    randpos = np.random.randint(0, len(db), size=mcopy)
    batch = vmat[randpos, :]
    batch[maskpos] = np.nan
    vmat[randpos, :] = batch
    print('\r{} > {}'.format(obs_level(), otarget), end='')
#%%
db[:] = vmat
# %%
data_inds = list(range(db.shape[0]))
np.random.shuffle(data_inds)
data_inds[:5]
# %%
split = len(db) // 3*2
fit_inds, test_inds = data_inds[:split], data_inds[split:]
len(fit_inds), len(test_inds)
# %%
fitdb = db.loc[fit_inds]
testdb = db.loc[test_inds]
# %%
fitdb.to_csv(f'{droot}/data_fit.csv')
# %%
testdb.to_csv(f'{droot}/data_test.csv')

# %%
print('done')