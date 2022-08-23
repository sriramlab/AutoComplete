#%%
import pandas as pd
import numpy as np
#%%
droot = 'datasets/phenotypes'
#%%
# the original dataset
original_data = pd.read_csv(f'{droot}/data.csv').set_index('ID')
original_data
#%%
# part of the dataset where missing values were simulated
simulated_missing_data = pd.read_csv(f'{droot}/data_test.csv').set_index('ID')
simulated_missing_data
# %%
# the simulated missing values imputed
imputed_data = pd.read_csv(f'{droot}/imputed_data_test.csv').set_index('ID')
imputed_data
# %%
# gather the r2 score
pheno = 'LifetimeMDD'
original_pheno = original_data.loc[simulated_missing_data.index][pheno]
simulated_pheno = simulated_missing_data[pheno]
score_ids = simulated_pheno.index[simulated_pheno.isna() & ~original_pheno.isna()]
imputed_pheno = imputed_data[pheno]
r2 = np.corrcoef(imputed_pheno.loc[score_ids], original_pheno.loc[score_ids])[0, 1]**2
print('Imutation accuracy: %.4f' % r2, '(simulated)')
# %%
