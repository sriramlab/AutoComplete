#%%
import pandas as pd
import numpy as np
import argparse
#%%
class args:
	data_file = 'datasets/phenotypes/data.csv'
	simulated_data_file = 'datasets/phenotypes/data_test.csv'
	imputed_data_file = 'datasets/phenotypes/imputed_data_test.csv'
	num_bootstraps = 100
#%%
parser = argparse.ArgumentParser(description='AutoComplete')
parser.add_argument('data_file', type=str, help='Ground truth data. CSV file where rows are samples and columns correspond to features.')
parser.add_argument('--simulated_data_file', type=str, help='Data with simulated missing values. This is required to check which values were simulated as missing.')
parser.add_argument('--imputed_data_file', type=str, help='Imputed data.')
parser.add_argument('--num_bootstraps', type=int, default=100, help='Number of times to bootstrap the test statistic.')
parser.add_argument('--saveas', type=str, default='results_r2.csv', help='Where to save the evaluation results.')
args = parser.parse_args()
#%%
# the original dataset
original_data = pd.read_csv(args.data_file).set_index('ID')
original_data
#%%
# data with simulated missing values
simulated_data = pd.read_csv(args.simulated_data_file).set_index('ID')
simulated_data
#%%
imputed_data = pd.read_csv(args.imputed_data_file).set_index('ID')
imputed_data
#%%
assert simulated_data.shape == imputed_data.shape
assert simulated_data.index.tolist() == imputed_data.index.tolist()
assert imputed_data.isna().sum().sum() == 0
assert len(imputed_data.index.intersection(original_data.index)) == len(imputed_data)
#%%
ests = []
stds = []
nsize = len(imputed_data)
for pheno in imputed_data.columns:
	missing_frac = simulated_data[pheno].isna().sum() / nsize

	est = np.nan
	stderr = np.nan
	if missing_frac != 0:
		stats = []
		# for n in range(args.num_bootstraps):
		n = 0
		while n < args.num_bootstraps:
			boot_idx = np.random.choice(range(nsize), size=nsize, replace=True)
			boot_obs = original_data.loc[imputed_data.index][pheno].iloc[boot_idx]
			boot_imp = imputed_data[pheno].iloc[boot_idx]

			simulated_missing_inds = simulated_data[pheno].iloc[boot_idx].isna() & ~boot_obs.isna()

			if simulated_missing_inds.sum() == 0:
				continue

			r2 = np.corrcoef(
				boot_obs.values[simulated_missing_inds],
				boot_imp.values[simulated_missing_inds])[0, 1]**2

			n += 1
			stats += [r2]
		est = np.nanmean(stats)
		stderr = np.nanstd(stats)
		print(f'{pheno} ({missing_frac*100:.1f}%): {est:.4f} ({stderr:.4f})')
	else:
		print(f'{pheno} ({missing_frac*100:.1f}%)')

	ests += [est]
	stds += [stderr]
# %%
results = pd.DataFrame(dict(pheno=imputed_data.columns, estimates=ests, stderrs=stds)).set_index('pheno')
results.to_csv(args.saveas)
results
# %%
