# AutoComplete

AutoComplete is a deep-learning based imputation method capable of imputing continuous and binary values simultaneously.

## Getting Started

AutoComplete can run with most Python 3 versions, and defines neural nets using [pytorch](https://pytorch.org).
The dependencies can be found in `requirements.txt` and installed using:


### With PIP
```
git clone https://github.com/sriramlab/AutoComplete
cd AutoComplete
pip install -r requirements.txt
```

### With Conda
```
git clone https://github.com/sriramlab/AutoComplete
cd AutoComplete

conda create -n ac python=3.7
conda activate ac
pip install -r requirements.txt
```

## Imputation Demo

An example procedure for fitting, imputing, and scoring a simulated phenotype dataset with missing values can be found in `tutorials/`. To run this pipeline, do:

```bash
./tutorials/phenotype_demo.sh
```

The script runs the following commands. First, artificial missing values are introduced to `datasets/phenotypes/data.csv` such that they can be withheld then scored after imputation.
```bash
python tutorials/phenotype_missingness_simulation.py
```

Then, the method is fit to a training split of the data saved to `datasets/phenotypes/data_fit.csv`.
```bash
python fit.py datasets/phenotypes/data_fit.csv \
    --id_name ID \
    --copymask_amount 0.5 \
    --batch_size 2048 \
    --epochs 100 \
    --lr 0.1 \
    --device cpu:0
```

The fitted model is used to impute the testing split of the data which is `datasets/phenotypes/data_test.csv`.
```bash
python fit.py datasets/phenotypes/data_fit.csv \
    --id_name ID \
    --impute_using_saved datasets/phenotypes/model.pth \
    --impute_data_file datasets/phenotypes/data_test.csv \
    --device cpu:0
```

Finally, the simulated missing values are scored against their originally observed values. The Pearson's r^2 correlation is used and 100 bootstrap replicates are used to obtain the point estimate of the accuracy and its standard error.
```bash
python bootstrap_r2_statistic.py datasets/phenotypes/data.csv \
    --simulated_data_file datasets/phenotypes/data_test.csv \
    --imputed_data_file datasets/phenotypes/imputed_data_test.csv \
    --num_bootstraps 100 \
    --saveas result_r2_phenotype_demo.csv
```

## General Usage

To show all available options:

```
python fit.py -h
```

AutoComplete can run easily for most datasets in CSV format such as:

```
python fit.py datasets/random/data.csv --id_name ID --batch_size 512 --epochs 50 --lr 0.1 --device cpu:0
```

The first row of the data file is expected to be a header with names for each column, where the `id_name` option specifies which column of the dataset to use as an identifier for each sample. Missing values should be left as blank entries in the CSV file without any NA or NaN tokens. The expected formatting is therefore eg. `1,,2` where there is a missing value implied between 1 and 2.

Continuous or binary-valued features will be automatically detected based on the number of unique values which are present per feature
(having only 2 values will be interpreted as a binary feature).

A version of the dataset with imputed values will be saved with a prefix in the same folder such as `imputed_{data_file}`. Alternatively the output file path can be manually specified using the `--output` option.

In the case of imputing another data file with a model that was already trained, the path to the saved model can be specified with the `--impute_using_weights` option and imputation will be performed without any training.

To use any GPU available, the `--device cpu:0` flag can be changed to `--device gpu:0`.


Additional configs for the neural net architecture are possible such as the encoding ratio which determines the size of the centermost dimension as a ratio of the # of input features and the depth of the neural net. The size of the intermediary layers are determined automatically as a linear arrangement from the input layer size to the centermost layer size.
```
# If there are 10 features in the data, the specification of layers will be:
# [10, floor(7.5), 5, floor(7.5), 10]

--encoding_ratio 0.5 --depth 2
```

## Multiple Imputations

Multiple imputations allows one way to account for uncertainty in the imputation process for downstream analysis. AutoComplete allows multiple imputations by bootstrapping a given dataset and fitting it multiple times with differently seeded intializations. The `--multiple` argument for `fit.py` allows the script to save a script file `multiple_imputation.sh` to the root directory where each line is an independent command corresponding to a single run of the multiple imputation pipeline. For instance, the following command:

```bash
python fit.py datasets/phenotypes/data.csv --id_name ID --copymask_amount 0.5 --batch_size 2048 --epochs 1 --lr 0.1 --device cuda:1 --multiple 5
```

will save 5 lines to `multiple_imputation.sh` with the originally passed arguments:

```bash
python fit.py datasets/random/data.csv --id_name ID --copymask_amount 0.5 --batch_size 2048 --epochs 100 --lr 0.1 --device cuda:1 --seed 0 --bootstrap --save_imputed
...
python fit.py datasets/random/data.csv --id_name ID --copymask_amount 0.5 --batch_size 2048 --epochs 100 --lr 0.1 --device cuda:1 --seed 4 --bootstrap --save_imputed
```

Each command is responsible for saving one imputed version of the original data matrix in the format of `{file_location}/imputed_{data}_seed0_bootstrap.csv` and so on. Since each run is independent, the multiple runs are fully parallelizeable. This is recommended in a number of ways such as `parallel -j 5 < multiple_imputation.sh` on UNIX based systems, piping each line further into a job scheduler on compute clusters, and splitting compute load across multiple GPUs by altering the device flag. The script may also be executed as-is, which will impute each matrix sequentially.

## Imputation Quality

The need and extent to assess the quality of the imputation result may vary for each application.
We find in general that high variance ratio (variance of imputed over that of observed values) and high Pearson r^2 in a 1% simulation of missing values are indicators of reasonable imputation quality.
The r^2 can be also used to inform the effective sample size after imputation as N<sub>imputed</sub> * r^2 + N<sub>observed</sub>.

A recommended starting point to threshold features with reasonable imputation quality would be a variance ratio `> 0.2` and r^2 `> 0.2`. These conditions work best for features that have notable amounts of missingness (`> 10%` missing) to avoid edge cases. Based on these conditions, the quality output will also include a flag where `NOM`: no missing values, `LOM`: low missing values (<10%), `LOQ`: low variance ratio and r^2 (<0.2), `LOV` or `LOR`: either low variance (V) or r^2 (R) metric, or `QOK`: all quality conditions are met. These indications are provided only as suggestions for follow up analyses for each feature (Please note the sample data in this repository are randomly generated - therefore only few features will appear to have `QOK`).

With the `--quality` flag of `fit.py`, the script is capable of printing out the variance ratio and r^2 for each feature. This information will also be saved to a csv next to the original data file as `{file_location}/{datafile}_quality.csv`. This command can be mixed with `--save_imputed` for a model which was fitted in the same run or `--impute_using_saved` to use weights which were previously saved.

For example, running:

```bash
python fit.py datasets/phenotypes/data.csv --id_name ID --copymask_amount 0.5 --batch_size 2048 --epochs 20 --lr 0.1 --device cuda:1 --quality
```

gives the following printout:

```bash
Saving model to: datasets/phenotypes/data.pth
Dataset size: 300000
Features loaded: contin=8, binary=7
[E1 train 118/118] - L0.7167 (0.0806 0.3668) 6.4s LR:0.1
...
[E20 val 30/30] - L0.2894 (0.0376 0.2452) 1.4s LR:0.1
Loading last best checkpoint
(impute) Dataset size: 300000
Starting # observed values: 3126831
 Simulating missing values: 3095562.69 < 3091703
Imputing: 146/147
=================================================
Imputation Quality:
NOM missing=0.0% (no missing values) age
LOM missing=0.1% var_info=0.00 r2=0.02 effective=x1.0 insomnia.baseline
LOM missing=0.1% var_info=0.00 r2=0.03 effective=x1.0 alcoholuse.baseline
LOM missing=0.1% var_info=0.00 r2=0.00 effective=x1.0 alcoholfreq.baseline
LOQ missing=18.8% var_info=0.00 r2=0.15 effective=x1.0 neuroticismscore.baseline
LOQ missing=67.0% var_info=0.03 r2=0.12 effective=x1.2 happiness.baseline
LOQ missing=67.2% var_info=0.01 r2=0.08 effective=x1.2 cannabis.evertaken
LOQ missing=93.3% var_info=0.14 r2=0.19 effective=x3.6 cannabis.maxfreq
NOM missing=0.0% (no missing values) sex
LOQ missing=67.1% var_info=0.03 r2=0.01 effective=x1.0 anxietysocialphobia.diagnosis
QOK missing=79.9% var_info=0.22 r2=0.25 effective=x2.0 LifetimeMDD
LOM missing=1.3% var_info=0.05 r2=0.05 effective=x1.0 GPpsy
LOM missing=1.1% var_info=0.03 r2=0.04 effective=x1.0 Psypsy
LOQ missing=24.7% var_info=0.14 r2=0.11 effective=x1.0 SelfRepDep
LOQ missing=37.1% var_info=0.06 r2=0.05 effective=x1.0 ICD10Dep
=================================================
done
```

and saves the quality information to the csv: `datasets/phenotypes/data_quality.csv`.

## Citing AutoComplete

TBA
