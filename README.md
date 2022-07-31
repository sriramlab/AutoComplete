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

## Usage

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

## Citing AutoComplete

TBD
