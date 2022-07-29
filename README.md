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

To use any GPU available, the `--device cpu:0` flag can be changed to `--device gpu:0`.

The `id_name` option specifies which column of the dataset to use as an identifier for each sample.
Continuous or binary-valued features will be automatically detected.

## Citing AutoComplete

TBD
