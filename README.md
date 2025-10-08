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

## Imputation tutorial

As detecting binary phenotypes automatically can be ambiguous, the scripts assume that continuous phenotype columns **appear first**, then the binary phenotypes.

In addition, it must be indicated which columns are binary in a `metadata.csv` in the same location as the data matrix being loaded.

Once the files are prepared, the following command can be run to train and save the imputation model:

```bash
python train.py path_to_dataset_file.csv deepcoder cuda:0 0.8 save_model_location.pth
```

The script can be pointed to a data matrix in csv or tsv format anywhere (`path_to_dataset_file.csv`). The first column must be named **FID** and is assumed to contain the unique identifier for each sample. The script will run with logs such as the following and save the trained model to `save_model_location.pth`.

```bash
Config:
deepcoder
EP 200
LR 0.1
MOM 0.9
BATCH 2048
COPYMASK True 0.8
DEVICE cuda:0
SEED 0
BOOT None
Random Masking: ON
Mask Type     : {'copy_mask': 0.8}
N             : 141700
Random Masking: ON
Mask Type     : {'fixed_copy_mask': True}
N             : 35426
# Cats: 98 274
Saving model to: here3.pth
WIDTH 1
DEPTH 1
MULT 0
In D 372
OutD 372
Dropout None
Zlist [372, 372]
Spec: [(372, 372)]
Zdim: 372
DeepCoder(
  (net): Sequential(
    (0): Linear(in_features=372, out_features=372, bias=True)
    (1): LeakyReLU(negative_slope=0.01)
    (2): Linear(in_features=372, out_features=372, bias=True)
  )
)
Weight hash: [-0.00038817  0.0278133  -0.0426729  -0.03815666 -0.01996932]
starting lr 0.1
cont/binary: 98/274
[E1 train 70/70] - L1.5987 7.8s LR:0.1
[E1 test 18/18] - L1.1981 1.0s LR:0.1
save!
[E2 train 70/70] - L1.1249 7.7s LR:0.1
[E2 test 18/18] - L0.9803 0.8s LR:0.1
save!
...
```

The default imputation neural network is defined in `nets.py` and can be further experimented/modified.

> NOTE: the script reserves ~20% of the loaded data matrix for validation. It is recommended to split or create a separate data matrix for testing.

Once the model is trained, it can be used to impute the same matrix, or another test matrix.

```bash
python impute.py path_to_dataset_file.csv cuda:0 save_model_location.pth imputed_file.csv
```

## Citation

If you found this project useful, please cite our work:

```bibtex
@article{an2023deep,
  title={Deep learning-based phenotype imputation on population-scale biobank data increases genetic discoveries},
  author={An, Ulzee and Pazokitoroudi, Ali and Alvarez, Marcus and Huang, Lianyun and Bacanu, Silviu and Schork, Andrew J and Kendler, Kenneth and Pajukanta, P{\"a}ivi and Flint, Jonathan and Zaitlen, Noah and others},
  journal={Nature Genetics},
  volume={55},
  number={12},
  pages={2269--2276},
  year={2023},
  publisher={Nature Publishing Group US New York}
}
```