#!/bin/bash

# simulate some missing values
python tutorials/phenotype_missingness_simulation.py

# fit the model
python fit.py datasets/phenotypes/data_fit.csv \
    --id_name ID \
    --copymask_amount 0.5 \
    --batch_size 2048 \
    --epochs 1 \
    --lr 0.1 \
    --device cpu:0

# impute a test split of the data
python fit.py datasets/phenotypes/data_fit.csv \
    --id_name ID \
    --impute_using_saved datasets/phenotypes/model.pth \
    --impute_data_file datasets/phenotypes/data_test.csv \
    --device cpu:0

# score the simulated missing portion
python bootstrap_r2_statistic.py datasets/phenotypes/data.csv \
    --simulated_data_file datasets/phenotypes/data_test.csv \
    --imputed_data_file datasets/phenotypes/imputed_data_test.csv \
    --num_bootstraps 100 \
    --saveas result_r2_phenotype_demo.csv

echo "done"