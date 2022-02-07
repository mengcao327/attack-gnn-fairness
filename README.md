# Fairness Attacks on Graph Neural Networks
This repository contains the source code for our paper "Fairness Attacks on Graph Neural Networks" submitted to KDD'22.

## Setup
This repository is built using PyTorch. We ran our code using Python == 3.9. You can install all the necessary libraries by running
`pip install -r ./requirements.txt`


## Datasets
We ran our experiments on three read-world datasets, i.e., Pokec_z, Pokec_n, and DBLP. All the data are present in the './datasets' folder.

## Usage
The main script running the experiments of our attack strategy on the state-of-the-art GNN models is in './src/train.py'.

### Examples

Script 1: Evaluate fairness and node classification performance of GCN on the clean graph:

`python train.py --dataset pokec_z  --model gcn  --attack_type none 
`

Script 2: Evaluate fairness and node classification performance of GCN on the perturbed graph with our FA-GNN at 5% perturbation rate, the attack are controlled by direction (node type of one attack subset) and strategy (same/different labels/sens):

`python train.py --dataset pokec_z   --model gcn  --attack_type fair_attack --direction y1s1 --strategy DD --ptb_rate  0.05
`

Script 3: Evaluate fairness and node classification performance of GAT on the perturbed graph with our FA-GNN at 5% perturbation rate, the attack are controlled by direction (node type of one attack subset) and strategy (same/different labels/sens): (GAT have hyperparameters different from the default ones only for Pokec_z and Pokec_n)

`python train.py --dataset pokec_z --model gat --attack_type fair_attack  --direction y1s1 --strategy DD  --ptb_rate 0.05 --dropout 0.8 --in_drop 0.338 --attn_drop 0.57 --negative_slope 0.15 --weight_decay 0.019 --num_layers 1 --num_heads 2  
`

### Outputs
The main script will output the fairness and classification metrics for each of the 5 random seeds as well as the mean and std values, the results will also be saved in a .csv file in the './results/' folder.
