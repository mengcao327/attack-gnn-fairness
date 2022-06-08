# Adversarial Attacks on the Fairness of Graph Neural Networks
This repository contains the source code for our paper "Adversarial Inter-Group Link Injection Leads to Unfair Graph Neural Networks" submitted to ICDM'22.

## Setup
This repository is built using PyTorch. The primary dependencies are listed as follows. You can install all the necessary libraries by running
`pip install -r ./requirements.txt`

### Dependencies
- Python == 3.9
- torch == 1.9.0
- cuda == 11.5
- numpy == 1.20.3
- scipy == 1.5.4
- networkx == 2.6.2
- dgl == 0.7.0
- deeprobust == 0.2.2
- scikit-learn == 0.24.2


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

## Appendix A: supplementary experiments on heterophilic and random graphs

### Results on heterophilic graphs
We perform the same experiments on synthetic datasets described in Sectio 3.4. We start with a heterophilic graph: the edge density is $0.0016$ for subsets of the same class and $0.004$ for subsets of opposite classes. 

The following Figure A.1 shows the statistical parity difference and the error rates of this simulation and supports our hypotheses. In this setup, EE only shows an increase in $FNR_{s_1}$ at the beginning. This could be due to $y_1s_1$ obtaining a similar neighborhood distribution to one subset and then gaining its unique neighborhood distribution.

![repo-het-result](https://user-images.githubusercontent.com/13431345/172593537-14ee234d-2ed7-4d4b-9b81-09746bd4ed8b.jpg)
<!-- <img src="https://user-images.githubusercontent.com/13431345/172593537-14ee234d-2ed7-4d4b-9b81-09746bd4ed8b.jpg" width="800"> -->
**Figure A.1** Fairness attacks on heterophilic synthetic graph. We show the statistical parity difference ($SPD$) (top) and error rates (bottom) of FA-GNN strategies on heterophilic synthetic graph. **(Top)** Each figure corresponds to a linking strategy applied on subset $y_1s_1$.% in a synthetic heterophilic graph While DE and ED do not have a targeted influence on $SPD$, DD decreases $SPD$, and EE only increases it at the beginning. **(Bottom)** $FPR_{s_j}$ and $FNR_{s_j}$ are the false positive and negative rates on subset $s_j$. Attacks that decrease label homophily (DD and DE) \textit{decrease} error on the involved subsets ($FNR_{s_1}, FPR_{s_0}$ with DD and $FNR_{s_1},FPR_{s_1}$ with DE). Attacks that increase label homophily (ED and EE) \textit{increase} the error rates on the involved subsets ($FNR_{s_1},FNR_{s_0}$ with ED and $FNR_{s_1}$ with EE at the beginning). This shows that for heterophilic graphs, DD and EE are still effective attacks.


### Results on random graphs
On random graphs, we generally see similar results, except that error rates do not go up with same-class linking (ED/EE).
We leave out the detailed results here.

## Appendix B: model implementations
### GNN models
- GCN, GAT, and GraphSAGE: we adopt the implementations provided in the DGL package (https://github.com/dmlc/dgl/). 
- FairGNN: we adopt the implementation provided by the authors (https://github.com/EnyanDai/FairGNN).
- NIFTY: we adopt the implementation provided by the authors (https://github.com/chirag126/nifty).

### Graph Attack Baselines
- Random and DICE: we adopt the implementation provided in the deeprobust package (https://github.com/DSE-MSU/DeepRobust). 
- PR-BCD: we use the implementation provided by the authors (https://github.com/sigeisler/robustness_of_gnns_at_scale).
