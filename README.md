# A Flexible Debiasing Framework for Fair Heterogeneous Information Network Embedding
This repository contains the source code for our paper "A Flexible Debiasing Framework for Fair Heterogeneous Information Network Embedding" submitted to ECAI'23.

## Setup
This repository is built using PyTorch. The primary dependencies are listed as follows. You can install all the necessary libraries by running
`pip install -r ./requirements.txt`

### Dependencies
- Python == 3.9
- torch==1.12.1+cu102
- numpy==1.24.2
- scipy==1.10.1
- pandas==2.0.1
- networkx==3.1
- scikit\_learn==1.2.2
- geomloss==0.2.6
- dgl==1.0.1
- fairlearn==0.8.0   


## Datasets
We ran our experiments on three read-world HIN datasets, i.e., MovieLens, IMDB, and ACM. All the data are present in the './openhgnn/dataset/HGBl/' folder.

## Usage
The main script running the experiments of our attack strategy on the state-of-the-art GNN models is in './main_openhgnn.py'.

