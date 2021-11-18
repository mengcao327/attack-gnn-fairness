import csv
import random
import time
import argparse
import numpy as np
# import scipy.sparse as sp
from random import choice
import torch
import torch.nn.functional as F
import torch.optim as optim
import dgl
import networkx as nx
from utils import *
import datetime
import matplotlib.pyplot as plt
import scipy.sparse as sp
import os


def plot_centrality(centrality_vals, metrics, idx_train, idx_val, idx_test, args, seed):
    print(f"plotting results for attack type {args.attack_type}...")
    for i in range(len(metrics)):
        fig, axs = plt.subplots(1, 3, sharey=True, sharex=True, tight_layout=True)
        # fig, axs = plt.subplots(1, 3)
        n_bins = 50
        axs[0].hist(centrality_vals[i][idx_train], n_bins)
        axs[1].hist(centrality_vals[i][idx_val], n_bins)
        axs[2].hist(centrality_vals[i][idx_test], n_bins)

        # plt.xlabel(dataset)
        # plt.ylabel(metrics[i])
        # plt.title(dataset + '-' + args.attack_type+ '-' +metrics[i])
        axs[0].set_xlabel('train')
        axs[0].set_ylabel("Count")
        axs[1].set_xlabel('val')
        axs[2].set_xlabel('test')
        if args.attack_type == "none":
            fname=args.dataset + '-' + args.attack_type + '-' + str(seed)+ '-' + metrics[i]
            fig.suptitle(fname)
            fig.savefig(
                '../results/graph_evaluate/' + fname + '.jpg',
                dpi=fig.dpi)
        else:
            fname=args.dataset + '-' + args.attack_type + '-' + args.ptb_rate + '-' + str(seed) + '-' + metrics[i]
            fig.suptitle(fname)
            fig.savefig(
                '../results/graph_evaluate/' + fname + '.jpg',
                dpi=fig.dpi)
        # plt.show()
    print("finish!")


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
'''
            Dataset args
'''
parser.add_argument(
    '--dataset',
    type=str,
    default='bail',
    choices=[
        'pokec_z',
        'pokec_n',
        'nba',
        'credit',
        'german',
        'bail',
        'dblp'])
parser.add_argument('--train_percent', type=float, default=0.5,
                    help='Percentage of labeled data as train set.')
parser.add_argument('--val_percent', type=float, default=0.25,
                    help='Percentage of labeled data as validation set.')
parser.add_argument('--sens_number', type=int, default=200,
                    help="the number of sensitive attributes")
parser.add_argument('--seed', type=int, default=100,
                    help="the random seed")
parser.add_argument('--attack_type', type=str, default='none',
                    # choices=['none', 'random', 'dice', 'metattack', 'sacide', 'structack_dg_comm', 'structack_pr_katz'],
                    help='Adversarial attack type.')
parser.add_argument("--preprocess_pokec", type=bool, default=False,
                    help="Include only completed accounts in Pokec datasets (only valid when dataset==pokec_n/pokec_z])")
parser.add_argument('--ptb_rate', type=str, default='0.05',
                    help="Attack perturbation rate [0-1]")

args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# seed_set = [42, 0, 1, 2, 100]
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
if args.cuda:
    torch.cuda.manual_seed(seed)
dataset_name = args.dataset
if args.dataset in ['pokec_z', 'pokec_n', 'nba']:
    if args.dataset == 'pokec_z':
        dataset = 'region_job'
        dataset_name = dataset
        sens_attr = "region"
        predict_attr = "I_am_working_in_field"
        # label_number = 100000
        sens_number = args.sens_number

        path = "../dataset/pokec/"
        test_idx = False

    elif args.dataset == 'pokec_n':
        dataset = 'region_job_2'
        dataset_name = dataset
        sens_attr = "region"
        predict_attr = "I_am_working_in_field"
        # label_number = 100000
        sens_number = args.sens_number

        path = "../dataset/pokec/"
        test_idx = False

    elif args.dataset == 'nba':
        dataset = 'nba'
        sens_attr = "country"
        predict_attr = "SALARY"
        # label_number = 200
        sens_number = 50
        # seed=20
        path = "../dataset/NBA"
        test_idx = False
    adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_pokec(dataset,
                                                                                           sens_attr,
                                                                                           predict_attr,
                                                                                           path=path,
                                                                                           train_percent=args.train_percent,
                                                                                           val_percent=args.val_percent,
                                                                                           sens_number=sens_number,
                                                                                           seed=seed,
                                                                                           test_idx=test_idx)
    if args.preprocess_pokec and 'pokec' in args.dataset:
        print(f'Preprocessing {dataset}')
        adj, features, labels, idx_train, idx_val, idx_test, sens = preprocess_pokec_complete_accounts(adj,
                                                                                                       features,
                                                                                                       labels,
                                                                                                       sens,
                                                                                                       seed)
        dataset += '_completed_accounts'

    if args.dataset == "nba":
        features = feature_norm(features)
else:

    # Load credit_scoring dataset
    if args.dataset == 'credit':
        sens_attr = "Age"  # column number after feature process is 1
        sens_idx = 1
        predict_attr = 'NoDefaultNextMonth'
        # label_number = 6000
        path_credit = "../dataset/credit"
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_credit(
            args.dataset, sens_attr, predict_attr, path=path_credit, train_percent=args.train_percent,
            val_percent=args.val_percent, sens_number=args.sens_number, seed=seed)
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features

    # Load german dataset
    elif args.dataset == 'german':
        sens_attr = "Gender"  # column number after feature process is 0
        sens_idx = 0
        predict_attr = "GoodCustomer"
        # label_number = 100
        path_german = "../dataset/german"
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_german(
            args.dataset, sens_attr, predict_attr, path=path_german, train_percent=args.train_percent,
            val_percent=args.val_percent, sens_number=args.sens_number, seed=seed)
    # Load bail dataset
    elif args.dataset == 'bail':
        sens_attr = "WHITE"  # column number after feature process is 0
        sens_idx = 0
        predict_attr = "RECID"
        # label_number = 100
        path_bail = "../dataset/bail"
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_bail(
            args.dataset, sens_attr, predict_attr, path=path_bail, train_percent=args.train_percent,
            val_percent=args.val_percent, sens_number=args.sens_number, seed=seed)
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features

    elif args.dataset == 'dblp':
        sens_attr = "gender"
        predict_attr = "label"
        path = "../dataset/dblp/"
        # label_number = 1000
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_dblp(args.dataset,
                                                                                              sens_attr,
                                                                                              predict_attr,
                                                                                              path=path,
                                                                                              train_percent=args.train_percent,
                                                                                              val_percent=args.val_percent,
                                                                                              sens_number=args.sens_number,
                                                                                              seed=seed)
        # features = feature_norm(features)
        #  normalization may cause problem for dblp: model not converge

    else:
        print('Invalid dataset name!!')
        exit(0)

if args.attack_type != 'none':
    # cached_filename = f'../dataset/cached_attacks/{args.dataset}_{args.attack_type}_{args.ptb_rate:.2f}_{seed}.npz'
    cached_filename = '../dataset/cached_attacks/' + dataset_name + '_' + args.attack_type + '_' + args.ptb_rate + '_' + str(
        args.seed) + '.npz'
    # check if modified_adj of (dataset_name, attack_name, ptb_rate, seed) are stored
    print(f'attempting to load {cached_filename}')
    if os.path.exists(cached_filename):
        print(f'Perturbed adjacency matrix already exists at {cached_filename}. Loading...')
        modified_adj = sp.load_npz(cached_filename)
        print('Perturbed adjacency matrix loaded successfully!')
    adj = modified_adj
    # from attack.attack import attack
    #
    # adj = attack(args.attack_type, ptb_rate, adj, features, labels, sens, idx_train, idx_val, idx_test,
    #              seed, dataset)

G = nx.from_scipy_sparse_matrix(adj)
# print(nx.number_of_nodes(G))

st = datetime.datetime.now()
dc = np.array(list(nx.degree_centrality(G).values()))
et = datetime.datetime.now()
print(f"degree running for {(et-st).seconds/60.0} minutes")

st = datetime.datetime.now()
pr = np.array(list(nx.pagerank_scipy(G).values()))
et = datetime.datetime.now()
print(f"pagerank running for {(et-st).seconds/60.0} minutes")

if args.dataset in ['nba', 'german']:  # small datasets
    st = datetime.datetime.now()
    cc = np.array(list(nx.closeness_centrality(G).values()))
    et = datetime.datetime.now()
    print(f"closeness running for {(et-st).seconds/60.0} minutes")

    st = datetime.datetime.now()
    bc = np.array(list(nx.betweenness_centrality(G).values()))
    et = datetime.datetime.now()
    print(f"betweeness running for {(et-st).seconds/60.0} minutes")

    centrality_vals = [bc, cc, dc, pr]
    metrics = ['betweeness', 'closeness', 'degree', 'pagerank']
else:
    centrality_vals = [dc, pr]
    metrics = ['degree', 'pagerank']

plot_centrality(centrality_vals, metrics, idx_train, idx_val, idx_test, args, seed)
