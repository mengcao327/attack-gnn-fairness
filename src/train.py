import csv
import random
import time
import argparse
import numpy as np
# import scipy.sparse as sp
from random import choice
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import dgl
from attack.attack import attack

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
'''
            Dataset args
'''
parser.add_argument(
    '--dataset',
    type=str,
    default='dblp',
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
# parser.add_argument('--sens_number', type=int, default=200,
#                     help="the number of sensitive attributes")
# parser.add_argument('--label_number', type=int, default=500,
#                     help="the number of labels")
'''
            Model args
'''
parser.add_argument('--model', type=str, default='gcn',
                    choices=['gcn', 'gat', 'gsage'])
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--attack_type', type=str, default='none',
                    choices=['none', 'random', 'dice', 'metattack'],
                    help='Adversarial attack type.')
parser.add_argument("--num_layers", type=int, default=2,
                    help="number of hidden layers")
parser.add_argument('--agg_type', type=str, default='mean',
                    choices=['gcn', 'mean', 'pool', 'lstm'],
                    help='Aggregator for GraphSAGE')
# ----args for GAT
parser.add_argument("--num_heads", type=int, default=8,
                    help="number of hidden attention heads")
parser.add_argument("--num_out_heads", type=int, default=1,
                    help="number of output attention heads")
parser.add_argument("--in_drop", type=float, default=.6,
                    help="input feature dropout")
parser.add_argument("--attn_drop", type=float, default=.6,
                    help="attention dropout")
parser.add_argument('--negative_slope', type=float, default=0.2,
                    help="the negative slope of leaky relu")
parser.add_argument("--residual", action="store_true", default=False,
                    help="use residual connection")

'''
            Optimization args
'''
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument(
    '--acc',
    type=float,
    default=0.6,
    help='the selected FairGNN accuracy on val would be at least this high')
parser.add_argument(
    '--roc',
    type=float,
    default=0.5,
    help='the selected FairGNN ROC score on val would be at least this high')

args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed_set = [42, 0, 1, 2, 100]
# seed_set = [42]

# %%
FINAL_RESULT = []
N = len(seed_set)
for repeat in range(N):
    seed = seed_set[repeat]
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)

    # Load data
    print(args.dataset)
    if args.dataset in ['pokec_z', 'pokec_n', 'nba']:
        if args.dataset == 'pokec_z':
            dataset = 'region_job'

            sens_attr = "region"
            predict_attr = "I_am_working_in_field"
            # label_number = 100000
            # sens_number = args.sens_number
            # seed=20
            path = "../dataset/pokec/"
            test_idx = False

        elif args.dataset == 'pokec_n':
            dataset = 'region_job_2'
            sens_attr = "region"
            predict_attr = "I_am_working_in_field"
            # label_number = 100000
            # sens_number = args.sens_number
            # seed=20
            path = "../dataset/pokec/"
            test_idx = False
        elif args.dataset == 'nba':
            dataset = 'nba'
            sens_attr = "country"
            predict_attr = "SALARY"
            # label_number = 200
            # sens_number = 50
            # seed=20
            path = "../dataset/NBA"
            test_idx = True
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_pokec(dataset,
                                                                               sens_attr,
                                                                               predict_attr,
                                                                               path=path,
                                                                               train_percent=args.train_percent,
                                                                               val_percent=args.val_percent,
                                                                               seed=seed, test_idx=test_idx)
        if args.dataset == "nba":
            features = feature_norm(features)
    else:
        # Load credit_scoring dataset
        if args.dataset == 'credit':
            sens_attr = "Age"  # column number after feature process is 1
            sens_idx = 1
            predict_attr = 'NoDefaultNextMonth'
            label_number = 6000
            path_credit = "../dataset/credit"
            adj, features, labels, idx_train, idx_val, idx_test, sens = load_credit(
                args.dataset, sens_attr, predict_attr, path=path_credit, label_number=label_number)
            norm_features = feature_norm(features)
            norm_features[:, sens_idx] = features[:, sens_idx]
            features = norm_features

        # Load german dataset
        elif args.dataset == 'german':
            sens_attr = "Gender"  # column number after feature process is 0
            sens_idx = 0
            predict_attr = "GoodCustomer"
            label_number = 100
            path_german = "../dataset/german"
            adj, features, labels, idx_train, idx_val, idx_test, sens = load_german(
                args.dataset, sens_attr, predict_attr, path=path_german, label_number=label_number, )
        # Load bail dataset
        elif args.dataset == 'bail':
            sens_attr = "WHITE"  # column number after feature process is 0
            sens_idx = 0
            predict_attr = "RECID"
            label_number = 100
            path_bail = "../dataset/bail"
            adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail(
                args.dataset, sens_attr, predict_attr, path=path_bail, label_number=label_number, )
            norm_features = feature_norm(features)
            norm_features[:, sens_idx] = features[:, sens_idx]
            features = norm_features

        elif args.dataset == 'dblp':
            sens_attr = "gender"
            predict_attr = "label"
            path = "../dataset/dblp/"
            label_number = 500
            adj, features, labels, idx_train, idx_val, idx_test, sens = load_dblp(args.dataset,
                                                                                  sens_attr,
                                                                                  predict_attr,
                                                                                  path=path,
                                                                                  label_num=label_number,
                                                                                  seed=seed)
            features = feature_norm(features)

        else:
            print('Invalid dataset name!!')
            exit(0)

    # could add a step to extract LCC as adj-----TBD

    # if args.attack_type == 'rand':  # for random attack
    #     edge_perturbations = 0.25  # % edges to pertubate
    #     adj = rand_attack(adj, edge_perturbations)
    # elif args.attack_type == 'meta':  # for meta attack, it extracts LCC so the whole graph changes, need reload
    #     if args.dataset == 'nba':  # one dataset of nba as example, the rest data not uploaded
    #         adj, features, labels, idx_train, idx_val, idx_test, sens = load_attacked_graph()
    #     else:
    #         print("Attacked graph not found!")

    if args.attack_type != 'none':
        adj = attack(args.attack_type, 0.05, adj, features, labels, sens, idx_train, idx_val, idx_test, seed)

    print("Test samples:", len(idx_test))
    if sens_attr:
        sens[sens > 0] = 1

    # labels[labels > 1] = 1
    # G = dgl.DGLGraph()
    # G.from_scipy_sparse_matrix(adj)
    G = dgl.from_scipy(adj)
    if args.cuda:
        G = G.to(device)
    # Model and optimizer
    if args.model == 'gcn':  # two layer GCN
        from models.GCN import GCN

        model = GCN(G,
                    features.shape[1],  #
                    args.hidden,
                    1,
                    args.num_layers,
                    F.relu,
                    args.dropout)
    elif args.model == 'gsage':
        from models.GraphSAGE import GraphSAGE

        model = GraphSAGE(G,
                          features.shape[1],
                          args.hidden,
                          1,
                          args.num_layers,
                          F.relu,
                          args.dropout,
                          args.agg_type)
    elif args.model == 'gat':
        from models.GAT import GAT

        heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        model = GAT(G,
                    args.num_layers,
                    features.shape[1],
                    args.hidden,
                    1,
                    heads,
                    args.in_drop,
                    args.attn_drop,
                    args.negative_slope,
                    args.residual)
    if args.cuda:
        model.cuda()
        features = features.cuda()
        # adj = adj.cuda()
        sens = sens.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    loss_fcn = torch.nn.BCEWithLogitsLoss()

    # Train model
    t_total = time.time()
    vali_max = [0, [0, 0, 0, 0, 0, 0], [100, 100, 100], -1]
    loss_all = []
    for epoch in range(args.epochs + 1):
        # t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features)
        loss_train = loss_fcn(
            output[idx_train],
            labels[idx_train].unsqueeze(1).float())
        loss_all.append(loss_train.detach().cpu().item())
        # loss_train = F.binary_cross_entropy_with_logits(output[idx_sens_train], sens[idx_sens_train].unsqueeze(1).float())
        # acc_train = accuracy(output[idx_sens_train], sens[idx_sens_train])
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features)

        acc_val, roc_val, _, _, _, _ = classification_metrics(
            output[idx_val], labels[idx_val])
        parity_val, equality_val, eq_odds_val, _ = fair_metric(
            labels, output, idx_val, sens, 'val')

        acc_test, roc_test, p, r, maf1_test, mif1_test = classification_metrics(
            output[idx_test], labels[idx_test])
        parity, equality, eq_odds, middle_results = fair_metric(
            labels, output, idx_test, sens, 'test')

        print("Epoch [{}] Test set results:".format(epoch),
              "acc_test= {:.4f}".format(acc_test.item()),
              "acc_val: {:.4f}".format(acc_val.item()))
        if acc_val > args.acc:  # and roc_val > args.roc:
            # ----most fair
            # if best_fair > parity_val + equality_val+eq_odds_val:
            #     best_fair=parity_val + equality_val+eq_odds_val
            # ----most accurate
            if acc_val > vali_max[0]:
                vali_max = [
                    acc_val, [
                        acc_test.item(), roc_test, p, r, maf1_test, mif1_test], [
                        parity, equality, eq_odds], epoch + 1, middle_results]

            print("=================================")

            print('Epoch: {:04d}'.format(epoch + 1),
                  'acc_val: {:.4f}'.format(acc_val.item()),
                  "roc_val: {:.4f}".format(roc_val),
                  "parity_val: {:.4f}".format(parity_val),
                  "equality: {:.4f}".format(equality_val))
            print("Test:",
                  "accuracy: {:.4f}".format(acc_test.item()),
                  "roc: {:.4f}".format(roc_test),
                  "parity: {:.4f}".format(parity),
                  "equality: {:.4f}".format(equality))
            print("Best:",
                  "accuracy: {:.4f}".format(vali_max[1][0]),
                  "parity: {:.4f}".format(vali_max[2][0]),
                  "epoch: {0}".format(vali_max[3]))
    FINAL_RESULT.append(list(vali_max))
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    print('============performace on test set=============')
    # if len(best_result) > 0:
    print("Test:",
          "accuracy: {:.4f}".format(vali_max[1][0]),
          "auc: {:.4f}".format(vali_max[1][1]),
          "precision: {:.4f}".format(vali_max[1][2]),
          "recall: {:.4f}".format(vali_max[1][3]),
          "maf1: {:.4f}".format(vali_max[1][4]),
          "mif1: {:.4f}".format(vali_max[1][5]),
          "parity: {:.4f}".format(vali_max[2][0]),
          "equality: {:.4f}".format(vali_max[2][1]),
          "eq odds: {:.4f}".format(vali_max[2][2]),
          "epoch: {0}".format(vali_max[3]))
    # else:
    #     print("Please set smaller acc/roc thresholds")
print("\n")
sum_acc = []
sum_roc = []
sum_p = []
sum_r = []
sum_maf1 = []
sum_mif1 = []

sum_sp = []
sum_eq = []
sum_eo = []
for i in range(len(FINAL_RESULT)):
    print(
        "{0}:\tvali: {1:.4f}\t | test:  ACC: {2:.4f} AUC: {3:.4f} Precision:: {4:.4f} Recall: {5:.4f} MaF1: {6:.4f} MiF1: {7:.4f}, Parity:: {8:.4f} Equality:: {9:.4f} Eq_odds:: {10:.4f}".format(
            i,
            FINAL_RESULT[i][0],
            FINAL_RESULT[i][1][0],
            FINAL_RESULT[i][1][1],
            FINAL_RESULT[i][1][2],
            FINAL_RESULT[i][1][3],
            FINAL_RESULT[i][1][4],
            FINAL_RESULT[i][1][5],
            FINAL_RESULT[i][2][0],
            FINAL_RESULT[i][2][1],
            FINAL_RESULT[i][2][2]))
    print("epoch=", FINAL_RESULT[i][3])
    sum_acc.append(FINAL_RESULT[i][1][0])
    sum_roc.append(FINAL_RESULT[i][1][1])
    sum_p.append(FINAL_RESULT[i][1][2])
    sum_r.append(FINAL_RESULT[i][1][3])
    sum_maf1.append(FINAL_RESULT[i][1][4])
    sum_mif1.append(FINAL_RESULT[i][1][5])
    sum_sp.append(FINAL_RESULT[i][2][0])
    sum_eq.append(FINAL_RESULT[i][2][1])
    sum_eo.append(FINAL_RESULT[i][2][2])
print("mean test acc:", np.mean(sum_acc))
print("std test acc:", np.std(sum_acc))
print("mean test roc:", np.mean(sum_roc))
print("std test roc:", np.std(sum_roc))
print("mean test precision:", np.mean(sum_p))
print("std test precision:", np.std(sum_p))
print("mean test recall:", np.mean(sum_r))
print("std test recall:", np.std(sum_r))
print("mean test maf1:", np.mean(sum_maf1))
print("std test maf1:", np.std(sum_maf1))
print("mean test mif1:", np.mean(sum_mif1))
print("std test mif1:", np.std(sum_mif1))

print("mean test statistical parity:", np.mean(sum_sp))
print("std test statistical parity:", np.std(sum_sp))
print("mean test equal opportunity:", np.mean(sum_eq))
print("std test equal opportunity:", np.std(sum_eq))
print("mean test equal odds:", np.mean(sum_eo))
print("std test equal odds:", np.std(sum_eo))

FINAL_RESULT_DICT_LIST = []

for i in range(N):
    FINAL_RESULT_DICT = {}
    FINAL_RESULT_DICT['acc'] = FINAL_RESULT[i][1][0]
    FINAL_RESULT_DICT['auc'] = FINAL_RESULT[i][1][1]
    FINAL_RESULT_DICT['precision'] = FINAL_RESULT[i][1][2]
    FINAL_RESULT_DICT['recall'] = FINAL_RESULT[i][1][3]
    FINAL_RESULT_DICT['maf1'] = FINAL_RESULT[i][1][4]
    FINAL_RESULT_DICT['mif1'] = FINAL_RESULT[i][1][5]

    FINAL_RESULT_DICT['yp1.a1'] = FINAL_RESULT[i][4]['yp1.a1']
    FINAL_RESULT_DICT['yp1.a0'] = FINAL_RESULT[i][4]['yp1.a0']
    FINAL_RESULT_DICT['yp1.y1a1'] = FINAL_RESULT[i][4]['yp1.y1a1']
    FINAL_RESULT_DICT['yp1.y1a0'] = FINAL_RESULT[i][4]['yp1.y1a0']
    FINAL_RESULT_DICT['yp1.y0a1'] = FINAL_RESULT[i][4]['yp1.y0a1']
    FINAL_RESULT_DICT['yp1.y0a0'] = FINAL_RESULT[i][4]['yp1.y0a0']

    FINAL_RESULT_DICT['parity'] = FINAL_RESULT[i][2][0]
    FINAL_RESULT_DICT['equality'] = FINAL_RESULT[i][2][1]
    FINAL_RESULT_DICT['eq_odds'] = FINAL_RESULT[i][2][2]

    FINAL_RESULT_DICT_LIST.append(FINAL_RESULT_DICT)
#
fieldnames = [
    'acc',
    'auc',
    'precision',
    'recall',
    'maf1',
    'mif1',
    'yp1.a1',
    'yp1.a0',
    'yp1.y1a1',
    'yp1.y1a0',
    'yp1.y0a1',
    'yp1.y0a0',
    'parity',
    'equality',
    'eq_odds']
fname = 'result-' + str(args.dataset) + '-' + str(args.model) + \
        '-' + str(args.attack_type) + '.csv'
with open(fname, 'w', encoding='UTF8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(FINAL_RESULT_DICT_LIST)
f.close()

# check loss
import scipy.io

fname = 'loss-' + str(args.dataset) + '-' + str(args.model) + \
        '-' + str(args.attack_type) + '.mat'
scipy.io.savemat(fname, mdict={'loss': loss_all})
