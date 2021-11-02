# %%
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from models.FairGNN import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=800,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--sens_number', type=int, default=200,
                    help="the number of sensitive attributes")
parser.add_argument('--dataset', type=str, default='pokec_z',
                    choices=['pokec_z', 'pokec_n', 'nba', 'credit',
                             'german',
                             'bail',
                             'dblp'])
parser.add_argument('--train_percent', type=float, default=0.5,
                    help='Percentage of labeled data as train set.')
parser.add_argument('--val_percent', type=float, default=0.25,
                    help='Percentage of labeled data as validation set.')
args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# %%
# %%
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
print(args.dataset)

if args.dataset in ['pokec_z', 'pokec_n', 'nba']:
    if args.dataset == 'pokec_z':
        dataset = 'region_job'

        sens_attr = "region"
        predict_attr = "I_am_working_in_field"
        # label_number = 100000
        sens_number = args.sens_number
        # seed=20
        path = "../dataset/pokec/"
        test_idx = False

    elif args.dataset == 'pokec_n':
        dataset = 'region_job_2'
        sens_attr = "region"
        predict_attr = "I_am_working_in_field"
        # label_number = 100000
        sens_number = args.sens_number
        # seed=20
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
        test_idx = True
    adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_pokec(dataset,
                                                                                           sens_attr,
                                                                                           predict_attr,
                                                                                           path=path,
                                                                                           train_percent=args.train_percent,
                                                                                           val_percent=args.val_percent,
                                                                                           sens_number=sens_number,
                                                                                           seed=args.seed,
                                                                                           test_idx=test_idx)
    # if args.preprocess_pokec and 'pokec' in args.dataset:
    #     print(f'Preprocessing {dataset}')
    #     adj, features, labels, idx_train, idx_val, idx_test, sens = preprocess_pokec_complete_accounts(adj, features,
    #                                                                                                    labels, sens,
    #                                                                                                    args.seed)
    #     dataset += '_completed_accounts'

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
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_credit(
            args.dataset, sens_attr, predict_attr, path=path_credit, label_number=label_number,
            sens_number=args.sens_number, seed=args.seed)
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
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_german(
            args.dataset, sens_attr, predict_attr, path=path_german, label_number=label_number,
            sens_number=args.sens_number, seed=args.seed)
    # Load bail dataset
    elif args.dataset == 'bail':
        sens_attr = "WHITE"  # column number after feature process is 0
        sens_idx = 0
        predict_attr = "RECID"
        label_number = 100
        path_bail = "../dataset/bail"
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_bail(
            args.dataset, sens_attr, predict_attr, path=path_bail, label_number=label_number,
            sens_number=args.sens_number, seed=args.seed)
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features

    elif args.dataset == 'dblp':
        sens_attr = "gender"
        predict_attr = "label"
        path = "../dataset/dblp/"
        label_number = 1000
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_dblp(args.dataset,
                                                                                              sens_attr,
                                                                                              predict_attr,
                                                                                              path=path,
                                                                                              label_num=label_number,
                                                                                              sens_number=args.sens_number,
                                                                                              seed=args.seed)
        # features = feature_norm(features)
        #  normalization may cause problem for dblp: model not converge

    else:
        print('Invalid dataset name!!')
        exit(0)

print(len(idx_test))
# %%
import dgl
# from utils import feature_norm

G = dgl.from_scipy(adj)
if args.cuda:
    G = G.to(device)
# %%
sens[sens > 0] = 1
if sens_attr:
    sens[sens > 0] = 1
# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    # adj = adj.cuda()
    sens = sens.cuda()
    # idx_sens_train = idx_sens_train.cuda()
    # idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    sens = sens.cuda()
    idx_sens_train = idx_sens_train.cuda()

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Train model
t_total = time.time()
best_acc = 0.0
best_test = 0.0
for epoch in range(args.epochs + 1):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(G, features)
    loss_train = F.binary_cross_entropy_with_logits(output[idx_sens_train], sens[idx_sens_train].unsqueeze(1).float())
    acc_train = accuracy(output[idx_sens_train], sens[idx_sens_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(G, features)
    if epoch % 10 == 0:
        acc_val = accuracy(output[idx_val], sens[idx_val])
        acc_test = accuracy(output[idx_test], sens[idx_test])
        print("Epoch [{}] Test set results:".format(epoch),
              "acc_test= {:.4f}".format(acc_test.item()),
              "acc_val: {:.4f}".format(acc_val.item()))
        if acc_val > best_acc:
            best_acc = acc_val
            best_test = acc_test
            torch.save(model.state_dict(), "./checkpoint/GCN_sens_{}_ns_{}".format(args.dataset, sens_number))
print("The accuracy of estimator: {:.4f}".format(best_test))

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# %%
