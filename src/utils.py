# %%
import numpy as np
import scipy.sparse as sp
import torch
import os
import pandas as pd
import dgl
import random
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, precision_score, roc_curve
from scipy.spatial import distance_matrix
import networkx as nx
from scipy.sparse.csgraph import connected_components

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, desc=None):
        return x


def load_perturbed_adj(dataset_name, attack_name, ptb_rate, seed):
    cached_filename = f'../dataset/cached_attacks/{dataset_name}_{attack_name}_{ptb_rate:.2f}_{seed}.npz'
    if os.path.exists(cached_filename):
        print(f'Perturbed adjacency matrix already exists at {cached_filename}. Loading...')
        modified_adj = sp.load_npz(cached_filename)
        print('Perturbed adjacency matrix loaded successfully!')
        return modified_adj
    else:
        print(f"Perturbed graph {cached_filename} does not exist")
        return None


def load_dataset(args, seed):
    sens_number=args.sens_number
    if args.dataset in ['pokec_z', 'pokec_n', 'nba']:
        if args.dataset == 'pokec_z':
            sens_attr = args.sensitive
            dataset = 'region_job'

            predict_attr = "I_am_working_in_field"
            # label_number = 100000
            sens_number = args.sens_number

            path = "../dataset/pokec/"
            test_idx = False

        elif args.dataset == 'pokec_n':
            sens_attr = args.sensitive
            dataset = 'region_job_2'
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
        # if args.preprocess_pokec and 'pokec' in args.dataset:
        #     # deprecated
        #     print(f'(DEPRECATED) Preprocessing {dataset}')
        #     adj, features, labels, idx_train, idx_val, idx_test, sens = preprocess_pokec_complete_accounts(adj,
        #                                                                                                    features,
        #                                                                                                    labels,
        #                                                                                                    sens,
        #                                                                                                    seed)
        #     dataset += '_completed_accounts'

        if args.dataset == "nba":
            features = feature_norm(features)
    else:
        # Load credit_scoring dataset
        if args.dataset == 'credit':
            dataset = 'credit'
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
            dataset = 'german'
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
            dataset = 'bail'
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
            dataset = 'dblp'
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
    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train, dataset, sens_attr, sens_number


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.
    Parameters
    ----------
    adj : gust.SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.
    Returns
    -------
    sparse_graph : gust.SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.
    """
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep


    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep

def load_data(path="../dataset/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    print(labels)

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def fair_metric(labels, output, idx, sens, status):
    mid_result = {}
    val_y = labels[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()] == 0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()] == 1

    idx_s0_y1 = np.bitwise_and(idx_s0, val_y == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, val_y == 1)
    idx_s0_y0 = np.bitwise_and(idx_s0, val_y == 0)
    idx_s1_y0 = np.bitwise_and(idx_s1, val_y == 0)

    pred_y = (output[idx].squeeze() > 0).type_as(labels).cpu().numpy()
    parity = abs(sum(pred_y[idx_s0]) / sum(idx_s0) -
                 sum(pred_y[idx_s1]) / sum(idx_s1))
    mid_result['yp1.a1'] = sum(pred_y[idx_s1]) / sum(idx_s1)
    mid_result['yp1.a0'] = sum(pred_y[idx_s0]) / sum(idx_s0)
    equality = abs(sum(pred_y[idx_s0_y1]) /
                   sum(idx_s0_y1) -
                   sum(pred_y[idx_s1_y1]) /
                   sum(idx_s1_y1))
    mid_result['yp1.y1a1'] = sum(pred_y[idx_s1_y1]) / sum(idx_s1_y1)
    mid_result['yp1.y1a0'] = sum(pred_y[idx_s0_y1]) / sum(idx_s0_y1)
    eq_odds = abs(sum(pred_y[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred_y[idx_s1_y1]) / sum(idx_s1_y1)) + \
              abs(sum(pred_y[idx_s0_y0]) / sum(idx_s0_y0) - sum(pred_y[idx_s1_y0]) / sum(idx_s1_y0))
    mid_result['yp1.y0a1'] = sum(pred_y[idx_s1_y0]) / sum(idx_s1_y0)
    mid_result['yp1.y0a0'] = sum(pred_y[idx_s0_y0]) / sum(idx_s0_y0)
    # dis_imp = (sum(pred_y[idx_s1]) / sum(idx_s1)) / (sum(pred_y[idx_s0]) / sum(idx_s0))
    mid_result['y1a1'] = sum(idx_s1_y1)
    mid_result['y1a0'] = sum(idx_s0_y1)
    mid_result['y0a1'] = sum(idx_s1_y0)
    mid_result['y0a0'] = sum(idx_s0_y0)
    # print(status,':y1a1:',mid_result['y1a1']) # evaluate the label and SA distribution
    # print(status,':y1a0:',mid_result['y1a0'])
    # print(status,':y0a1:',mid_result['y0a1'])
    # print(status,':y0a0:',mid_result['y0a0'])
    # print(status, ':y1:', mid_result['y1a1']+mid_result['y1a0'])
    # print(status, ':y0:', mid_result['y0a1']+mid_result['y0a0'])
    return parity, equality, eq_odds, mid_result  # ,dis_imp,mid_result


def load_pokec(
        dataset,
        sens_attr,
        predict_attr,
        path="../dataset/pokec/",
        train_percent=0.5,
        val_percent=0.25,
        sens_number=500,
        seed=42,
        test_idx=False):
    """Load data"""
    print('Loading {} dataset from {}'.format(dataset, path))

    idx_features_labels = pd.read_csv(
        os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    if 'region_job' in dataset:
        for attr in 'gender AGE region'.split(): # TODO binarize AGE
            header.remove(attr)
    else:
        header.remove(sens_attr)
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    sens = idx_features_labels[sens_attr].values
    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=str)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        os.path.join(
            path,
            "{}_relationship.txt".format(dataset)),
        dtype=str)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    # the original edges has some redundancy
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # this operation will make the adj symmetric and some redundant edges in
    # original data will have no influence

    # extract largested connected components
    print(f"{adj.shape[0]} of nodes before selecting LCC")
    _adj = adj.tocsr()
    lcc = largest_connected_components(_adj)
    _A = _adj[lcc][:, lcc]
    features = features[lcc]
    labels = labels[lcc]
    sens = sens[lcc]
    adj = _A.tocoo()
    print(f"{adj.shape[0]} of nodes after selecting LCC")
    # features = normalize(features)
    # so the true number of edges should be (nnz-selfloop)/2 in adj
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    random.seed(seed)
    label_idx = np.where(labels >= 0)[0]
    random.shuffle(label_idx)
    # print("num labels:", len(label_idx))

    idx_train = label_idx[:int(train_percent * len(label_idx))]
    idx_val = label_idx[int(train_percent * len(label_idx)):int((train_percent + val_percent) * len(label_idx))]
    if test_idx:
        idx_test = label_idx[int(train_percent * len(label_idx)):]
        idx_val = idx_test
    else:
        idx_test = label_idx[int(
            (train_percent + val_percent) * len(label_idx)):]

    labels[labels > 1] = 1


    sens_idx = set(np.where(sens >= 0)[0])
    print("num nodes with sa:", len(sens_idx))
    idx_test = np.asarray(list(sens_idx & set(idx_test)))

    check_dataset(labels, sens, idx_train, idx_val, idx_test)

    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # random.shuffle(sens_idx)

    return adj, features, labels, idx_train, idx_val, idx_test, sens,idx_sens_train


def preprocess_pokec_complete_accounts(adj, features, labels, sens, seed):
    """
    Pre-processes Pokec dataset by retaining only complete accounts (accounts where label!=-1).
    :param adj: original adjacency matrix
    :param features: original feature matrix
    :param labels: original labels
    :param sens: original sensitive attributes
    :return: The largest connected component in that subgraph with the projected features, labels and sens
    """
    idx = np.array(range(labels.shape[0]))[labels != -1]
    G = nx.from_scipy_sparse_matrix(adj)
    G = nx.subgraph(G, idx)
    print(f'Initial subgraph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges')
    cc = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    print(f'{len(cc)} connected components')
    idx = np.array(sorted(list(cc[0])))
    idx_map = {x: i for i, x in enumerate(idx)}
    G = nx.subgraph(G, idx)
    print(f'Final subgraph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges')
    G = nx.relabel_nodes(G, idx_map)
    adj = nx.to_scipy_sparse_matrix(G)
    features = features[idx, :]
    labels = labels[idx]
    sens = sens[idx]


    idx_all = np.array(range(len(labels)))[labels != -1]

    np.random.seed(seed)

    idx_train = np.random.choice(idx_all, int(features.shape[0] * 0.9), replace=False)
    idx_all = np.array(list(set(idx_all).difference(set(idx_train))))

    idx_val = np.random.choice(idx_all, int(features.shape[0] * 0.05), replace=False)
    idx_all = np.array(list(set(idx_all).difference(set(idx_val))))
    idx_test = idx_all

    print(idx_train.shape)
    print(idx_val.shape)
    print(idx_test.shape)

    return adj, features, labels, torch.LongTensor(idx_train), torch.LongTensor(idx_val), torch.LongTensor(idx_test), sens

def check_dataset(labels,sens,idx_train,idx_val,idx_test):
    label_idx = np.where(labels >= 0)[0]

    print("num labels:", len(label_idx))
    # check label balancing
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    print("num labels 0:", len(label_idx_0))
    print("num labels 1:", len(label_idx) - len(label_idx_0))
    print("num labels 1:", len(label_idx_1))


    sens_idx = set(np.where(sens >= 0)[0])
    print("num nodes with sa:", len(sens_idx))
    # check sa
    sens_idx_0 = set(np.where(sens == 0)[0])
    print("num nodes with sa=0:", len(sens_idx_0))
    print("num nodes with sa=1:", len(sens_idx) - len(sens_idx_0))

    print(f"Data splits: {len(idx_train)} train, {len(idx_val)} val, {len(idx_test)} test. ")


def load_dblp(dataset,
              sens_attr,
              predict_attr,
              path="../dataset/dblp/",
              train_percent=0.5,
              val_percent=0.25,
              sens_number=200,
              seed=42):
    """Load data"""
    print('Loading {} dataset from {}'.format(dataset, path))

    idx_features_labels = pd.read_csv(
        os.path.join(path, "{}.csv".format(dataset)))

    # features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    # features=np.load(os.path.join(path, "{}_emb.npy".format(dataset)))
    features = sp.load_npz(os.path.join(path, "{}_csr_emb.npz".format(dataset)))

    labels = idx_features_labels[predict_attr].values

    # Sensitive Attribute
    idx_features_labels['gender'][idx_features_labels['gender']
                                  == 'f'] = 1
    idx_features_labels['gender'][idx_features_labels['gender'] == 'm'] = 0
    idx_features_labels['gender'][idx_features_labels['gender'] == 'none'] = -1
    sens = idx_features_labels[sens_attr].values.astype(int)

    # build graph
    idx = np.array(idx_features_labels["index"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        f'{path}{dataset}_relationship.txt').astype('int')
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    # the original edges has some redundancy

    # extract largested connected components
    _adj=adj.tocsr()
    lcc=largest_connected_components(_adj)
    _A = _adj[lcc][:, lcc]
    features=features[lcc]
    labels=labels[lcc]
    sens=sens[lcc]
    adj=_A.tocoo()

    print(f"{adj.shape[0]} of nodes after selecting LCC")
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # this operation will make the adj symmetric and some redundant edges in
    # original data will have no influence

    # features = normalize(features)
    # so the true number of edges should be (nnz-selfloop)/2 in adj

    # np.where(sum(features[:])==0) ==> 2492,2527,2521,2529 columns with all zeros and columns after 2492 are very sparse
    # eliminate them since normalization over all zeros will generate NAN
    features=features[:,:2491]
    adj = adj + sp.eye(adj.shape[0])
    random.seed(seed)
    features = torch.FloatTensor(np.array(features.todense()))
    # features= torch.FloatTensor(np.random.random((adj.shape[0],1000)))
    labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)


    label_idx = np.where(labels > 0)[0]
    random.shuffle(label_idx)
    # print("num labels:", len(label_idx))

    idx_train = label_idx[:int(train_percent * len(label_idx))]
    idx_val = label_idx[int(train_percent * len(label_idx)):int((train_percent+val_percent) * len(label_idx))]
    idx_test = label_idx[int((train_percent+val_percent) * len(label_idx)):]

    labels[labels ==0] = -1
    labels[labels > 1] = 0  # 1:database 0: DM,IR,ML combined

    sens_idx = set(np.where(sens >= 0)[0])

    idx_test = np.asarray(list(sens_idx & set(idx_test)))

    check_dataset(labels,sens,idx_train,idx_val,idx_test)

    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, labels, idx_train, idx_val, idx_test, sens,idx_sens_train


def load_attacked_graph(
        dataset='./NBAattack100edges/nba',
        adj_fname='nba_adj_after_attack.npz'):
    adj = sp.load_npz(adj_fname)  #
    features = sp.load_npz(dataset + '_feature.npz')  #
    labels = np.load(dataset + "_label.npy")
    idx_train = np.load(dataset + "_train_idx.npy")
    idx_val = np.load(dataset + "_val_idx.npy")
    idx_test = np.load(dataset + "_test_idx.npy")
    sens = np.load(dataset + "_sens.npy")

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    sens = torch.FloatTensor(sens)
    return adj, features, labels, idx_train, idx_val, idx_test, sens


def rand_attack(adj, edge_perturbations):
    print("Begin random attack...")
    perturbations = int(edge_perturbations * (adj.sum() // 2))
    _N = adj.shape[0]
    adj2 = adj.todense()
    for _it in tqdm(range(perturbations), desc="Perturbing graph"):
        attack_nodes = random.choices(np.arange(_N), k=2)
        if adj[attack_nodes[0], attack_nodes[1]] == 1.0:
            adj2[attack_nodes[0], attack_nodes[1]] = 0.0
            adj2[attack_nodes[1], attack_nodes[0]] = 0.0
        else:
            adj2[attack_nodes[0], attack_nodes[1]] = 1.0
            adj2[attack_nodes[1], attack_nodes[0]] = 1.0
    adj = sp.csr_matrix(adj2)
    return adj


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]

    return 2 * (features - min_values).div(max_values - min_values) - 1  # -1~1


def accuracy(output, labels):
    output = output.squeeze()
    preds = (output > 0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def classification_metrics(output, labels):
    output = output.squeeze()
    preds = (output > 0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    acc = correct / len(labels)

    roc = roc_auc_score(
        labels.cpu().numpy(),
        output.detach().cpu().numpy())

    p = precision_score(labels.cpu().numpy(),
                        preds.detach().cpu().numpy())
    r = recall_score(labels.cpu().numpy(),
                     preds.detach().cpu().numpy())  #
    maf1 = f1_score(
        labels.cpu().numpy(),
        preds.detach().cpu().numpy(),
        average='macro')
    mif1 = f1_score(
        labels.cpu().numpy(),
        preds.detach().cpu().numpy(),
        average='micro')

    return acc, roc, p, r, maf1, mif1


def accuracy_softmax(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# %%


# %%
def load_pokec_emb(
        dataset,
        sens_attr,
        predict_attr,
        path="../dataset/pokec/",
        label_number=1000,
        sens_number=500,
        seed=19,
        test_idx=False):
    print('Loading {} dataset from {}'.format(dataset, path))

    graph_embedding = np.genfromtxt(
        os.path.join(path, "{}.embedding".format(dataset)),
        skip_header=1,
        dtype=float
    )
    embedding_df = pd.DataFrame(graph_embedding)
    embedding_df[0] = embedding_df[0].astype(int)
    embedding_df = embedding_df.rename(index=int, columns={0: "user_id"})

    idx_features_labels = pd.read_csv(
        os.path.join(path, "{}.csv".format(dataset)))
    idx_features_labels = pd.merge(
        idx_features_labels,
        embedding_df,
        how="left",
        on="user_id")
    idx_features_labels = idx_features_labels.fillna(0)
    # %%

    header = list(idx_features_labels.columns)
    header.remove("user_id")

    header.remove(sens_attr)
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    # %%
    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        os.path.join(
            path,
            "{}_relationship.txt".format(dataset)),
        dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(seed)
    label_idx = np.where(labels >= 0)[0]
    random.shuffle(label_idx)

    idx_train = label_idx[:min(int(0.5 * len(label_idx)), label_number)]
    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
    if test_idx:
        idx_test = label_idx[label_number:]
    else:
        idx_test = label_idx[int(0.75 * len(label_idx)):]

    sens = idx_features_labels[sens_attr].values

    sens_idx = set(np.where(sens >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(
        1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    # print('building edge relationship complete')
    idx_map = np.array(idx_map)

    return idx_map


def load_credit(
        dataset,
        sens_attr="Age",
        predict_attr="NoDefaultNextMonth",
        path="./dataset/credit/",
        train_percent=0.5,
        val_percent=0.25,
        sens_number=200,
        seed=20):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(
        os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('Single')

    #    # Normalize MaxBillAmountOverLast6Months
    #    idx_features_labels['MaxBillAmountOverLast6Months'] = (idx_features_labels['MaxBillAmountOverLast6Months']-idx_features_labels['MaxBillAmountOverLast6Months'].mean())/idx_features_labels['MaxBillAmountOverLast6Months'].std()
    #
    #    # Normalize MaxPaymentAmountOverLast6Months
    #    idx_features_labels['MaxPaymentAmountOverLast6Months'] = (idx_features_labels['MaxPaymentAmountOverLast6Months'] - idx_features_labels['MaxPaymentAmountOverLast6Months'].mean())/idx_features_labels['MaxPaymentAmountOverLast6Months'].std()
    #
    #    # Normalize MostRecentBillAmount
    #    idx_features_labels['MostRecentBillAmount'] = (idx_features_labels['MostRecentBillAmount']-idx_features_labels['MostRecentBillAmount'].mean())/idx_features_labels['MostRecentBillAmount'].std()
    #
    #    # Normalize MostRecentPaymentAmount
    #    idx_features_labels['MostRecentPaymentAmount'] = (idx_features_labels['MostRecentPaymentAmount']-idx_features_labels['MostRecentPaymentAmount'].mean())/idx_features_labels['MostRecentPaymentAmount'].std()
    #
    #    # Normalize TotalMonthsOverdue
    #    idx_features_labels['TotalMonthsOverdue'] = (idx_features_labels['TotalMonthsOverdue']-idx_features_labels['TotalMonthsOverdue'].mean())/idx_features_labels['TotalMonthsOverdue'].std()

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(
            f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(
            idx_features_labels[header], thresh=0.7)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(seed)
    label_idx = np.where(labels >= 0)[0]
    random.shuffle(label_idx)
    idx_train = label_idx[:int(train_percent * len(label_idx))]
    idx_val = label_idx[int(train_percent * len(label_idx)):int((train_percent + val_percent) * len(label_idx))]
    idx_test = label_idx[int((train_percent + val_percent) * len(label_idx)):]
    # label_idx_0 = np.where(labels == 0)[0]
    # label_idx_1 = np.where(labels == 1)[0]
    # random.shuffle(label_idx_0)
    # random.shuffle(label_idx_1)
    #
    # idx_train = np.append(label_idx_0[:min(int(0.5 *
    #                                            len(label_idx_0)), label_number //
    #                                        2)], label_idx_1[:min(int(0.5 *
    #                                                                  len(label_idx_1)), label_number //
    #                                                              2)])
    # idx_val = np.append(label_idx_0[int(0.5 *
    #                                     len(label_idx_0)):int(0.75 *
    #                                                           len(label_idx_0))], label_idx_1[int(0.5 *
    #                                                                                               len(label_idx_1)):int(
    #     0.75 *
    #     len(label_idx_1))])
    # idx_test = np.append(label_idx_0[int(
    #     0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])


    sens = idx_features_labels[sens_attr].values.astype(int)
    check_dataset(labels, sens, idx_train, idx_val, idx_test)
    sens_idx = set(np.where(sens >= 0)[0])
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train


def load_bail(
        dataset,
        sens_attr="WHITE",
        predict_attr="RECID",
        path="../dataset/bail/",
        train_percent=0.5,
        val_percent=0.25,
        sens_number=200,
        seed=20):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(
        os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)

    # # Normalize School
    # idx_features_labels['SCHOOL'] = 2*(idx_features_labels['SCHOOL']-idx_features_labels['SCHOOL'].min()).div(idx_features_labels['SCHOOL'].max() - idx_features_labels['SCHOOL'].min()) - 1

    # # Normalize RULE
    # idx_features_labels['RULE'] = 2*(idx_features_labels['RULE']-idx_features_labels['RULE'].min()).div(idx_features_labels['RULE'].max() - idx_features_labels['RULE'].min()) - 1

    # # Normalize AGE
    # idx_features_labels['AGE'] = 2*(idx_features_labels['AGE']-idx_features_labels['AGE'].min()).div(idx_features_labels['AGE'].max() - idx_features_labels['AGE'].min()) - 1

    # # Normalize TSERVD
    # idx_features_labels['TSERVD'] = 2*(idx_features_labels['TSERVD']-idx_features_labels['TSERVD'].min()).div(idx_features_labels['TSERVD'].max() - idx_features_labels['TSERVD'].min()) - 1

    # # Normalize FOLLOW
    # idx_features_labels['FOLLOW'] = 2*(idx_features_labels['FOLLOW']-idx_features_labels['FOLLOW'].min()).div(idx_features_labels['FOLLOW'].max() - idx_features_labels['FOLLOW'].min()) - 1

    # # Normalize TIME
    # idx_features_labels['TIME'] = 2*(idx_features_labels['TIME']-idx_features_labels['TIME'].min()).div(idx_features_labels['TIME'].max() - idx_features_labels['TIME'].min()) - 1

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(
            f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(
            idx_features_labels[header], thresh=0.6)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(seed)
    label_idx = np.where(labels >= 0)[0]
    random.shuffle(label_idx)
    idx_train = label_idx[:int(train_percent * len(label_idx))]
    idx_val = label_idx[int(train_percent * len(label_idx)):int((train_percent + val_percent) * len(label_idx))]
    idx_test = label_idx[int((train_percent + val_percent) * len(label_idx)):]
    # label_idx_0 = np.where(labels == 0)[0]
    # label_idx_1 = np.where(labels == 1)[0]
    # random.shuffle(label_idx_0)
    # random.shuffle(label_idx_1)
    # idx_train = np.append(label_idx_0[:min(int(0.5 *
    #                                            len(label_idx_0)), label_number //
    #                                        2)], label_idx_1[:min(int(0.5 *
    #                                                                  len(label_idx_1)), label_number //
    #                                                              2)])
    # idx_val = np.append(label_idx_0[int(0.5 *
    #                                     len(label_idx_0)):int(0.75 *
    #                                                           len(label_idx_0))], label_idx_1[int(0.5 *
    #                                                                                               len(label_idx_1)):int(
    #     0.75 *
    #     len(label_idx_1))])
    # idx_test = np.append(label_idx_0[int(
    #     0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    check_dataset(labels, sens, idx_train, idx_val, idx_test)
    sens_idx = set(np.where(sens >= 0)[0])
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train


def load_german(
        dataset,
        sens_attr="Gender",
        predict_attr="GoodCustomer",
        path="../dataset/german/",
        train_percent=0.5,
        val_percent=0.25,
        sens_number=200,
        seed=20):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(
        os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan')

    # Sensitive Attribute
    idx_features_labels['Gender'][idx_features_labels['Gender']
                                  == 'Female'] = 1
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0

    #    for i in range(idx_features_labels['PurposeOfLoan'].unique().shape[0]):
    #        val = idx_features_labels['PurposeOfLoan'].unique()[i]
    #        idx_features_labels['PurposeOfLoan'][idx_features_labels['PurposeOfLoan'] == val] = i

    #    # Normalize LoanAmount
    #    idx_features_labels['LoanAmount'] = 2*(idx_features_labels['LoanAmount']-idx_features_labels['LoanAmount'].min()).div(idx_features_labels['LoanAmount'].max() - idx_features_labels['LoanAmount'].min()) - 1
    #
    #    # Normalize Age
    #    idx_features_labels['Age'] = 2*(idx_features_labels['Age']-idx_features_labels['Age'].min()).div(idx_features_labels['Age'].max() - idx_features_labels['Age'].min()) - 1
    #
    #    # Normalize LoanDuration
    #    idx_features_labels['LoanDuration'] = 2*(idx_features_labels['LoanDuration']-idx_features_labels['LoanDuration'].min()).div(idx_features_labels['LoanDuration'].max() - idx_features_labels['LoanDuration'].min()) - 1
    #
    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(
            f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(
            idx_features_labels[header], thresh=0.8)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels == -1] = 0

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(seed)
    label_idx = np.where(labels >= 0)[0]
    random.shuffle(label_idx)
    idx_train = label_idx[:int(train_percent * len(label_idx))]
    idx_val = label_idx[int(train_percent * len(label_idx)):int((train_percent + val_percent) * len(label_idx))]
    idx_test = label_idx[int((train_percent + val_percent) * len(label_idx)):]
    # label_idx_0 = np.where(labels == 0)[0]
    # label_idx_1 = np.where(labels == 1)[0]
    # random.shuffle(label_idx_0)
    # random.shuffle(label_idx_1)
    #
    # idx_train = np.append(label_idx_0[:min(int(0.5 *
    #                                            len(label_idx_0)), label_number //
    #                                        2)], label_idx_1[:min(int(0.5 *
    #                                                                  len(label_idx_1)), label_number //
    #                                                              2)])
    # idx_val = np.append(label_idx_0[int(0.5 *
    #                                     len(label_idx_0)):int(0.75 *
    #                                                           len(label_idx_0))], label_idx_1[int(0.5 *
    #                                                                                               len(label_idx_1)):int(
    #     0.75 *
    #     len(label_idx_1))])
    # idx_test = np.append(label_idx_0[int(
    #     0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    check_dataset(labels, sens, idx_train, idx_val, idx_test)
    sens_idx = set(np.where(sens >= 0)[0])
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train