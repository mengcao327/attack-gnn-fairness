from deeprobust.graph.global_attack import DICE, Random, Metattack
from attack.sacide import SACIDE
from structack.structack import build_custom
import structack.node_selection as ns
import structack.node_connection as nc
from deeprobust.graph.utils import normalize_adj, sparse_mx_to_torch_sparse_tensor, preprocess, to_scipy
from deeprobust.graph.defense import GCN
import scipy.sparse as sp
import numpy as np
import torch


def build_random(adj=None, features=None, labels=None, idx_train=None, device=None):
    return Random()


def build_dice(adj=None, features=None, labels=None, idx_train=None, device=None):
    return DICE()


def build_sacide(adj=None, features=None, labels=None, idx_train=None, device=None):
    return SACIDE()


def attack_random(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled, sens):
    model.attack(adj, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)


def attack_dice(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled, sens):
    model.attack(adj, labels, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)


def attack_sacide(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled, sens):
    model.attack(adj, sens, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)


def postprocess_adj(adj):
    adj = normalize_adj(adj)
    #     adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


def attack_structack(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(adj, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)


def apply_perturbation(model_builder, attack, adj, features, labels, sens,
                       idx_train, idx_val, idx_test, ptb_rate=0.05,
                       cuda=False, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if cuda else "cpu")

    idx_unlabeled = np.union1d(idx_val, idx_test)

    n_perturbations = int(ptb_rate * (adj.sum() // 2))
    print(f'n_perturbations = {n_perturbations}')

    if model_builder == build_metattack:
        adj, features, labels = preprocess(adj, sp.coo_matrix(features.cpu().numpy()), labels.cpu().numpy(),
                                           preprocess_adj=False)

    # build the model
    model = model_builder(adj, features, labels, idx_train, device)

    # perform the attack
    modified_adj = attack(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled, sens)
    return modified_adj


def build_metattack(adj=None, features=None, labels=None, idx_train=None, device=None):
    lambda_ = 0

    # Setup Surrogate Model
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16,
                    dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)
    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train)
    print(f'{torch.cuda.device_count()} GPUs available')
    print('built surrogate')
    model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
                      attack_structure=True, attack_features=False, device=device, lambda_=lambda_, lr=0.005)
    print('built model')
    model = model.to(device)
    print('to device')
    return model


def attack_metattack(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled, sens):
    model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations, ll_constraint=False)
    return to_scipy(model.modified_adj)


def attack(attack_name, ptb_rate, adj, features, labels, sens, idx_train, idx_val, idx_test, seed, dataset_name=None):
    """
    builds the attack, applies the perturbation
    :param attack_name: random, dice, metattack
    :param ptb_rate: [0,1]
    :param adj: scipy_sparse
    :param features:
    :param labels:
    :param sens:
    :param idx_train:
    :param idx_val:
    :param idx_test:
    :param seed:
    :param dataset_name: required only for structack
    :return: perturbed graph (scipy_sparse)
    """
    print(f'Applying {attack_name} attack to input graph')
    builds = {'random': build_random, 'dice': build_dice, 'metattack': build_metattack, 'sacide': build_sacide}
    attacks = {'random': attack_random, 'dice': attack_dice, 'metattack': attack_metattack, 'sacide': attack_sacide}
    baseline_attacks = list(builds.keys())

    if attack_name in baseline_attacks:
        modified_adj = apply_perturbation(builds[attack_name], attacks[attack_name], adj, features, labels, sens,
                                          idx_train,
                                          idx_val, idx_test, ptb_rate=ptb_rate, seed=seed)
        print(f'Attack finished, returning perturbed graph')
        return modified_adj
    elif 'structack' in attack_name:
        selections = {'dg': ns.get_nodes_with_lowest_degree,
                      'pr': ns.get_nodes_with_lowest_pagerank,
                      'ev': ns.get_nodes_with_lowest_eigenvector_centrality,
                      'bt': ns.get_nodes_with_lowest_betweenness_centrality,
                      'cl': ns.get_nodes_with_lowest_closeness_centrality}
        connections = {'katz': nc.katz_hungarian_connection,
                       'comm': nc.community_hungarian_connection,
                       'dist': nc.distance_hungarian_connection}
        _, selection, connection = attack_name.split('_')
        modified_adj = attack_structack(build_custom(selections[selection], connections[connection], dataset_name), adj,
                                        features, labels, int(ptb_rate * (adj.sum() // 2)), idx_train,
                                        np.union1d(idx_val, idx_test))
        print(f'Attack finished, returning perturbed graph')
        return modified_adj
    else:
        print('attack not recognized')
        return None
