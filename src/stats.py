import argparse
from utils import *
import networkx as nx
from attack.attack import attack
import pandas as pd


def homophily(G, prop, level='macro'):
    """

    :param adj:
    :param prop:
    :param level: 'macro' or 'micro'
    :return:
    """
    if level == 'macro':
        return len([edge for edge in G.edges() if prop[edge[0]] == prop[edge[1]]]) / G.number_of_edges()
    elif level == 'micro':
        return sum([local_homophily(G, node, prop) for node in G.nodes()])
    else:
        print(f"Unrecognized level {level}, use \'macro\' or \'micro\'")


def local_homophily(G, node, prop):
    neighbors = list(G.neighbors(node))
    d = len(neighbors)
    if d == 0: return 0
    return len([neighbor for neighbor in neighbors if prop[node] == prop[neighbor]]) / d / G.number_of_nodes()


parser = argparse.ArgumentParser()

parser.add_argument('--attack_type', type=str, default='dice',
                    # choices=['none', 'random', 'dice', 'metattack', 'sacide', 'structack_dg_comm', 'structack_pr_katz'],
                    help='Adversarial attack type.')
parser.add_argument('--sens_number', type=int, default=200,  # TODO why do we need this? We shouldn't
                    help="the number of sensitive attributes")
parser.add_argument(
    '--dataset',
    type=str,
    default='nba',
    choices=[
        'pokec_z',
        'pokec_n',
        'nba',
        'credit',
        'german',
        'bail',
        'dblp'])
parser.add_argument('--ptb_rate', type=float, nargs='+', default=[0.05],
                    help="Attack perturbation rate [0-1]")
parser.add_argument('--seed', type=int, nargs='+', default=[42, 0, 1],
                    help="Attack seed")
parser.add_argument('--train_percent', type=float, default=0.5,
                    help='Percentage of labeled data as train set.')
parser.add_argument('--val_percent', type=float, default=0.25,
                    help='Percentage of labeled data as validation set.')
args = parser.parse_known_args()[0]

df_stats = pd.DataFrame()
for ptb_rate in (args.ptb_rate if args.attack_type != 'none' else [0]):
    for seed in args.seed:
        print(args.dataset)
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train, dataset, sens_attr = \
            load_dataset(args, seed)

        if args.attack_type != 'none':
            adj = attack(args.attack_type, ptb_rate, adj, features, labels, sens, idx_train, idx_val, idx_test,
                         seed, dataset)

        G = nx.from_scipy_sparse_matrix(adj)

        label_homophily_macro = homophily(G, labels, 'macro')
        label_homophily_micro = homophily(G, labels, 'micro')
        sens_homophily_macro = homophily(G, sens, 'macro')
        sens_homophily_micro = homophily(G, sens, 'micro')

        row = {"ptb_rate": ptb_rate, "seed": seed,
               "label_homophily_macro": label_homophily_macro, "label_homophily_micro": label_homophily_micro,
               "sens_homophily_macro": sens_homophily_macro, "sens_homophily_micro": sens_homophily_micro}
        df_stats = df_stats.append(row, ignore_index=True)

fname = '../results/stats-' + str(args.dataset) + \
        '-' + str(args.attack_type) + '.csv'

df_stats.to_csv(fname, index=False)