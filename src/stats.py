import argparse
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--attack_type', type=str, default='none',
                    # choices=['none', 'random', 'dice', 'metattack', 'sacide', 'structack_dg_comm', 'structack_pr_katz'],
                    help='Adversarial attack type.')
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
parser.add_argument('--seed', type=int, nargs='+', default=[42],
                    help="Attack seed")

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
    neighbors = G.neighbors(node)
    return len([neighbor for neighbor in neighbors if prop[node] == prop[neighbor]]) / len(neighbors)


