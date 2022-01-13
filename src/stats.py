import argparse
from utils import *
import networkx as nx
from attack.attack import attack
import pandas as pd
from community.community_louvain import best_partition
from synth.evaluate import get_latent_and_prediction

import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def homophily(G, prop, level='macro'):
    """

    :param adj:
    :param prop:
    :param level: 'macro' or 'micro'
    :return:
    """
    if level == 'macro':
        return len([edge for edge in G.edges() if prop[edge[0]]!=-1 and prop[edge[0]] == prop[edge[1]]]) / len([edge for edge in G.edges() if prop[edge[0]]!=-1 and prop[edge[1]]!=-1])
    elif level == 'micro':
        local_homophily_pairs = [local_homophily(G, node, prop) for node in G.nodes() if prop[node]!=-1]
        nom = sum([x[0] for x in local_homophily_pairs])
        den = sum([x[1] for x in local_homophily_pairs])
        return nom/den
    else:
        print(f"Unrecognized level {level}, use \'macro\' or \'micro\'")


def local_homophily(G, node, prop):
    neighbors = [u for u in G.neighbors(node) if prop[u]!=-1]
    d = len(neighbors)
    if d == 0: return 0,0
    return len([neighbor for neighbor in neighbors if prop[node]!=-1 and prop[node] == prop[neighbor]]) / d,1


def calc_uncertainty(df_community):
    
    def agg(x):
        return len(x.unique())

    communities = df_community.community.unique()
    labels = df_community.label.unique()

    mtx = df_community.pivot_table(index='community', columns='label',values='node',aggfunc=agg).fillna(0) / len(df_community)
    
    def Pmarg(c):
        return len(df_community[df_community.community == c]) / len(df_community)
    
    def Pcond(l,c):
        return mtx.loc[c,l]/Pmarg(c)
    
    H = 0
    for c in communities:
        h = 0
        for l in labels:
            if Pcond(l,c) == 0:
                continue
            h += Pcond(l,c) * np.log2(1./Pcond(l,c))
        H += h * Pmarg(c)
    
    def Pl(l):
        return len(df_community[df_community.label == l]) / len(df_community)
    
    Hl = 0
    for l in labels:
        if Pl(l) == 0:
            continue
        Hl += Pl(l) * np.log2(1./Pl(l))
    
    IG = Hl-H
    return IG/Hl


def community_correlation(G, prop, seeds=[42,0,1,2,100]):
    ret = []
    for seed in seeds:
        community_mapping = best_partition(G,random_state=seed)
        df_community = pd.DataFrame([[u,community_mapping[u],prop[u].item()] for u in G.nodes() if prop[u]!=-1],columns=['node','community','label'])
        ret.append(calc_uncertainty(df_community))
    return ret, seeds


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
parser.add_argument('--sensitive', type=str, default='region',
                    choices=['gender', 'region'],
                    help='Sensitive attribute of Pokec.')
parser.add_argument(
    '--stats_type',
    type=str,
    default='latent',
    choices=['homophily', 'community', 'latent', 'density'])
parser.add_argument('--ptb_rate', type=float, nargs='+', default=[0.05,0.1,0.15,0.2,0.25,0.3],
                    help="Attack perturbation rate [0-1]")
parser.add_argument('--seed', type=int, nargs='+', default=[42, 0, 1, 2, 100],
                    help="Attack seed")
parser.add_argument('--train_percent', type=float, default=0.5,
                    help='Percentage of labeled data as train set.')
parser.add_argument('--val_percent', type=float, default=0.25,
                    help='Percentage of labeled data as validation set.')
args = parser.parse_known_args()[0]

df_stats = pd.DataFrame()
for ptb_rate in (args.ptb_rate if args.attack_type != 'none' else [0]):
    for seed in (args.seed if args.attack_type != 'none' else [0]):
        print("===== Computing stats =====")
        print(args.dataset)
        print(f'seed={seed} ptb_rate={ptb_rate}')
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train, dataset, sens_attr, sens_number = \
            load_dataset(args, seed)

        if args.attack_type != 'none':
            adj = attack(args,args.attack_type, ptb_rate, adj, features, labels, sens, idx_train, idx_val, idx_test,
                         seed, dataset, sens_attr)

        G = nx.from_scipy_sparse_matrix(adj)
        if args.stats_type=='homophily': 
            label_homophily_macro = homophily(G, labels, 'macro')
            print(label_homophily_macro)
            label_homophily_micro = homophily(G, labels, 'micro')
            print(label_homophily_micro)
            sens_homophily_macro = homophily(G, sens, 'macro')
            print(sens_homophily_macro)
            sens_homophily_micro = homophily(G, sens, 'micro')
            print(sens_homophily_micro)
            row = {"ptb_rate": ptb_rate, "seed": seed,
                       "label_homophily_macro": label_homophily_macro, "label_homophily_micro": label_homophily_micro,
                       "sens_homophily_macro": sens_homophily_macro, "sens_homophily_micro": sens_homophily_micro}
        elif args.stats_type == 'community':
            community_label_correlations,seeds = community_correlation(G, labels) # assume the same seeds parameter for the next call
            community_sens_correlations,seeds = community_correlation(G, sens)
            row = [{"ptb_rate": ptb_rate, "seed": seed,
                   "community_label_correlation":community_label_correlation,
                   "community_sens_correlation":community_sens_correlation,
                   "seed_comm":seed_comm,
                   } for community_label_correlation,community_sens_correlation,seed_comm in zip(community_label_correlations,community_sens_correlations,seeds)]
        elif args.stats_type=='latent':
            x,y = get_latent_and_prediction(adj,features,labels,idx_train,device)
            row = {"ptb_rate": ptb_rate, "seed": seed,
            'hidden_representations':"[" + ','.join(['['+','.join([str(b) for b in a])+']' for a in x.detach().to('cpu').numpy()]) + "]",
            'label':f"[{','.join([str(a) for a in labels.detach().to('cpu').numpy()])}]",
            'sens':f"[{','.join([str(a) for a in sens.detach().to('cpu').numpy()])}]"}
        elif args.stats_type=='density':
            pass
        row['attack'] = args.attack_type
        df_stats = df_stats.append(row, ignore_index=True)

fname = f'../results/stats-{args.stats_type}-' + str(args.dataset) + ('-'+sens_attr if 'pokec' in args.dataset else '') + \
        '-' + str(args.attack_type) + '.csv'

df_stats.to_csv(fname, index=False)