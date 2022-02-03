import random
import numpy as np
import math
import scipy.sparse as sp
import networkx as nx
from deeprobust.graph.global_attack import BaseAttack, Metattack
from deeprobust.graph.global_attack import Random

from deeprobust.graph.defense import GCN
from deeprobust.graph import utils
import torch
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm
from attack.utils import *


def get_strategy_nodeset(direction, strategy):
    y, s = bool(int(direction[1])), bool(int(direction[3]))
    y_ = y if strategy[0] == 'E' else not y
    s_ = s if strategy[1] == 'E' else not s
    return str(int(y)) + str(int(s)), str(int(y_)) + str(int(s_))


class Fair_Attack(BaseAttack):

    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        super(Fair_Attack, self).__init__(model, nnodes, attack_structure=attack_structure,
                                          attack_features=attack_features, device=device)

        self.sens_surrogate = False
        assert not self.attack_features, 'Fair_Attack does NOT support attacking features'

    def attack(self, ori_adj, features, y, s, idx_train, n_perturbations, direction, strategy, deg, deg_direction,
               dataset, idx_sens_train, **kwargs):
        """
        Attempts to increase the statistical parity by linking nodes in [y1s1 or y1s0] to nodes in another
        group with [D]ifferent/[E]qual label(y)|sens(s), which corresponds to four types of strategies

        :param y:
        :param ori_adj:
        :param s:
        :param n_perturbations:
        :param direction: FairAttack direction
        :param strategy: FairAttack strategy indicating [D]ifferent/[E]qual label(y)|sens(s)
        :param kwargs:
        :return:
        """
        modified_adj = ori_adj.tolil()

        print("Training surrogate to get labels")
        y_s = test_surrogate(ori_adj, features, y, idx_train, dataset, device=self.device)  # for german use a different surrogate
        # label calibration
        y_s[idx_train]=y[idx_train] #label calibrate--in test
        y=y_s
        
        if self.sens_surrogate:
            print("Training surrogate to get sens")
            s_s = test_surrogate(ori_adj, features, s, idx_sens_train, dataset, device=self.device)  # for german use a different surrogate
            # sens calibration
            s_s[idx_sens_train]=y[idx_sens_train] #label calibrate--in test
            s=s_s

        # remember that we might have s[i]=-1 when the sensitive attribute is not available
        y1 = y == 1
        s1 = s == 1
        s0 = s == 0
        y0 = y == 0

        y0s0 = y0 & s0
        y1s0 = y1 & s0
        y0s1 = y0 & s1
        y1s1 = y1 & s1

        all = sum(y0s0 + y1s0 + y0s1 + y1s1)

        print('initial distribution:')
        print(f'{sum(y1s1) / all:.2f}|{sum(y1s0) / all:.2f}\n{sum(y0s1) / all:.2f}|{sum(y0s0) / all:.2f}')
        print('-----------------------')

        G = nx.from_scipy_sparse_matrix(ori_adj)

        nodes_y0s0 = [u for u in G.nodes() if y0s0[u]]
        nodes_y1s0 = [u for u in G.nodes() if y1s0[u]]
        nodes_y0s1 = [u for u in G.nodes() if y0s1[u]]
        nodes_y1s1 = [u for u in G.nodes() if y1s1[u]]

        n_map = {'00': nodes_y0s0, '10': nodes_y1s0, '01': nodes_y0s1, '11': nodes_y1s1}

        # nodes_direction = nodes_y1s1 if direction == 'y1s1' else nodes_y1s0
        nd,ns=get_strategy_nodeset(direction, strategy)
        nodes_direction, nodes_strategy = n_map[nd],n_map[ns]

        if deg == 0:  # don't consider degree
            assert (deg_direction == 'null')
            subject = list(np.random.choice(nodes_direction, n_perturbations))
            influencer = list(np.random.choice(nodes_strategy, n_perturbations))

            assert (len(subject) == len(influencer))
            # remove duplicate in sampling
            tu = [(subject[i], influencer[i]) for i in range(len(subject))]
            ts = set(tu)
            print(f"duplicate samples:{len(tu)-len(ts)}")
            ts=list(ts)
            subject = [ts[i][0] for i in range(len(ts))]
            influencer = [ts[i][1] for i in range(len(ts))]

            dup_edges = modified_adj[subject, influencer].nnz+(len(tu)-len(ts)) # existing edges+duplicate samples
            print(f'{dup_edges} edges already exist')
            if dup_edges > 0:
                print(f"selecting {dup_edges} more edges..")
                i = 0
                while i < dup_edges:
                    n1 = np.random.choice(nodes_direction, 1)
                    n2 = np.random.choice(nodes_strategy, 1)
                    if n1[0] != n2[0] and not G.has_edge(n1[0], n2[0]):
                        # G.add_edge(n1[0], n2[0])
                        subject.append(n1[0])
                        influencer.append(n2[0])
                        i += 1
                print("Finish all perturbations")
            modified_adj[subject, influencer] = 1
            modified_adj[influencer, subject] = 1

        else:  # considering degree difference in two directions
            i = 0
            while i < n_perturbations:
                n1 = np.random.choice(nodes_direction, 1)
                n2 = np.random.choice(nodes_strategy, 1)
                if not G.has_edge(n1[0], n2[0]):
                    if deg_direction == 'hl' and G.degree[n1[0]] > deg * G.degree[n2[0]]:
                        G.add_edge(n1[0], n2[0])
                        i += 1
                    elif deg_direction == 'lh' and G.degree[n2[0]] > deg * G.degree[n1[0]]:
                        G.add_edge(n1[0], n2[0])
                        i += 1
            print("Finish all perturbations considering degree")
            modified_adj = nx.adjacency_matrix(G)

        self.check_adj(modified_adj)
        self.modified_adj = modified_adj
