import random
import numpy as np
import scipy.sparse as sp
import networkx as nx
from deeprobust.graph.global_attack import BaseAttack


class SPI_heuristic(BaseAttack):

    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        super(SPI_heuristic, self).__init__(model, nnodes, attack_structure=attack_structure,
                                     attack_features=attack_features, device=device)

        assert not self.attack_features, 'SPI_heuristic does NOT support attacking features'

    def attack(self, ori_adj, y, s, n_perturbations, **kwargs):
        """
        Attempts to increase the statistical parity by linking single [subject] low-degree nodes with (y=0,
        s=0) to multiple [influencer] low-degree nodes with (y=1,s=0) and vice versa for (y=0,s=1) and (y=1,s=1)

        The aim is to move more nodes with s=0 to have ^y=0 and more nodes with s=1 to ^y=1

        :param y:
        :param ori_adj:
        :param s:
        :param n_perturbations:
        :param kwargs:
        :return:
        """
        modified_adj = ori_adj.tolil()

        # remember that we might have s[i]=-1 when the sensitive attribute is not available
        y1 = y==1
        s1 = s==1
        s0 = s==0
        y0 = y==0

        y0s0 = y0 & s0
        y1s0 = y1 & s0
        y0s1 = y0 & s1
        y1s1 = y1 & s1

        all = sum(y0s0+y1s0+y0s1+y1s1)

        print('initial distribution:')
        print(f'{sum(y1s1)/all:.2f}|{sum(y1s0)/all:.2f}\n{sum(y0s1)/all:.2f}|{sum(y0s0)/all:.2f}')
        print('-----------------------')

        G = nx.from_scipy_sparse_matrix(ori_adj)

        subject_s0 = sorted(G.nodes[y0s0], key=lambda x: G.degree(x)[1])
        influencer_s0 = sorted(G.nodes[y1s0], key=lambda x: G.degree(x)[1])

        subject_s1 = sorted(G.nodes[y1s1], key=lambda x: G.degree(x)[1])
        influencer_s1 = sorted(G.nodes[y0s1], key=lambda x: G.degree(x)[1])

        # equally attack both sets
        n_perturbations_s0 = n_perturbations // 2
        n_perturbations_s1 = n_perturbations - n_perturbations_s0

        # For now, let's choose the ratio of 1 subject node to 2 influencer nodes since we choose both of a low degree
        subject = subject_s0[:n_perturbations_s0//2] + subject_s0[:n_perturbations_s0 - n_perturbations_s0//2] + \
                  subject_s1[:n_perturbations_s1//2] + subject_s1[:n_perturbations_s1 - n_perturbations_s1//2]

        influencer = influencer_s0[:n_perturbations_s0] + influencer_s1[:n_perturbations_s1]

        assert(len(subject) == len(influencer))

        print(f'{modified_adj[subject, influencer].nnz} edges already exist')

        modified_adj[subject, influencer] = 1

        self.check_adj(modified_adj)
        self.modified_adj = modified_adj







