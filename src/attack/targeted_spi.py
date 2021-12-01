import random
import numpy as np
import math
import scipy.sparse as sp
import networkx as nx
from deeprobust.graph.global_attack import BaseAttack, Metattack
from deeprobust.graph.global_attack import Random
from deeprobust.graph.targeted_attack import Nettack, RND

from deeprobust.graph import utils
import torch
from torch import optim
from torch.nn import functional as F
from attack.utils import *
from tqdm import tqdm


class BaseTargetedSPI(BaseAttack):

    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        super(BaseTargetedSPI, self).__init__(model, nnodes, attack_structure=attack_structure,
                                              attack_features=attack_features, device=device)

        assert not self.attack_features, 'BaseTargetedSPI does NOT support attacking features'

    def attack(self, ori_adj, features, labels, sens, idx_train, n_perturbations, **kwargs):
        pass

    def select_targets(self, ori_adj, features, labels, sens, idx_train, surrogate):
        y = surrogate.predict(features, ori_adj)
        y = y.max(1)[0]

        y0 = y < .5
        y1 = y >= .5
        s0 = sens == 0
        s1 = sens == 1

        y1s0 = y1 & s0
        y0s1 = y0 & s1

        initial_targets = y1s0 | y0s1

        ordered_targets = [u for u in sorted(range(len(y)), key=lambda k: y[k]) if initial_targets[u]]
        return ordered_targets


class TargetedSPI(BaseTargetedSPI):

    def attack(self, ori_adj, features, labels, sens, idx_train, n_perturbations, n_perturbation_per_node=16, **kwargs):

        modified_adj = ori_adj.tolil()
        if n_perturbations % n_perturbation_per_node:
            print(f'{n_perturbations % n_perturbation_per_node} perturbations will be wasted')

        for i in tqdm(range(n_perturbations // n_perturbation_per_node)):
            if i % 100 == 0:
                print(f'training surrogate on {self.device}')
                self.surrogate = fit_surrogate(ori_adj, features, labels, idx_train, self.device)
                targets = self.select_targets(modified_adj, features, labels, sens, idx_train, self.surrogate)

            modified_adj = self.single_node_attack(features, modified_adj, labels, targets[i], idx_train, n_perturbation_per_node)

        self.check_adj(modified_adj)
        self.modified_adj = modified_adj
        return modified_adj

    def single_node_attack(self, features, adj, labels, target_node, idx_train, n_perturbations):
        pass


class RandomSPI(TargetedSPI):

    def single_node_attack(self, features, adj, labels, target_node, idx_train, n_perturbations):
        model = RND()
        model.attack(adj, labels, idx_train, target_node, n_perturbations)
        return model.modified_adj


class NettackSPI(TargetedSPI):

    def single_node_attack(self, features, adj, labels, target_node, idx_train, n_perturbations):
        model = Nettack(self.surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=True, device=self.device)
        model.attack(adj, labels, idx_train, target_node, n_perturbations)
        return model.modified_adj

