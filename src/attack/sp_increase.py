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
from attack.utils import *


EDGE_BATCH = 1000
FIT_FREQ = 10
# (1-TOLERANCE) defines how much improvement we wish to have on each iteration
# if TOLERANCE is .7, it means our best hope is a 30% improvement
TOLERANCE = .7
IMPROVEMENT_MARGIN = .001  # TODO make paramters of the class


class BaseMetropolisHastingSPI(BaseAttack):

    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        super(BaseMetropolisHastingSPI, self).__init__(model, nnodes, attack_structure=attack_structure,
                                                       attack_features=attack_features, device=device)

        assert not self.attack_features, 'IterativePerturbationSPI does NOT support attacking features'

    def attack(self, ori_adj, features, labels, s, idx_train, n_perturbations, **kwargs):
        """
        TODO: rewires links of subject nodes from unwanted y values to wanted y values
        :param labels:
        :param ori_adj:
        :param s:
        :param n_perturbations:
        :param kwargs:
        :return:
        """
        modified_adj = ori_adj.tolil()
        n = len(labels)

        modified_adj = ori_adj
        n_remaining = n_perturbations

        min_iter = math.ceil(n_perturbations / EDGE_BATCH)
        print(f'{n_perturbations} perturbations in {min_iter} turns')
        max_iter = min_iter * 5
        accepted_moves = 0
        iter = 0
        while n_remaining > 0:

            if iter >= max_iter:
                print(f'Reached max iterations {iter}')
                break
            # set the number of perturbations in this turn
            B = min((n_remaining, EDGE_BATCH))
            print(f'iter {iter}, {B} perturbations')

            # Train a surrogate on the current graph
            if iter % FIT_FREQ == 0:
                print('Fitting surrogate')
                self.surrogate = fit_surrogate(modified_adj, features, labels, idx_train, device=self.device)
            iter += 1

            print('Obtaining predictions')
            y = self.surrogate.predict(features, modified_adj)
            y = y.max(1)[1]
            dSP = compute_statistical_parity(s.to(self.device), y.to(self.device))
            y = torch.tensor(y).type_as(labels)
            print(f'Curr dSP = {dSP:.4f}')

            # propose B perturbations
            print('Building perturbed graph proposal')
            proposed_adj = self.propose_perturbation(modified_adj, features, y, s, idx_train, B)

            # compute the statistical parity
            print('Computing statistical parity')
            proposed_sp = compute_statistical_parity(s.to(self.device), self.surrogate.predict(features, proposed_adj).max(1)[1].to(self.device))
            print(f'Proposed dSP = {proposed_sp:.4f}')

            min_accept = dSP * (1 - IMPROVEMENT_MARGIN)
            max_accept = dSP * (1 + IMPROVEMENT_MARGIN)
            linear_accept = (proposed_sp - dSP) / (max_accept - min_accept)
            print(f'Linear {linear_accept:.3f}')
            acceptance = F.sigmoid(torch.Tensor([linear_accept])).item()
            print(f'Acceptance {acceptance:.3f}')
            # flip a coin!
            if acceptance > random.uniform(0, 1):
                # update the modified_adj and continue
                modified_adj = proposed_adj
                n_remaining -= B
                print('Accepted')
                accepted_moves += 1
            else:
                print('Rejected')
        print(f'Attack finished. Accepted {accepted_moves} moves.')
        self.check_adj(modified_adj)
        self.modified_adj = modified_adj

    def propose_perturbation(self, modified_adj, features, y, s, idx_train, B):
        pass


class RandomMetropolisHastingSPI(BaseMetropolisHastingSPI):

    def propose_perturbation(self, adj, features, y, s, idx_train, B):
        s = s.to(self.device)
        y = y.to(self.device)
        modified_adj = adj.copy()

        n_remove = B // 2
        n_add = B - n_remove

        model = Random()
        model.attack(modified_adj, n_remove, type='remove')
        modified_adj = model.modified_adj
        model.attack(modified_adj, n_add, type='add')
        modified_adj = model.modified_adj

        self.check_adj(modified_adj)
        return modified_adj


class RewireMetropolisHastingSPI(BaseMetropolisHastingSPI):

    def propose_perturbation(self, adj, features, y, s, idx_train, B):
        s = s.to(self.device)
        y = y.to(self.device)

        modified_adj = adj.copy()

        # remember that we might have s[i]=-1 when the sensitive attribute is not available
        y1 = y == 1
        s1 = s == 1
        s0 = s == 0
        y0 = y == 0

        y0s0 = y0 & s0
        y1s0 = y1 & s0
        y0s1 = y0 & s1
        y1s1 = y1 & s1

        G = nx.from_scipy_sparse_matrix(modified_adj)

        nodes_y0s0 = [u for u in G.nodes() if y0s0[u]]
        nodes_y1s0 = [u for u in G.nodes() if y1s0[u]]
        nodes_y0s1 = [u for u in G.nodes() if y0s1[u]]
        nodes_y1s1 = [u for u in G.nodes() if y1s1[u]]

        n_remove = B // 2
        n_remaining = B - n_remove

        removable_edges = [[e[0], e[1]] for e in G.edges() if y[e[0]] == y[e[1]]]
        G_rem = nx.from_edgelist(removable_edges)

        nodes_y1s0_cand = [u for u in nodes_y1s0 if
                           G_rem.degree(u) != 0]  # nodes from nodes_y1s0 that do have edges in G_rem
        nodes_y0s1_cand = [u for u in nodes_y0s1 if
                           G_rem.degree(u) != 0]  # nodes from nodes_y0s1 that do have edges in G_rem

        # equally attack both sets
        n_perturbations_s0 = n_remaining // 2
        n_perturbations_s1 = n_remaining - n_perturbations_s0

        subject_s0 = list(np.random.choice(nodes_y1s0_cand, n_perturbations_s0))
        subject_s1 = list(np.random.choice(nodes_y0s1_cand, n_perturbations_s1))

        subject = subject_s0 + subject_s1

        influencer_s0 = list(np.random.choice(nodes_y0s0, n_perturbations_s0))
        influencer_s1 = list(np.random.choice(nodes_y1s1, n_perturbations_s1))
        influencer = influencer_s0 + influencer_s1

        unwanted = [np.random.choice(list(G_rem.neighbors(u))) for u in subject]

        assert (len(subject) == len(influencer))

        modified_adj[subject, influencer] = 1
        modified_adj[influencer, subject] = 1

        modified_adj[subject, unwanted] = 0
        modified_adj[unwanted, subject] = 0

        self.check_adj(modified_adj)
        return modified_adj


class RewireSPI(BaseAttack):

    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        super(RewireSPI, self).__init__(model, nnodes, attack_structure=attack_structure,
                                        attack_features=attack_features, device=device)

        assert not self.attack_features, 'RewireSPI does NOT support attacking features'

    def attack(self, ori_adj, features, y, s, idx_train, n_perturbations, **kwargs):
        """
        TODO: rewires links of subject nodes from unwanted y values to wanted y values
        :param y:
        :param ori_adj:
        :param s:
        :param n_perturbations:
        :param kwargs:
        :return:
        """
        modified_adj = ori_adj.tolil()

        y = test_surrogate(ori_adj, features, y, s, idx_train, device=self.device)

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
        print(f'{sum(y0s0) / all:.2f}|{sum(y0s1) / all:.2f}\n{sum(y1s0) / all:.2f}|{sum(y1s1) / all:.2f}')
        print('-----------------------')

        G = nx.from_scipy_sparse_matrix(ori_adj)

        nodes_y0s0 = [u for u in G.nodes() if y0s0[u]]
        nodes_y1s0 = [u for u in G.nodes() if y1s0[u]]
        nodes_y0s1 = [u for u in G.nodes() if y0s1[u]]
        nodes_y1s1 = [u for u in G.nodes() if y1s1[u]]

        def avg_deg(nodes):
            return np.mean(np.array([G.degree(u) for u in nodes]))

        # print('Average degree')
        # print(f'{avg_deg(nodes_y0s0):.2f}|{avg_deg(nodes_y0s1):.2f}\n{avg_deg(nodes_y1s0):.2f}|'
        #       f'{avg_deg(nodes_y1s1):.2f}')

        def avg_hom(nodes):
            return np.mean(np.array(
                [len([v for v in G.neighbors(u) if y[v] == y[u]]) / len([v for v in G.neighbors(u) if y[v] != -1]) for u
                 in nodes]))

        # print('Average homophily')
        # print(f'{avg_hom(nodes_y0s0):.2f}|{avg_hom(nodes_y0s1):.2f}\n{avg_hom(nodes_y1s0):.2f}|'
        #       f'{avg_hom(nodes_y1s1):.2f}')

        n_remove = n_perturbations // 2
        n_remaining = n_perturbations - n_remove

        removable_edges = [[e[0], e[1]] for e in G.edges() if y[e[0]] == y[e[1]]]
        G_rem = nx.from_edgelist(removable_edges)

        nodes_y1s1_cand = [u for u in nodes_y1s1 if
                           G_rem.degree(u) != 0]  # nodes from nodes_y1s0 that do have edges in G_rem
        nodes_y0s0_cand = [u for u in nodes_y0s0 if
                           G_rem.degree(u) != 0]  # nodes from nodes_y0s1 that do have edges in G_rem

        # equally attack both sets
        n_perturbations_s0 = n_remaining // 2
        n_perturbations_s1 = n_remaining - n_perturbations_s0

        subject_s0 = list(np.random.choice(nodes_y1s1_cand, n_perturbations_s0))
        subject_s1 = list(np.random.choice(nodes_y0s0_cand, n_perturbations_s1))

        subject = subject_s0 + subject_s1

        influencer_s0 = list(np.random.choice(nodes_y0s1, n_perturbations_s0))
        influencer_s1 = list(np.random.choice(nodes_y1s0, n_perturbations_s1))
        influencer = influencer_s0 + influencer_s1

        print(len(influencer_s0))
        print(len(influencer_s1))
        print(len(subject_s0))
        print(len(subject_s1))

        unwanted = [np.random.choice(list(G_rem.neighbors(u))) for u in subject]

        # print(subject)
        # print(influencer)

        print(len(subject))
        print(len(influencer))

        assert (len(subject) == len(influencer))

        print(f'{modified_adj[subject, influencer].nnz} edges already exist')

        modified_adj[subject, influencer] = 1
        modified_adj[influencer, subject] = 1

        modified_adj[subject, unwanted] = 0
        modified_adj[unwanted, subject] = 0

        # G = nx.from_scipy_sparse_matrix(modified_adj)
        # print('Average homophily')
        # print(f'{avg_hom(nodes_y0s0):.2f}|{avg_hom(nodes_y0s1):.2f}\n{avg_hom(nodes_y1s0):.2f}|'
        #       f'{avg_hom(nodes_y1s1):.2f}')

        self.check_adj(modified_adj)
        self.modified_adj = modified_adj
        test_surrogate(modified_adj, features, y, s, idx_train, device=self.device)


class SPI_heuristic_old(BaseAttack):

    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        super(SPI_heuristic, self).__init__(model, nnodes, attack_structure=attack_structure,
                                            attack_features=attack_features, device=device)

        assert not self.attack_features, 'SPI_heuristic does NOT support attacking features'

    def attack(self, ori_adj, features, y, s, idx_train, n_perturbations, **kwargs):
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

        y = test_surrogate(ori_adj, features, y, s, idx_train, device=self.device)

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

        # n_remove = n_perturbations//2
        # n_remaining = n_perturbations-n_remove
        #
        # edges_to_remove = [[e[0], e[1]] for e in G.edges() if (y1[e[0]] and y1[e[1]] and s[e[0]] != s[e[1]]) or (y0 and ())]

        # equally attack both sets
        n_perturbations_s0 = n_perturbations // 2
        n_perturbations_s1 = n_perturbations - n_perturbations_s0

        subject_s0 = list(np.random.choice(nodes_y1s1, n_perturbations_s0 // 2))
        subject_s0 += subject_s0
        subject_s1 = list(np.random.choice(nodes_y0s0, n_perturbations_s1 // 2))
        subject_s1 += subject_s1

        subject = subject_s0 + subject_s1

        influencer_s0 = list(np.random.choice(nodes_y0s0 + nodes_y0s1, (n_perturbations_s0 // 2) * 2))
        influencer_s1 = list(np.random.choice(nodes_y1s1 + nodes_y1s0, (n_perturbations_s1 // 2) * 2))
        influencer = influencer_s0 + influencer_s1

        print(len(influencer_s0))
        print(len(influencer_s1))
        print(len(subject_s0))
        print(len(subject_s1))

        # For now, let's choose the ratio of 1 subject node to 2 influencer nodes since we choose both of a low degree
        # subject = subject_s0[:n_perturbations_s0 // 2] + subject_s0[:n_perturbations_s0 - n_perturbations_s0 // 2] + \
        #           subject_s1[:n_perturbations_s1 // 2] + subject_s1[:n_perturbations_s1 - n_perturbations_s1 // 2]
        # influencer = influencer_s0[:n_perturbations_s0] + influencer_s1[:n_perturbations_s1]

        # subject = subject_s0[:n_perturbations_s0] + subject_s1[:n_perturbations_s1]
        # influencer = influencer_s0[:n_perturbations_s0//2] + influencer_s0[:n_perturbations_s0 - n_perturbations_s0//2] + \
        #           influencer_s1[:n_perturbations_s1//2] + influencer_s1[:n_perturbations_s1 - n_perturbations_s1//2]

        print(subject)
        print(influencer)

        print(len(subject))
        print(len(influencer))

        assert (len(subject) == len(influencer))

        print(f'{modified_adj[subject, influencer].nnz} edges already exist')

        modified_adj[subject, influencer] = 1
        modified_adj[influencer, subject] = 1

        self.check_adj(modified_adj)
        self.modified_adj = modified_adj


class SPI_heuristic(BaseAttack):

    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        super(SPI_heuristic, self).__init__(model, nnodes, attack_structure=attack_structure,
                                            attack_features=attack_features, device=device)

        assert not self.attack_features, 'SPI_heuristic does NOT support attacking features'

    def attack(self, ori_adj, features, y, s, idx_train, n_perturbations, **kwargs):
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

        y = test_surrogate(ori_adj, features, y, s, idx_train, device=self.device)

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

        subject = list(np.random.choice(nodes_y0s0, n_perturbations))
        influencer = list(np.random.choice(nodes_y1s1, n_perturbations))

        assert (len(subject) == len(influencer))

        print(f'{modified_adj[subject, influencer].nnz} edges already exist')

        modified_adj[subject, influencer] = 1
        modified_adj[influencer, subject] = 1

        self.check_adj(modified_adj)
        self.modified_adj = modified_adj
        

class MetaSPI(Metattack):
    """
    Attempts to increase the statistical parity by meta learning
    """

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, device='cpu',
                 with_bias=False, lambda_=0.5, train_iters=100, lr=0.1, momentum=0.9):
        super(MetaSPI, self).__init__(model, nnodes, feature_shape, attack_structure, attack_features, device,
                                      with_bias, lambda_, train_iters, lr, momentum)

        self.sens_norm = None
        assert not self.attack_features, 'SPI_heuristic does NOT support attacking features'

    def attack(self, ori_features, ori_adj, labels, sens, idx_train, idx_unlabeled, n_perturbations, ll_constraint=True,
               ll_cutoff=0.004):
        s1 = sens.sum()
        s0 = len(sens) - s1
        self.sens_norm = (1 / s0) - (1 / s0 + 1 / s1) * sens
        print(f'sens_norm = {self.sens_norm}')

        return super().attack(ori_features, ori_adj, labels, idx_train, idx_unlabeled, n_perturbations,
                              ll_constraint=True, ll_cutoff=0.004)

    def get_meta_grad(self, features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training):

        hidden = features
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                hidden = adj_norm @ hidden @ w + b
            if self.with_relu and ix != len(self.weights) - 1:
                hidden = F.relu(hidden)

        output = F.log_softmax(hidden, dim=1)
        yhat = torch.max(F.sigmoid(hidden), dim=1).values

        # print(output)
        loss_labeled = -10000 * yhat[idx_train].dot(self.sens_norm[idx_train]) ** 2
        loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])
        loss_test_val = -10000 * yhat[idx_unlabeled].dot(self.sens_norm[idx_unlabeled]) ** 2

        if self.lambda_ == 1:
            attack_loss = loss_labeled
        elif self.lambda_ == 0:
            attack_loss = loss_unlabeled
        else:
            attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

        print('')
        print('GCN SP on unlabled data: {}'.format(loss_test_val.item()))
        # print('GCN SP acc on unlabled data: {}'.format(utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()))
        print('attack loss: {}'.format(attack_loss.item()))

        adj_grad, feature_grad = None, None
        if self.attack_structure:
            adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
        if self.attack_features:
            feature_grad = torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]
        return adj_grad, feature_grad


class MetaDiscriminator(Metattack):
    """
    Attempts to increase the statistical parity by meta learning
    """

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, device='cpu',
                 with_bias=False, lambda_=0.5, train_iters=100, lr=0.1, momentum=0.9):
        super(MetaDiscriminator, self).__init__(model, nnodes, feature_shape, attack_structure, attack_features, device,
                                                with_bias, lambda_, train_iters, lr, momentum)

        assert not self.attack_features, 'SPI_heuristic does NOT support attacking features'

    def get_meta_grad(self, features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training):

        hidden = features
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                hidden = adj_norm @ hidden @ w + b
            if self.with_relu and ix != len(self.weights) - 1:
                hidden = F.relu(hidden)

        output = F.log_softmax(hidden, dim=1)

        loss_labeled = -F.nll_loss(output[idx_train], labels[idx_train])
        loss_unlabeled = -F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])
        loss_test_val = -F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])

        if self.lambda_ == 1:
            attack_loss = loss_labeled
        elif self.lambda_ == 0:
            attack_loss = loss_unlabeled
        else:
            attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

        print('GCN loss on unlabled data: {}'.format(loss_test_val.item()))
        print(
            'GCN acc on unlabled data: {}'.format(utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()))
        print('attack loss: {}'.format(attack_loss.item()))

        adj_grad, feature_grad = None, None
        if self.attack_structure:
            adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
        if self.attack_features:
            feature_grad = torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]
        return adj_grad, feature_grad
