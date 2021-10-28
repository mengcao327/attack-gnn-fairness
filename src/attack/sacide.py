import random
import numpy as np
import scipy.sparse as sp
from deeprobust.graph.global_attack import BaseAttack


class SACIDE(BaseAttack):

    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        super(SACIDE, self).__init__(model, nnodes, attack_structure=attack_structure,
                                     attack_features=attack_features, device=device)

        assert not self.attack_features, 'SACIDE does NOT support attacking features'

    def attack(self, ori_adj, sens, n_perturbations, **kwargs):
        """
        Sensitive attribute connect internally, disconnect externally. This is a baseline fairness attack on GNNs that
        links nodes with the same value of the sensitive attribute and unlinks nodes with different value of the
        sensitive attribute. The aim is to segregate groups to monitor the response of GNNs in segregated communities.

        :param ori_adj: scipy.sparse.csr_matrix
                   Original (unperturbed) adjacency matrix.
        :param sens:
                    node sensitive attributes
        :param n_perturbations: Number of edge removals/additions
        :return:
        """

        print('SACIDE: number of pertubations: %s' % n_perturbations)
        modified_adj = ori_adj.tolil()

        remove_or_insert = np.random.choice(2, n_perturbations)
        n_remove = sum(remove_or_insert)

        indices = sp.triu(modified_adj).nonzero()
        possible_indices = [x for x in zip(indices[0], indices[1])
                            if sens[x[0]] != sens[x[1]]]

        remove_indices = np.random.permutation(possible_indices)[: n_remove]
        modified_adj[remove_indices[:, 0], remove_indices[:, 1]] = 0
        modified_adj[remove_indices[:, 1], remove_indices[:, 0]] = 0

        n_insert = n_perturbations - n_remove

        # sample edges to add
        edges_to_add = []
        while len(edges_to_add) < n_insert:
            n_remaining = n_insert - len(edges_to_add)
            candidate_edges = np.array([np.random.choice(ori_adj.shape[0], n_remaining),
                                        np.random.choice(ori_adj.shape[0], n_remaining)]).T
            candidate_edges = [[u, v] for u, v in candidate_edges if sens[u] == sens[v] and modified_adj[u, v] == 0]
            edges_to_add += candidate_edges
        edges_to_add = np.array(edges_to_add)
        modified_adj[edges_to_add[:, 0], edges_to_add[:, 1]] = 1
        modified_adj[edges_to_add[:, 1], edges_to_add[:, 0]] = 1

        self.check_adj(modified_adj)
        self.modified_adj = modified_adj




