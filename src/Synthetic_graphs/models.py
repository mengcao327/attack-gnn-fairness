# Models for generating synthetic graphs
import networkx as nx
import numpy as np


def self_connections(intra_density, overall_density_factor, seed,
                     cluster_size=250):  # TODO unit test to check if the desired density is reached
    """
    Assuming the groups are ordered as ['y0s0', 'y0s1', 'y1s0', 'y1s1']
    :param intra_density: the edge density in the diagonal (scalar)
    :param overall_density_factor: overall graph density (scalar).
    :param cluster_size:
    The entire density matrix will be multiplied by this factor
    :return:
    """
    n_clusters = 4
    sizes = [cluster_size] * n_clusters

    # distribute the remaining density on the non-diagonal elements
    probs = (1 - intra_density) / (n_clusters - 1) * np.ones([n_clusters, n_clusters])
    # assign intra-density
    for i in range(n_clusters): probs[i, i] = intra_density
    probs *= overall_density_factor

    return nx.stochastic_block_model(sizes, probs, seed=seed)


def label_connections(intra_density, overall_density_factor, seed, cluster_size=250):
    """
    Assuming the groups are ordered as ['y0s0', 'y0s1', 'y1s0', 'y1s1']
    :param intra_density:
    :param overall_density_factor:
    :param seed:
    :param cluster_size:
    :return:
    """
    sizes = [cluster_size] * 4

    # distribute the remaining density on the non-diagonal elements
    probs = (1 - intra_density) / 2 * np.ones([4, 4])
    # assign intra-density
    for i in range(4):
        probs[i, i] = intra_density / 2
        probs[i, i ^ 1] = intra_density / 2
    probs *= overall_density_factor

    return nx.stochastic_block_model(sizes, probs, seed=seed)


def sensitive_attribute_connections(intra_density, overall_density_factor, seed, cluster_size=250):
    """
    Assuming the groups are ordered as ['y0s0', 'y0s1', 'y1s0', 'y1s1']
    :param intra_density:
    :param overall_density_factor:
    :param seed:
    :param cluster_size:
    :return:
    """
    sizes = [cluster_size] * 4

    # distribute the remaining density on the non-diagonal elements
    probs = (1 - intra_density) / 2 * np.ones([4, 4])
    # assign intra-density
    for i in range(4):
        probs[i, i] = intra_density / 2
        probs[i, (i + 2) % 4] = intra_density / 2
    probs *= overall_density_factor

    return nx.stochastic_block_model(sizes, probs, seed=seed)


if __name__ == "__main__":
    for intra_density in [1, .8, .6, .4, .2, 0]:
        G = sensitive_attribute_connections(intra_density, .35, 0)
        H = nx.quotient_graph(G, G.graph["partition"], relabel=True)
        for v in H.nodes(data=True):
            print(round(v[1]["density"], 3))
        print('')
