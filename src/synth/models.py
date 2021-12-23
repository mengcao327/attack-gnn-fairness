# Models for generating synthetic graphs
import networkx as nx
import numpy as np
import itertools


def self_connections(intra_density, overall_density_factor, seed,
                     cluster_size=250):
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


def sensitive_attribute_same_label(intra_density, overall_density_factor, seed, cluster_size=250):
    """
    Assuming the groups are ordered as ['y0s0', 'y0s1', 'y1s0', 'y1s1']
    :param intra_density:
    :param overall_density_factor:
    :param seed:
    :param cluster_size:
    :return:
    """
    sizes = [cluster_size] * 4

    probs = np.zeros([4, 4])
    # assign intra-density
    for i in range(4):
        probs[i, i] = intra_density
        probs[i, i ^ 1] = 1-intra_density
    probs *= overall_density_factor

    return nx.stochastic_block_model(sizes, probs, seed=seed)


def label_same_sensitive_attribute(intra_density, overall_density_factor, seed, cluster_size=250):
    """
    Assuming the groups are ordered as ['y0s0', 'y0s1', 'y1s0', 'y1s1']
    :param intra_density:
    :param overall_density_factor:
    :param seed:
    :param cluster_size:
    :return:
    """
    sizes = [cluster_size] * 4

    probs = np.zeros([4, 4])
    # assign intra-density
    for i in range(4):
        probs[i, i] = intra_density
        probs[i, (i + 2) % 4] = 1-intra_density
    probs *= overall_density_factor

    return nx.stochastic_block_model(sizes, probs, seed=seed)


def cross_label_cross_sens(intra_density, overall_density_factor, seed, cluster_size=250):
    """
    Assuming the groups are ordered as ['y0s0', 'y0s1', 'y1s0', 'y1s1']
    :param intra_density:
    :param overall_density_factor:
    :param seed:
    :param cluster_size:
    :return:
    """
    sizes = [cluster_size] * 4

    probs = np.zeros([4, 4])
    # assign intra-density
    for i in range(4):
        probs[i, i] = intra_density
        probs[i, 3-i] = 1-intra_density
    probs *= overall_density_factor

    return nx.stochastic_block_model(sizes, probs, seed=seed)


def cross_label(intra_density, overall_density_factor, seed, cluster_size=250):
    """
    Assuming the groups are ordered as ['y0s0', 'y0s1', 'y1s0', 'y1s1']
    :param intra_density:
    :param overall_density_factor:
    :param seed:
    :param cluster_size:
    :return:
    """
    sizes = [cluster_size] * 4

    probs = np.zeros([4, 4])
    # assign intra-density
    for i in range(4):
        probs[i, i] = intra_density
        probs[i, 3-i] = (1-intra_density)/2
        probs[i, (i + 2) % 4] = (1-intra_density)/2
    probs *= overall_density_factor

    return nx.stochastic_block_model(sizes, probs, seed=seed)


def uniform_with_anomaly(intra_density, a, b, common_density, overall_density_factor, seed, cluster_size=250):
    sizes = [cluster_size] * 4

    # set uniform density
    probs = common_density*np.ones([4, 4])
    # add some homophily
    # for i in range(4):
    #     probs[i][i] += .5 * common_density
    #     probs[i][i ^ 1] += .5 * common_density

    # set desired cell to specified density
    probs[a][b] = intra_density
    probs[b][a] = intra_density

    probs = probs / np.sum(probs)

    probs *= overall_density_factor

    return nx.stochastic_block_model(sizes, probs, seed=seed)


def uniform_with_anomaly_assortative(intra_density, a, b, common_density, overall_density_factor, seed, cluster_size=250):
    sizes = [cluster_size] * 4

    # set uniform density
    probs = common_density*np.ones([4, 4])
    # add some homophily
    for i in range(4):
        probs[i][i] += .5 * common_density
        probs[i][i ^ 1] += .5 * common_density

    # set desired cell to specified density
    probs[a][b] = intra_density
    probs[b][a] = intra_density

    probs = probs / np.sum(probs)

    probs *= overall_density_factor

    return nx.stochastic_block_model(sizes, probs, seed=seed)


def symmetric_cross_same(intra_density, common_density, overall_density_factor, seed, cluster_size=250):
    sizes = [cluster_size] * 4

    # set uniform density
    probs = common_density*np.ones([4, 4])
    # add some homophily
    # for i in range(4):
    #     probs[i][i] += .5 * common_density
    #     probs[i][i ^ 1] += .5 * common_density

    # set desired cell to specified density
    probs[0][2] = intra_density
    probs[2][0] = intra_density
    probs[1][3] = intra_density
    probs[3][1] = intra_density

    probs = probs / np.sum(probs)

    probs *= overall_density_factor

    return nx.stochastic_block_model(sizes, probs, seed=seed)


def symmetric_cross_cross(intra_density, common_density, overall_density_factor, seed, cluster_size=250):
    sizes = [cluster_size] * 4

    # set uniform density
    probs = common_density*np.ones([4, 4])
    # add some homophily
    # for i in range(4):
    #     probs[i][i] += .5 * common_density
    #     probs[i][i ^ 1] += .5 * common_density

    # set desired cell to specified density
    probs[0][3] = intra_density
    probs[3][0] = intra_density
    probs[1][2] = intra_density
    probs[2][1] = intra_density

    probs = probs / np.sum(probs)

    probs *= overall_density_factor

    return nx.stochastic_block_model(sizes, probs, seed=seed)


def uniform_increase_intra(intra_density, common_density, overall_density_factor, seed, cluster_size=250):
    sizes = [cluster_size] * 4

    # set uniform density
    probs = common_density*np.ones([4, 4])
    # for i in range(4):
    #     probs[i][i] = intra_density
    probs[0][0] = intra_density
    probs[3][3] = intra_density
    probs *= overall_density_factor

    return nx.stochastic_block_model(sizes, probs, seed=seed)


def uniform_with_symmetric_anomaly(intra_density, common_density, overall_density_factor, seed, cluster_size=250):
    sizes = [cluster_size] * 4

    # set uniform density
    probs = common_density*np.ones([4, 4])
    # add some homophily
    for i in range(4):
        probs[i][i] += .5 * common_density
        probs[i][i ^ 1] += .5 * common_density

    # set desired cell to specified density
    probs[0][3] = intra_density
    probs[3][0] = intra_density
    probs[1][2] = intra_density
    probs[2][1] = intra_density

    probs *= overall_density_factor

    return nx.stochastic_block_model(sizes, probs, seed=seed)



def generate_features(dimensions, labels, cov_diag=.5):
    """
    Generate features between -1 and 1
    Labels can refer to any categorical feature (e.g., classes or sensitive attributes)
    :param dimensions:
    :return:
    """

    n = len(labels)
    labels = np.array(labels)
    n_labels = len(np.unique(labels))
    means = list(itertools.product([1, -1], repeat=dimensions))

    features = np.zeros((n, dimensions))
    for label, mean in zip(range(n_labels), means[:n_labels]):
        num_features_per_label = sum(labels == label)
        cov = np.diag([cov_diag] * dimensions)  # defines separability of classes
        sample = np.random.multivariate_normal(mean, cov, num_features_per_label)
        features[labels == label] = sample

    return features


if __name__ == "__main__":
    for intra_density in [1, .8, .6, .4, .2, 0]:
        cluster_size = 250
        G = sensitive_attribute_connections(intra_density, .35, 0, cluster_size)
        H = nx.quotient_graph(G, G.graph["partition"], relabel=True)
        for v in H.nodes(data=True):
            print(round(v[1]["density"], 3))
        labels = np.array(([0] * 2 * cluster_size) + ([1] * 2 * cluster_size))
        sens = np.array((([0] * cluster_size) + ([1] * cluster_size)) * 2)
        x1 = generate_features(10, labels, 0)  # features based on labels
        x2 = generate_features(10, sens, 0)  # features based on sensitive attribute
        x3 = generate_features(10, labels * 2 + sens, 0)  # features based on labels and sensitive attribute

        print(np.mean([u.dot(v) for u in x1[:500] for v in x1[:500]]) - np.mean([u.dot(v) for u in x1[:500] for v in x1[500:]]))  # big
        print(np.mean([u.dot(v) for u in x2[:500] for v in x2[:500]]) - np.mean([u.dot(v) for u in x2[:500] for v in x2[500:]]))  # small
        print(np.mean([u.dot(v) for u in x2[:250] for v in x2[:250]]) - np.mean([u.dot(v) for u in x2[:250] for v in x2[250:500]]))  # big
        print(np.mean([u.dot(v) for u in x2[:250] for v in x2[:250]]) - np.mean([u.dot(v) for u in x2[:250] for v in x2[500:750]]))  # small
        print(np.mean([u.dot(v) for u in x3[:250] for v in x3[:250]]) - np.mean([u.dot(v) for u in x3[:250] for v in x3[500:750]]))  # big
        print(np.mean([u.dot(v) for u in x3[:250] for v in x3[:250]]) - np.mean([u.dot(v) for u in x3[:250] for v in x3[250:500]]))  # big
        print('-------------')
