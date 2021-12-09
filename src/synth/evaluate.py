from deeprobust.graph.defense import GCN
import torch
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    from models import *
else:
    from src.synth.models import *
import networkx as nx
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def generate_graph(generator, intra_density, seed, feature_size=10, cluster_size=250, train_size=20, val_size=40):
    G = generator(intra_density, .3, seed, cluster_size)
    labels = np.array(([0] * 2 * cluster_size) + ([1] * 2 * cluster_size))
    sens = np.array((([0] * cluster_size) + ([1] * cluster_size)) * 2)

    x1 = generate_features(feature_size, labels, np.random.randint(1000000007))  # features based on labels
    x2 = generate_features(feature_size, sens, np.random.randint(1000000007))  # features based on sensitive attribute
    x3 = generate_features(feature_size, labels * 2 + sens, np.random.randint(1000000007))  # features based on labels and sensitive attribute
    features = torch.Tensor(np.concatenate([x1, x2, x3], axis=1)).to(device)
    n = features.shape[0]
    labels = torch.LongTensor(labels).to(device)
    sens = torch.LongTensor(sens).to(device)
    adj = nx.to_scipy_sparse_matrix(G)

    train_idx = np.random.choice(range(n), train_size)
    val_idx = np.random.choice(range(n), val_size)
    assert(n-train_size-val_size > 0)
    test_idx = np.random.choice(range(n), n-train_size-val_size)

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return adj, features, labels, sens, train_idx, val_idx, test_idx

def evaluate(generator, intra_density, seed):

    adj, features, labels, sens, train_idx, val_idx, test_idx = generate_graph(generator, intra_density, seed)

    gcn = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16,
                    dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, train_idx, train_iters=200)

    y = gcn.predict(features, adj)
    y = y.max(1)[1]

    return compute_accuracy(labels[test_idx], y[test_idx]), compute_statistical_parity(sens[test_idx], y[test_idx])

def compute_accuracy(labels, y):
    return accuracy_score(labels, y)

def compute_statistical_parity(sens, y):
    y1 = y == 1
    s1 = sens == 1
    s0 = sens == 0
    y0 = y == 0

    y1s0 = y1 & s0
    y1s1 = y1 & s1

    dSP = abs(sum(y1s0) / sum(s0) - sum(y1s1) / sum(s1))
    return dSP


def main():


    for generator in [self_connections,label_connections,sensitive_attribute_connections]:

        print('Acc\t\tdSP')
        for intra_density in [1, .8, .6, .4, .2, 0]:
            acc, dSP = evaluate(generator, intra_density)
            print(f'{acc:.2f}\t{dSP:.2f}')
        print(f'=======================')


if __name__ == "__main__":
    main()
