from deeprobust.graph.defense import GCN
import torch
from sklearn.metrics import accuracy_score
import scipy.sparse as sp
from deeprobust.graph import utils
import torch.nn.functional as F

from models import *
# if __name__ == "__main__":
#     from models import *
# else:
#     from src.synth.models import *

import networkx as nx
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def generate_graph(G, seed, feature_size=10, cluster_size=250, train_size=20, val_size=40):
    labels = np.array(([0] * 2 * cluster_size) + ([1] * 2 * cluster_size))
    sens = np.array((([0] * cluster_size) + ([1] * cluster_size)) * 2)

    np.random.seed(seed)
    x1 = generate_features(feature_size, labels)  # features based on labels
    x2 = generate_features(feature_size, sens)  # features based on sensitive attribute
    # x3 = generate_features(feature_size, labels * 2 + sens, np.random.randint(1000000007))  # features based on labels and sensitive attribute
    # x1 = np.eye(cluster_size*4)
    features = torch.Tensor(np.concatenate([x1], axis=1)).to(device)
    n = features.shape[0]
    labels = torch.LongTensor(labels).to(device)
    sens = torch.LongTensor(sens).to(device)
    adj = nx.to_scipy_sparse_matrix(G)

    train_idx = np.random.choice(range(n), train_size)
    val_idx = np.random.choice(range(n), val_size)
    assert (n - train_size - val_size > 0)
    test_idx = np.random.choice(range(n), n - train_size - val_size)

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return adj, features, labels, sens, train_idx, val_idx, test_idx


class GCN2(GCN):

    def forward2(self, x, adj):
        if self.with_relu:
            x = F.relu(self.gc1(x, adj))
        else:
            x = self.gc1(x, adj)

        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)
        return x

    def get_hidden_representations(self, features=None, adj=None):
        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward2(self.features, self.adj_norm)


def get_latent_and_prediction(adj,features,labels,train_idx,device):

    gcn = GCN2(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16,
              dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)

    # gcn = SGC(nfeat=features.shape[1], nclass=labels.max().item() + 1, K=2, device=device)
    gcn = gcn.to(device)
    gcn.fit(features.to('cpu'), adj, labels.to('cpu'), train_idx, train_iters=200)

    y = gcn.predict(features.to('cpu'), adj)
    x = gcn.get_hidden_representations(features.to('cpu'), adj)
    y = y.max(1)[1]

    return x,y

def evaluate(G, seed, cluster_size):
    adj, features, labels, sens, train_idx, val_idx, test_idx = generate_graph(G, seed, cluster_size=cluster_size)
    
    x, y = get_latent_and_prediction(adj,features,labels,train_idx,device)

    return compute_accuracy(labels[test_idx].to('cpu').numpy(), y[test_idx].to('cpu').numpy()), compute_statistical_parity(sens[test_idx], y[test_idx]), x.detach().to('cpu').numpy()


def compute_accuracy(labels, y):
    return accuracy_score(labels, y)


def compute_statistical_parity(sens, y):
    y1 = y == 1
    s1 = sens == 1
    s0 = sens == 0
    y0 = y == 0

    y1s0 = y1 & s0
    y1s1 = y1 & s1

    dSP = sum(y1s0) / sum(s0) - sum(y1s1) / sum(s1) # now returning the signed value to ensure consistency
    return dSP.item()


def main():
    for generator in [sensitive_attribute_same_label, label_same_sensitive_attribute]:

        print('Acc\t\tdSP')
        for intra_density in np.arange(0, 1.01, .1):
            G = generator(intra_density, .3, 0, 250)
            acc, dSP, x = evaluate(G, 0)
            print(f'{acc:.2f}\t{dSP:.2f}')
        print(f'=======================')


if __name__ == "__main__":
    main()
