from deeprobust.graph.defense import GCN
import torch

def fit_surrogate(adj, features, labels, idx_train, device):
    # Setup Surrogate Model
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16,
                    dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)
    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels.cpu(), idx_train, train_iters=500)
    return surrogate


def compute_statistical_parity(sens, y):
    y1 = y > .5
    s1 = sens == 1
    s0 = sens == 0
    y0 = y <= .5

    y1s0 = y1 & s0
    y1s1 = y1 & s1

    # all = sum(y0s0 + y1s0 + y0s1 + y1s1)

    # print('result distribution:')
    # print(f'{sum(y0s0) / all:.2f}|{sum(y0s1) / all:.2f}\n{sum(y1s0) / all:.2f}|{sum(y1s1) / all:.2f}')
    # print(f'dSP = {abs(sum(y1s0) / sum(s0) - sum(y1s1) / sum(s1))}')
    # print('-----------------------')

    dSP = abs(sum(y1s0) / sum(s0) - sum(y1s1) / sum(s1))
    return dSP


def test_surrogate(adj, features, labels, sens, idx_train, device):
    surrogate = fit_surrogate(adj, features, labels, idx_train, device)
    y = surrogate.predict(features, adj)
    y = y.max(1)[1]
    print(f"y device:{y.device}")
    print(f"sens device:{sens.device}")
    print(f'dSP = {compute_statistical_parity(sens.to(device), y.to(device))}')
    return torch.tensor(y > 0.5).type_as(labels)
