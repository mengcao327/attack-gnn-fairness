import argparse
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.nn.pytorch.conv import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.fc = nn.Linear(n_hidden, n_classes)

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        # self.layers.append(
        #     SAGEConv(
        #         n_hidden,
        #         n_classes,
        #         aggregator_type))  # activation None

    def forward(self, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(self.g, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        h = self.fc(h)
        return h
