# Models for generating synthetic graphs
import networkx as nx

node_classes = ['0,0', '0,1', '1,0', '1,1']

sizes = [250, 250, 250, 250]
probs = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
g = nx.stochastic_block_model(sizes, probs, seed=0)

print(len(g.graph['partition']))