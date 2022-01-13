import numpy as np
import seaborn as sns
from matplotlib.pyplot import figure
import matplotlib.pylab as plt
import pandas as pd

# datasets = ['nba','region_job','region_job_2','dblp','german','bail','credit']
datasets = ['region_job','region_job_2','dblp','german','bail','credit']
# datasets=['nba']
for i in range(len(datasets)):
# i=3
    dataset=datasets[i]
    filename=dataset + "_homo_edges_rate.csv"
    content = pd.read_csv(filename,header=None)
    data = content.values
    figure(figsize=(4, 3))
    ax = sns.heatmap(data, linewidth=0.15,annot=True,fmt=".4f",cmap="Blues",
                     # vmin=0.0,vmax=0.12,
                     xticklabels=["y0s0","y0s1","y1s0","y1s1"],
                     yticklabels=["y0s0","y0s1","y1s0","y1s1"])
    # ax.set_title(dataset)
    ax.tick_params(axis="x", bottom=False, labeltop=True,top=True,labelbottom=False)
    # plt.show()
    plt.savefig(dataset + '-homo-edges_rate.pdf')#,dpi=300

