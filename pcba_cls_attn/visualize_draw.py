
from torch_geometric import utils 
from collections import defaultdict 
import networkx as nx 
import matplotlib.pyplot as plt


def draw(data): 
    graph = utils.to_networkx(data, node_attrs=['tag', 'attn_lgi', 'attn_gnnt', 'attn_parallel'], to_undirected=True) 

    nodecolor = ['tag', 'attn_lgi', 'attn_gnnt', 'attn_parallel'] 

    plt.switch_backend("agg")
    fig = plt.figure(figsize=(4*len(nodecolor), 4), dpi=300) 

    node_colors = defaultdict(list) 

    titles = {
        'tag': "molecule",
        'attn_lgi': "LGI", 
        'attn_gnnt': "GNN+Transformer", 
        'attn_parallel': "parallel" 
    }

    for i in graph.nodes():
        for key in nodecolor:
            node_colors[key].append(graph.nodes[i][key])

    vmax = {}
    cmap = {}
    for key in nodecolor:
        vmax[key] = 19
        cmap[key] = 'tab20'
        if 'attn' in key:
            vmax[key] = max(node_colors[key])
            cmap[key] = 'Reds'

    pos_layout = nx.kamada_kawai_layout(graph, weight=None)

    for i, key in enumerate(nodecolor):
        ax = fig.add_subplot(1, len(nodecolor), i+1)
        ax.set_title(titles[key], fontweight='bold', font_dict={'size': 18})
        nx.draw(
            graph,
            pos=pos_layout,
            with_labels=False,
            font_size=4,
            node_color=node_colors[key],
            vmin=0,
            vmax=vmax[key],
            cmap=cmap[key],
            width=1.3,
            node_size=100,
            alpha=1.0,
        )
        if 'attn' in key: 
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            sm = plt.cm.ScalarMappable(cmap=cmap[key], norm=plt.Normalize(vmin=0, vmax=vmax[key]))
            sm._A = []
            plt.colorbar(sm, cax=cax) 
        
        # if 'tag' in key: 
        #     from mpl_toolkits.axes_grid1 import make_axes_locatable 
        #     divider = make_axes_locatable(ax) 
        #     cax = divider.append_axes('right', size='5%', pad=0.05) 
        #     sm = plt.cm.ScalarMappable(cmap=cmap[key], norm=None) 
        #     sm._A = []
        #     plt.colorbar(sm, ticks=[5, 6, 7], cax=cax) 

    fig.axes[0].xaxis.set_visible(False)
    fig.canvas.draw()

    plt.savefig("Test.svg", dpi=600) 