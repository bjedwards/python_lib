import networkx as nx
from ..utilities import p_utils
import functools

def pagerank(G):
    try:
        pr = nx.pagerank(G).values()
    except:
        pr = []
    return pr

def _betreduce(bt1,bt2):
    for n in bt1:
        bt1[n] += bt2[n]
    return bt1

def between_cent(G,processes=None):
    p = p_utils.ePool(processes=processes)
    node_divisor = len(p._pool)*4
    node_chunks = list(p_utils.chunks(G.nodes(),G.order()/node_divisor))
    num_chunks = len(node_chunks)
    bt_sc = p.emap(nx.betweenness_centrality_source,
                   [G]*num_chunks,
                   [True]*num_chunks,
                   [False]*num_chunks,
                   node_chunks)
    bt_c = p.reduce(_betreduce,bt_sc)
    return bt_c.values()

def _pathmap(G,n):
    return sum(nx.single_source_shortest_path_length(G,n).values())/float(G.order())

def avg_path_len(G,processes=None):
    order = G.order()
    p = p_utils.ePool(processes=processes)
    pl =  p.emap(_pathmap,
                 [G]*order,
                 G.nodes())
    return pl

def rich_club(G):
    rc = nx.rich_club_coefficient(G,normalized=False)
    return rc

def rich_club_norm(G):
    rc = nx.rich_club_coefficient(G,normalized=True)
    return rc

def clust(G):
    try:
        cc = nx.clustering(G).values()
    except:
        cc = []
    return cc

def cores(G):
    return nx.find_cores(G).values()

def assort(models):
    p = p_utils.ePool()
    k = sorted(models.keys())
    ass = p.emap(nx.degree_assortativity, [models[m] for m in k])
    return dict(zip(k,ass))
