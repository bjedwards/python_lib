#!/usr/bin/env python
"""
Atlas of all graphs of 6 nodes or less.

"""
__author__ = """Aric Hagberg (hagberg@lanl.gov)"""
#    Copyright (C) 2004 by 
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.
#    BSD license.

import networkx as nx
#from networkx import *
#from networkx.generators.atlas import *
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic as isomorphic
import random

def atlas6():
    """ Return the atlas of all connected graphs of 6 nodes or less.
        Attempt to check for isomorphisms and remove.
    """

    Atlas=nx.graph_atlas_g()[0:208] # 208
    # remove isolated nodes, only connected graphs are left
    U=nx.Graph() # graph for union of all graphs in atlas
    for G in Atlas: 
        zerodegree=[n for n in G if G.degree(n)==0]
        for n in zerodegree:
            G.remove_node(n)
        U=nx.disjoint_union(U,G)

    # list of graphs of all connected components        
    C=nx.connected_component_subgraphs(U)        
    
    UU=nx.Graph()        
    # do quick isomorphic-like check, not a true isomorphism checker     
    nlist=[] # list of nonisomorphic graphs
    for G in C:
        # check against all nonisomorphic graphs so far
        if not iso(G,nlist):
            nlist.append(G)
            UU=nx.disjoint_union(UU,G) # union the nonisomorphic graphs  
    return UU            

def iso(G1, glist):
    """Quick and dirty nonisomorphism checker used to check isomorphisms."""
    for G2 in glist:
        if isomorphic(G1,G2):
            return True
    return False        


if __name__ == '__main__':

    import networkx as nx

    G=atlas6()

    print("graph has %d nodes with %d edges"\
          %(nx.number_of_nodes(G),nx.number_of_edges(G)))
    print(nx.number_connected_components(G),"connected components")


    try:
        from networkx import graphviz_layout
    except ImportError:
        raise ImportError("This example needs Graphviz and either PyGraphviz or Pydot")

    import nx_opengl
    # layout graphs with positions using graphviz neato
#    pos=nx.spring_layout(G)
    """
    max_pos_x = max([pos[n][0] for n in pos])
    min_pos_x = min([pos[n][0] for n in pos])
    max_pos_y = max([pos[n][1] for n in pos])
    min_pos_y = min([pos[n][1] for n in pos])
    for n in pos:
            pos[n] = (1.9*(((pos[n][0] - min_pos_x)/(max_pos_x-min_pos_x)) - .5),
                      1.9*(((pos[n][1] - min_pos_y)/(max_pos_y-min_pos_y)) - .5))
    """
    # color nodes the same in each connected subgraph
    C=nx.connected_component_subgraphs(G)
    colors = [0.0]*G.order()
    for g in C:
        c=random.random() # random color...
        for n in g.nodes():
            colors[n] = c
    nx_opengl.draw_opengl(G,3,
                          node_size=5,
                          node_color=colors,
                          with_node_labels=False,
                          gl_fancy_nodes=True)
