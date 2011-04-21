import networkx as nx
import random
import types
import copy
from itertools import repeat
from ..utilities import p_utils

def heuristically_optimized_trade_offs(n,alpha=5.0,centrality_function="center",seed=None):
    """ This function generates a graph using the method described in [1]

    The algorithm is as follows:

        '... our model a tree is built as nodes arrive uniformly
        at random in the unit square (the shape is, as usual,
        inconsequential). When the ith node arrives, it attaches
        itself on one of the previous nodes... Node i attaches
        itself to the node j that minimizes the weighted sum
        of the two objectives:

            min_{j<i} alpha*dij + hj

        where dij is the Euclidean distance, and hj is some
        measure of the 'centrality' of node j, such as (a)
        the average number of hops from other nodes; (b) the
        maximum number of hops from another node; (c) the
        number of hops from a fixed center of the tree; our
        experiments show that all three measures result in
        similar power laws, even though we prove it for (c).
        alpha is a parameter best though as a function of the final
        number n of points, gauging the relative importance of
        the two objectives'[1]

    Parameters
    ----------
    n                   : int
                          The number of nodes the final graph is to have
    alpha               : numeric
                          The weight for the distance parameter
    centrality_function : string or function
                          If a string can be either max, avg or center
                          corresponding to
                              max: max distance to other nodes
                              avg: average distance to other nodes
                              center: distance to center.
                          The first two are relatively slow, running in
                          O(N^2E + N^3LogN), where as, center (the
                          default) is O(N^2)

                          If a function is given this is a custom
                          centrality function. Note that in this case
                          this is the inverse of how we usually measure
                          centrality. The function should take a graph
                          and a node label n and return a numeric value.
                          This numeric value should be higher for a node
                          with a 'lower' notion of centrality, and lower
                          for a higher notion of centrality.
    seed                : A random seed

    Returns
    -------
    G : networkx Graph
        A graph built using the above algorithm

    References
    ----------
    [1] A Fabrikant, E Koutsoupias, C H Papdimitriou, 'Heuristically
        Optimzed Trade-Offs: A new Paradigm for Power Laws in the
        Internet', Lecture Notes in Computer Sciencel; Vol 2380
        Proceedings of the 29th Internation Colloquium on Automata
        Languages and Programming. Pages 110-122. 2002.
    """
    if type(centrality_function) is types.StringType:
        if centrality_function == "center":
            G = nx.Graph()
            G.add_node(0,x=random.random(),y=random.random())
            distCent={}
            distCent[0] = 0
            for i in range(1,n):
                G.add_node(i,x=random.random(),y=random.random())
                minNode = -1
                minDist = float('infinity')
                for j in G:
                    if not (i==j):
                        dist = alpha*(((G.node[i]['x']-G.node[j]['x'])**2 + (G.node[i]['y']-G.node[j]['y'])**2)**(0.5)) + distCent[j]
                    if dist < minDist:
                        minDist = dist
                        minNode = j
                G.add_edge(i,minNode)
                distCent[i] = distCent[minNode]+1
            return G
        elif centrality_function == "max":
            centrality_function = lambda G,n: max(nx.shortest_path_length(G,n).values())
        elif centrality_function == "avg":
            centrality_function = lambda G,n: sum(nx.shortest_path_length(G,n).values())/float(G.order() -1)
    elif not (type(centrality_function) is types.FunctionType):
        nx.NetworkXError("centrality_function must be a function or a string ('center', 'max', 'avg')")
    if not (seed==None):
        random.seed(seed)
    if centrality_function==None:
        centrality_function = lambda G,n: nx.shortest_path_length(G,n,0)
    G = nx.Graph()
    G.add_node(0,x=random.random(),y=random.random())
    for i in range(1,n):
        G.add_node(i,x=random.random(),y=random.random())
        minNode = -1
        minDist = float('infinity')
        for j in G:
            if not (i==j):
                dist = alpha*(((G.node[i]['x']-G.node[j]['x'])**2 + (G.node[i]['y']-G.node[j]['y'])**2)**(0.5)) + centrality_function(G,j)
                if dist < minDist:
                    minDist = dist
                    minNode = j
        G.add_edge(i,minNode)
    return G

def multivariate_heuristically_optimized_trade_offs(n,
                                                    k=0.05,
                                                    beta=1.0,
                                                    N=3,
                                                    radius=0.1,
                                                    p_d=6.0,
                                                    seed=None):

    """ This function generates a graph using the method described in [1]

        Nodes are given a selection set of size N, which determines
        what kinds of services the nodes 'specialize' in, as well as
        as set N of a nodes ability in each area.

        Nodes make a single connection to nodes within radius, which
        'dominate' on a nodes selection criteria.

        In contrast to the orignal model the nodes are allowed to have
        multiple points of presence, and the minimization of distance
        occurs over all locations for all nodes. An existing node u is
        allowed to add new locations with probility given by:

            p_loc(u) = K*rank(u)^(-beta)

        with K and beta being user provided parameters, and rank(u) is
        the rank of a node u when all existing nodes in the graph
        are sorted by the number of their children nodes in a
        monotonically decreasing order.

        Finally, this model allows for nodes to be removed. Every p_d
        iterations a node is removed, and it's children are reattached
        by minimizing the expected distance between them and a potential
        new parent.

    Parameters
    ----------
    n                   : int
                          The number of nodes the final graph is to have
    k                   : numeric
                          location expansion tuning parameter
    beta                : numeric
                          location expansion tuning parameter
    N                   : int
                          Size of seleciton set
    radius              : numeric
                          Radius in which to connect.
    p_d                 : int
                          Interval between node deletions
    seed                : A random seed

    Returns
    -------
    G : networkx Graph
        A graph built using the above algorithm

    References
    ----------
    [1] Chang et al. 'Internet connectivity at the AS-level: An optimzation
    driven modeling approach. SIGCOMM 2003
    """
    G = nx.Graph()
    dist_cent = {}
    num_child = {}
    children = {}
    descendants = {}
    parent_node = {}
    score_vec = {}
    select_set = {}
    poss_select_set = make_select_set(N)
    poss_select_set.remove(list(repeat(0,N)))
    G.add_node(0)
    G.node[0]['loc_list'] = [(random.random(),random.random())]
    num_child[0] = 0
    children[0] = []
    descendants[0] = []
    parent_node[0] = 0
    score_vec[0] = [random.random() for count in range(0,N)]
    select_set[0] = random.choice(poss_select_set)
    num_zeros = 0
    for i in range(1,int(n*(p_d/(p_d-1)))):
        print("i=",i)
        if (i % int(p_d)) == 0 and i>0:
            d = random.choice(G.nodes())
            while d==0:
                d = random.choice(G.nodes())
            G.remove_node(d)
            for c in children[d]:
                cand = find_nodes_in_range(G,
                                           c,
                                           radius,
                                           score_vec,
                                           select_set[c],
                                           exclude_nodes = descendants[c])
                min_node = random.choice(cand)
                G.add_edge(c,min_node)
                parent_node[c] = min_node
                num_child[min_node] += 1
                children[min_node].append(c)
                x = c
                y = min_node
                while not (x==y):
                    descendants[y] = descendants[y] + descendants[c] + [c]
                    x = y
                    y = parent_node[y]
            for w in G.nodes():
                try:
                    children[w].remove(d)
                    descendants[w].remove(d)
                except:
                    pass
            num_child.pop(d)
            children.pop(d)
            descendants.pop(d)
            parent_node.pop(d)
            score_vec.pop(d)
            select_set.pop(d)
        G.add_node(i)
        G.node[i]['loc_list'] = [(random.random(),random.random())]
        score_vec[i] = [random.random() for count in range(0,N)]
        select_set[i] = random.choice(poss_select_set)
        cand = find_nodes_in_range(G,i,radius,score_vec,select_set[i])
        min_node = random.choice(cand)
        G.add_edge(i,min_node)
        num_child[i] = 0
        children[i] = []
        descendants[i] = []
        parent_node[i] = min_node
        num_child[min_node] += 1
        children[min_node].append(i)
        x = i
        y = min_node
        while not (x==y):
            descendants[y].append(i)
            x = y
            y = parent_node[y]
        rank = sorted([(num_child[v],v) for v in G.nodes()],reverse=True)
        for u in range(len(rank)):
            p_loc = k*((u+1.0)**(-beta))
            if random.random() < p_loc and not (u==i):
                G.node[rank[u][1]]['loc_list'].append((random.random(),random.random()))
    return G

def bivariate_heuristically_optimized_trade_offs(n,
                                                 k=0.05,
                                                 beta=1.0,
                                                 N=3,
                                                 radius=0.1,
                                                 seed=None):
    
    """ This function generates a graph using the method described in [1]

        Nodes are given a selection set of size N, which determines
        what kinds of services the nodes 'specialize' in, as well as
        as set N of a nodes ability in each area.

        Nodes make a single connection to nodes within radius, which
        'dominate' on a nodes selection criteria.

        In contrast to the orignal model the nodes are allowed to have
        multiple points of presence, and the minimization of distance
        occurs over all locations for all nodes. An existing node u is
        allowed to add new locations with probility given by:

            p_loc(u) = K*rank(u)^(-beta)

        with K and beta being user provided parameters, and rank(u) is
        the rank of a node u when all existing nodes in the graph
        are sorted by the number of their children nodes in a
        monotinically decreasing order.

    Parameters
    ----------
    n                   : int
                          The number of nodes the final graph is to have
    k                   : numeric
                          location expansion tuning parameter
    beta                : numeric
                          location expansion tuning parameter
    N                   : int
                          Size of seleciton set
    radius              : numeric
                          Radius in which to connect.
    seed                : A random seed

    Returns
    -------
    G : networkx Graph
        A graph built using the above algorithm

    References
    ----------
    [1] Chang et al. 'Internet connectivity at the AS-level: An optimzation
    driven modeling approach. SIGCOMM 2003
    """
    G = nx.Graph()
    dist_cent = {}
    num_child = {}
    parent_node = {}
    score_vec = {}
    select_set = {}
    poss_select_set = make_select_set(N)
    poss_select_set.remove(list(repeat(0,N)))
    G.add_node(0)
    G.node[0]['loc_list'] = [(random.random(),random.random())]
    num_child[0] = 0
    parent_node[0] = 0
    score_vec[0] = []
    select_set[0] = []
    num_zeros = 0
    score_vec[0] = [random.random() for i in range(0,N)]
    select_set[0] = random.choice(poss_select_set)
    for i in range(1,n):
        print(i)
        G.add_node(i)
        G.node[i]['loc_list'] = [(random.random(),random.random())]
        score_vec[i] = [random.random() for count in range(0,N)]
        select_set[i] = random.choice(poss_select_set)
        r = radius
        cand = find_nodes_in_range(G,i,radius,score_vec,select_set[i])
        min_node = random.choice(cand)
        G.add_edge(i,min_node)
        num_child[i] = 0
        parent_node[i] = min_node
        num_child[min_node] += 1
        rank = sorted([(num_child[v],v) for v in range(0,i)],reverse=True)
        for u in range(0,i):
            p_loc = k*((u+1.0)**(-beta))
            if random.random() < p_loc:
                G.node[rank[u][1]]['loc_list'].append((random.random(),random.random()))
    return G

def univariate_heuristically_optimized_trade_offs(n,
                                                  k=0.05,
                                                  beta=1.0,
                                                  seed=None):
    """ This function generates a graph using the method described in [1]

        The algorithm constructs a heuristically optimized tradeoff model
        with several modifications. That is new nodes are added with a
        single link which minimizes

            min_{j<i} dij 

        where dij is the Euclidean distance,  we do not
        allow the centraility function to vary as in the original
        HOT model.

        In contrast to the orignal model the nodes are allowed to have
        multiple points of presence, and the minimization of distance
        occurs over all locations for all nodes. An existing node u is
        allowed to add new locations with probility given by:

            p_loc(u) = K*rank(u)^(-beta)

        with K and beta being user provided parameters, and rank(u) is
        the rank of a node u when all existing nodes in the graph
        are sorted by the number of their children nodes in a
        monotinically decreasing order.

    Parameters
    ----------
    n                   : int
                          The number of nodes the final graph is to have
    k                   : numeric
                          location expansion tuning parameter
    beta                : numeric
                          location expansion tuning parameter
    seed                : A random seed

    Returns
    -------
    G : networkx Graph
        A graph built using the above algorithm

    References
    ----------
    [1] Chang et al. 'Internet connectivity at the AS-level: An optimzation
    driven modeling approach. SIGCOMM 2003
    """
    G = nx.Graph()
    num_child = {}
    parent_node = {}
    G.add_node(0)
    G.node[0]['loc_list'] = [(random.random(),random.random())]
    num_child[0] = 0
    parent_node[0] = 0
    for i in range(1,n):
        print(i)
        G.add_node(i)
        G.node[i]['loc_list'] = [(random.random(),random.random())]
        min_node = -1
        min_dist = float('infinity')
        for j in G:
            if not (i==j):
                dist = min_euclid_dist(G,i,j)
            if dist < min_dist:
                min_dist = dist
                min_node = j
        G.add_edge(i,min_node)
        num_child[i] = 0
        parent_node[i] = min_node
        num_child[min_node] += 1
        rank = sorted([(num_child[v],v) for v in range(0,i)],reverse=True)
        for u in range(0,i):
            p_loc = k*((u+1.0)**(-beta))
            if random.random() < p_loc:
                G.node[rank[u][1]]['loc_list'].append((random.random(),random.random()))
    return G

def find_nodes_in_range(G,i,radius,score_vec,select_set,exclude_nodes=[]):
    candidates = []
    locs_i = float(len(G.node[i]['loc_list']))
    for j in G:
        dist = 0
        for (xi,yi) in G.node[i]['loc_list']:
            min_dist = float('infinity')
            for (xj,yj) in G.node[j]['loc_list']:
                test_dist = (((xi-xj)**2 + (yi-yj)**2)**(0.5))
                if test_dist < min_dist:
                    min_dist = test_dist
            dist += min_dist/locs_i
        if dist <= radius and not i==j:
            candidates.append(j)
    for n in exclude_nodes:
        try:
            candidates.remove(n)
        except:
            pass
    if candidates == []:
        return find_nodes_in_range(G,i,radius*2,score_vec,select_set,exclude_nodes)
    else:
        candidates = remove_dom(score_vec,candidates,select_set)
        return candidates

def remove_dom(score_vec,cand,s_i):
    cand_rem = copy.copy(cand)
    for i in cand:
        for j in cand:
            n = 0
            m = 0
            for x in range(len(s_i)):
                if score_vec[i][x] >= score_vec[j][x]:
                    n += 1
                if score_vec[i][x] > score_vec[j][x]:
                    m += 1
            if n == len(s_i) and m > 0:
                try:
                    cand_rem.remove(j)
                except:
                    pass
    return cand_rem

def make_select_set(N,seed = [[]]):
    if len(seed[0]) == N:
        return seed
    else:
        return make_select_set(N,map(lambda s: s + [0],seed)) + \
               make_select_set(N,map(lambda s: s + [1],seed))

def min_euclid_dist(G,i,j):
    min_dist = float('infinity')
    xi = G.node[i]['loc_list'][0][0]
    yi = G.node[i]['loc_list'][0][1]
    for (xj,yj) in G.node[j]['loc_list']:
        dist = ((xi-xj)**2 + (yi-yj)**2)**(0.5)
        if dist < min_dist:
            min_dist = dist
    return min_dist

