import networkx as nx
import random
import bisect
from math import log10

def generalized_barabasi_albert_graph(n,
                                      m,
                                      alpha=1.0,
                                      G0=None,
                                      create_using=None,
                                      seed=None):
    """Return random graph using Barabasi-Albert preferential attachment
    model.

    As opposed to the vanilla attachment model, attachment proceeds as

           p(v) = d_v^alpha/sum(d_v^alpha)
        
    A graph of n nodes is grown by attaching new nodes each with m
    edges that are preferentially attached to existing nodes with high
    degree.
    
    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    alpha : float
        Tuning parameter.
    G0 : networkx graph
         A seed graph
    create_using : graph, optional (default Graph)
        The graph instance used to build the graph.
    seed : int, optional
        Seed for random number generator (default=None).   

    Returns
    -------
    G : Graph
        
    Notes
    -----
    The initialization is a graph with with m nodes and no edges.

    References
    ----------
    .. [1] A. L. Barabasi and R. Albert 'Emergence of scaling in
       random networks', Science 286, pp 509-512, 1999.
    """
        
    if m < 1 or  m >=n:
        raise nx.NetworkXError(\
              "Barabasi-Albert network must have m>=1 and m<n, m=%d,n=%d"%(m,n))

    if create_using is not None and create_using.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")

    if seed is not None:
        random.seed(seed)    

    if G0 is None:
        G = random_tree(m,create_using)
    else:
        G = G0.copy()
    
    node_weights = {}

    for v in G.nodes():
        node_weights[v] = G.degree(v)**alpha

    i = G.order()
    while i < n:
        G.add_node(i)
        for j in range(m):
            u = weighted_random_choice(node_weights)
            while G.has_edge(i,u):
                u = weighted_random_choice(node_weights)
            G.add_edge(i,u)
            node_weights[u] = G.degree(u)**alpha
        node_weights[i] = G.degree(i)**alpha
        i +=1
    return G

def interactive_growth(n,p,m=2,G0=None,create_using=None,seed=None):
    """ Return a random graph built using the interactive growth model.

    The model proceeds as the Barabasi-Albert model, except it identifies
    two different attachment methods. With probability p a node connects
    into the existing graph with one edge, and two edges are created
    within the existing network based on preferential attachment
    probability. With probabilty 1-p a new node attaches to two nodes
    in the existing network and a single new edge is created within the
    network.


    Parameters
    ----------
    n : int
        Number of nodes
    p : float
        Attachment strategy probability.
    m : int
        size of seed graph if none provided
    G0 : networkx graph
         A seed graph
    create_using : graph, optional (default Graph)
        The graph instance used to build the graph.
    seed : int, optional
        Seed for random number generator (default=None).   

    Returns
    -------
    G : Graph
        
"""
    if seed is not None:
        random.seed(seed)    

    if G0 is None:
        if m < 6:
            m0 = m*6
        else:
            m0 = m
        G = random_tree(m0,create_using)
    else:
        G = G0.copy()
    
    node_list = []
    for v in G:
        for i in range(G.degree(v)):
            node_list.append(v)

    i = G.order()
    while G.order() < n:
        rand_case = random.random()
        print i
        if rand_case < p:
            G.add_node(i)
            w = random.choice(node_list)
            G.add_edge(i,w)
            u1 = random.choice(node_list)
            v1 = random.choice(node_list)
            while (u1 == v1) or G.has_edge(u1,v1):
                u1 = random.choice(node_list)
                v1 = random.choice(node_list)
            u2 = random.choice(node_list)
            v2 = random.choice(node_list)
            while (u2 == v2) or G.has_edge(u2,v2):
                u2 = random.choice(node_list)
                v2 = random.choice(node_list)
            G.add_edge(u1,v1)
            G.add_edge(u2,v2)
            node_list.extend([i,w,u1,u2,v1,v2])
        else:
            G.add_node(i)
            u = random.choice(node_list)
            v = random.choice(node_list)
            while u==v:
                u = random.choice(node_list)
                v = random.choice(node_list)
            G.add_edge(i,u)
            G.add_edge(i,v)
            u1 = random.choice(node_list)
            v1 = random.choice(node_list)
            while (u1 == v1) or G.has_edge(u1,v1):
                u1 = random.choice(node_list)
                v1 = random.choice(node_list)
            G.add_edge(u1,v1)
            node_list.extend([i,u,v,u1,v1])
        i += 1
    return G

def positive_feedback_preference_1(n,p,q,delta,
                                   m=2,
                                   G0=None,
                                   create_using=None,
                                   seed=None):

    """ Return a random graph built using the positive feedback preference
    model.

    The model proceeds as the Barabasi-Albert model, except it identifies
    three different attachment methods. With probability p a node makes
    two connections into an existing network, with probability q-p a node
    makes a single connection into the network, and the node connected to
    makes two additional links into the existing network. With probability
    1-p-q a new node makes two connections into the existing network and
    one of the nodes connected two makes another connection into the
    network.

    New connections are made based on tuning parameter delta, with the
    probability of attachment proportional to:

                    d_v^{1+delta*log10(d_v)

    where d_v is the degree of node v.
    

    Parameters
    ----------
    n : int
        Number of nodes
    p : float
        Attachment strategy probability.
    q : float
        Attachment strategy probability
    delta: float
           Preferential attachment tuning parameter
    m : int
        size of seed graph if none provided
    G0 : networkx graph
         A seed graph
    create_using : graph, optional (default Graph)
        The graph instance used to build the graph.
    seed : int, optional
        Seed for random number generator (default=None).   

    Returns
    -------
    G : Graph
        
"""

    if q > 1.0-p:
        nx.NetworkXError("q must be <= 1-p")
    
    if seed is not None:
        random.seed(seed)    

    if G0 is None:
        G = random_tree(2*(m+1),create_using)
    else:
        G = G0.copy()
    
    node_weights = {}

    for v in G.nodes():
        node_weights[v] = G.degree(v)**(1.0+delta*log10(G.degree(v)))

    i = G.order()
    while i < n:
        print i
        rnd_case = random.random()
        if rnd_case < p:
            u = weighted_random_choice(node_weights)
            v = weighted_random_choice(node_weights)
            while u==v:
                u = weighted_random_choice(node_weights)
                v = weighted_random_choice(node_weights)
            G.add_node(i)
            G.add_edge(i,u)
            G.add_edge(u,v)
            node_weights[u] = G.degree(u)**(1.0+delta*log10(G.degree(u)))
            node_weights[v] = G.degree(v)**(1.0+delta*log10(G.degree(v)))
        elif p <= rnd_case < p + q:
            u = weighted_random_choice(node_weights)
            v = weighted_random_choice(node_weights)
            w = weighted_random_choice(node_weights)
            while v==w or v==u or w==u or G.has_edge(u,v) or G.has_edge(u,w):
                v = weighted_random_choice(node_weights)
                w = weighted_random_choice(node_weights)
            G.add_node(i)
            G.add_edge(i,u)
            G.add_edge(u,v)
            G.add_edge(u,w)
            node_weights[u] = G.degree(u)**(1.0+delta*log10(G.degree(u)))
            node_weights[v] = G.degree(v)**(1.0+delta*log10(G.degree(v)))
            node_weights[w] = G.degree(w)**(1.0+delta*log10(G.degree(w)))
        else:
            u = weighted_random_choice(node_weights)
            v = weighted_random_choice(node_weights)
            while u==v:
                u = weighted_random_choice(node_weights)
                v = weighted_random_choice(node_weights)
            G.add_node(i)
            G.add_edge(i,u)
            G.add_edge(i,v)
            rand_host = random.choice([u,v])
            w = weighted_random_choice(node_weights)
            while w==rand_host or G.has_edge(w,rand_host):
                w = weighted_random_choice(node_weights)
            G.add_edge(w,rand_host)
            node_weights[u] = G.degree(u)**(1.0+delta*log10(G.degree(u)))
            node_weights[v] = G.degree(v)**(1.0+delta*log10(G.degree(v)))
            node_weights[w] = G.degree(w)**(1.0+delta*log10(G.degree(w)))
        node_weights[i] = 1
        i+=1
    return G                

def positive_feedback_preference_2(n,p,delta,
                                   m=2,
                                   G0=None,
                                   create_using=None,
                                   seed=None):

    """ Return a random graph built using the positive feedback preference
    model.

    The model proceeds as the Barabasi-Albert model, except it
    identifies three different attachment methods. With probability p
    a new node makes a single link into the existing network and two
    links are made in the network internally. With probability 1-p a
    new node makes two connections into the existing network and one
    of the nodes connected two makes another connection into the
    network.

    New connections are made based on tuning parameter delta, with the
    probability of attachment proportional to:

                    d_v^{1+delta*log10(d_v)

    where d_v is the degree of node v.
    

    Parameters
    ----------
    n : int
        Number of nodes
    p : float
        Attachment strategy probability.
    delta: float
           Preferential attachment tuning parameter
    m : int
        size of seed graph if none provided
    G0 : networkx graph
         A seed graph
    create_using : graph, optional (default Graph)
        The graph instance used to build the graph.
    seed : int, optional
        Seed for random number generator (default=None).   

    Returns
    -------
    G : Graph
        
"""
    if seed is not None:
        random.seed(seed)    

    if G0 is None:
        G = random_tree(2*(m+1),create_using)
    else:
        G = G0.copy()
    
    node_weights = {}

    for v in G.nodes():
        node_weights[v] = G.degree(v)**(1.0+delta*log10(G.degree(v)))

    i = G.order()
    while i < n:
        print i
        rnd_case = random.random()
        if rnd_case < p:
            u = weighted_random_choice(node_weights)
            v = weighted_random_choice(node_weights)
            w = weighted_random_choice(node_weights)
            while v==w or v==u or w==u or G.has_edge(u,v) or G.has_edge(u,w):
                v = weighted_random_choice(node_weights)
                w = weighted_random_choice(node_weights)
            G.add_node(i)
            G.add_edge(i,u)
            G.add_edge(u,v)
            G.add_edge(u,w)
            node_weights[u] = G.degree(u)**(1.0+delta*log10(G.degree(u)))
            node_weights[v] = G.degree(v)**(1.0+delta*log10(G.degree(v)))
            node_weights[w] = G.degree(w)**(1.0+delta*log10(G.degree(w)))
        else:
            u = weighted_random_choice(node_weights)
            v = weighted_random_choice(node_weights)
            while u==v:
                u = weighted_random_choice(node_weights)
                v = weighted_random_choice(node_weights)
            G.add_node(i)
            G.add_edge(i,u)
            G.add_edge(i,v)
            rand_host = random.choice([u,v])
            w = weighted_random_choice(node_weights)
            while w==rand_host or G.has_edge(w,rand_host):
                w = weighted_random_choice(node_weights)
            G.add_edge(w,rand_host)
            node_weights[u] = G.degree(u)**(1.0+delta*log10(G.degree(u)))
            node_weights[v] = G.degree(v)**(1.0+delta*log10(G.degree(v)))
            node_weights[w] = G.degree(w)**(1.0+delta*log10(G.degree(w)))
        node_weights[i] = 1
        i+=1
    return G

def generalized_linear_preference(n,
                                  m,
                                  p,
                                  beta=0.0,
                                  G0=None,
                                  create_using=None,
                                  seed=None):
    """ Return a random graph built using generalized linear preference
    model.

    The model proceeds by selecting one of two strategies. With probability
    p m new edges are created in the existing graph with node selection
    proportional to:

                  d_v - beta

    where d_v is the degree of node v. With probabilty 1-p a new node
    is created and it connects with m new edges with the same
    attachment probability as above.


    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to create with each iteration
    p : float
        Attachment strategy probability.
    beta: float < 1
           Preferential attachment tuning parameter
    G0 : networkx graph
         A seed graph
    create_using : graph, optional (default Graph)
        The graph instance used to build the graph.
    seed : int, optional
        Seed for random number generator (default=None).   

    Returns
    -------
    G : Graph
        
"""
    if m < 1 or  m >=n:
        raise nx.NetworkXError(\
              "Generalized Linear Preference network must have m>=1 and m<n, m=%d,n=%d"%(m,n))
    if beta > 1:
        raise nx.NetworkXError("Beta must be less than 1")

    if seed is not None:
        random.seed(seed)

    if G0 is None:
        if m < 2:
            m0 = 2
        else:
            m0 = m
        G = random_tree(m0,create_using)
    else:
        G = G0.copy()

    node_weights = {}

    for v in G.nodes():
        node_weights[v] = G.degree(v)-beta

    i = G.order()
    while i < n:
        print i
        rnd_case = random.random()
        if rnd_case < p:
            for l in range(m):
                u = weighted_random_choice(node_weights)
                v = weighted_random_choice(node_weights)
                while u==v or G.has_edge(u,v):
                    u = weighted_random_choice(node_weights)
                    v = weighted_random_choice(node_weights)
                G.add_edge(u,v)
                node_weights[u] += 1.0
                node_weights[v] += 1.0
        else:
            G.add_node(i)
            for l in range(m):
                u = weighted_random_choice(node_weights)
                while G.has_edge(i,u):
                    u = weighted_random_choice(node_weights)
                G.add_edge(i,u)
                node_weights[u] += 1.0
            node_weights[i] = m-beta
            i += 1
    return G    
            
def weighted_random_choice(w):
    """ Select a weighted random choice from a dictionary of key weight
    pairs

    Parameters:
    -----------
    w: dict
       dictionary of item weight pairs

    Returns:
    --------
    key_ind[rand_ret] : Random key
    """
    cs = list(nx.utils.cumulative_sum(w.values()))
    key_ind = dict(zip(range(len(w)),w.keys()))
    rnd = random.random()*cs[-1]
    rand_ret = bisect.bisect_left(cs,rnd)
    return key_ind[rand_ret]

def random_tree(n, create_using=None,seed=None):
    """ Returns a random tree of size n

    Proceeds by creating nodes and selecting uniformly at random
    an existing node to connect to.

    Parameters:
    -----------
    n : int
        Number of nodes
    create_using: networkx graph
                  graph to determine type
    seed: int
          Random seed value

    Returns:
    --------
    G: networkx Graph
       A random tree
    """

    if seed is not None:
        random.seed(seed)

    G = nx.empty_graph(0,create_using)

    G.add_node(0)
    for i in range(1,n):
        u = random.choice(G.nodes())
        G.add_node(i)
        G.add_edge(i,u)
    return G
