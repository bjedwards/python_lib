import networkx as nx
import random
from math import e
from ..utilities.stats.power_law import rand_discrete as rand_pl

def waxman(n,alpha=.0015,beta=0.6):
    """Return a random graph created according to the waxman model.

    This model proceeds by placing n nodes uniformly at random in
    the unit square. Each node is then connected to every other with
    probabilty

             alpha*exp(d_ij/(beta*L))

    where d_ij is the distance between the nodes, alpha and beta
    are tuning parameters and $L$ is the maximum distance between
    any two nodes.

    Parameters:
    -----------
    n : int
        Size of graph
    alpha: numeric
           Tuning parameter
    beta: numeric
          Tuning parameter
    Returns:
    --------
    G: Graph
    """
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
        G.node[i]['loc'] = (random.random(),random.random())

    dist = {}
    L = 0
    for u in range(n):
        print(u)
        for v in range(u+1,n):
            dist = euclid_distance(G,u,v)
            if dist > L:
                L = dist

    for u in range(n):
        print(u)
        for v in range(u+1,n):
            p_edge = alpha*(e**(-euclid_distance(G,u,v)/(beta*L)))
            if random.random() < p_edge:
                G.add_edge(u,v)

    cc = nx.connected_components(G)

    print(len(cc))
    for c in range(len(cc)-1):
        u = random.choice(cc[c])
        v = random.choice(cc[c+1])
        G.add_edge(u,v)

    return G
        
def inet(n,alpha,tau):
    """Creates a random graph according to the INET model.

    Proceeds by giving a each node a specificed degree proportional to
    a power law with exponent alpha. It then takes the top tau nodes (by
    degree) and makes a complete graph. It then proceeds by using 25% of each
    of the top tau nodes connections to connect to degree two nodes.

    It then takes each unconnected node and selects a connection uniformly
    at random into the existing network. Then each nodes edges are exhausted
    connecting into the existing network.

    Parameters:
    -----------
    n : int
        Size of network
    alpha: numeric
           Parameter of degree distribution
    tau : numeric
          Core network size

    Returns:
    --------
    G:Graph
    """
    G= nx.MultiGraph()
    degree = {}
    full_nodes = []
    connected_nodes = []
    unconnected_nodes = range(n)
    sum_deg = 0
    
    for i in range(n):
        G.add_node(i)
        degree[i] = rand_pl(alpha,1)
        sum_deg += degree[i]

    
    deg_sort = sorted([(degree[i],i) for i in range(n)],reverse=True)
    top_tau = [deg_sort[i][1] for i in range(tau)]

    for i in range(tau):
        connected_nodes.append(top_tau[i])
        unconnected_nodes.remove(top_tau[i])
        degree[top_tau[i]] -= (tau-1)
        for j in range(i+1,tau):
            G.add_edge(top_tau[i],top_tau[j])
            sum_deg -= 2


    deg_two_nodes = [i for i in range(n) if degree[i] == 2]
    
    for t in top_tau:
        for j in range(int(degree[t]*0.25)):
            try:
                x = random.choice(deg_two_nodes)
            except:
                break
            G.add_edge(t,x)
            deg_two_nodes.remove(x)
            degree[t] -= 1
            degree[x] -= 1
            sum_deg -= 2
            connected_nodes.append(x)
            unconnected_nodes.remove(x)
        
    while not (unconnected_nodes == []):
        u = random.choice(unconnected_nodes)
        v = random.choice(connected_nodes)
        if not (degree[v]==0):
            G.add_edge(u,v)
            connected_nodes.append(u)
            unconnected_nodes.remove(u)
            degree[u] -= 1
            degree[v] -= 1
            sum_deg -= 2
            if degree[u] == 0:
                connected_nodes.remove(u)
                full_nodes.append(u)
            if degree[v] == 0:
                connected_nodes.remove(v)
                full_nodes.append(v)

    num_repeats = 0
    while not (connected_nodes == []):
        if len(connected_nodes) % 1 == 0:
            print(len(connected_nodes))
        u = random.choice(connected_nodes)
        v = random.choice(connected_nodes)
        #print(connected_nodes)
        #print(G.edges(connected_nodes))
        if not(u==v) and not G.has_edge(u,v):
            sum_deg -= 2
            G.add_edge(u,v)
            degree[v] -= 1
            degree[u] -= 1
            if degree[u] == 0:
                connected_nodes.remove(u)
                full_nodes.append(u)
            if degree[v] == 0:
                connected_nodes.remove(v)
                full_nodes.append(v)
        elif (u==v) and len(connected_nodes) ==1:
            G.add_edge(u,v)
            degree[u] -= 2
            connected_nodes.remove(u)
            full_nodes.append(u)
            sum_deg -= 2
        elif G.has_edge(u,v) and num_repeats < 10: # This is definitely a hack
            num_repeats += 1
        elif G.has_edge(u,v) and num_repeats >= 10:
            num_repeats = 0
            G.add_edge(u,v)
            degree[v] -= 1
            degree[u] -= 1
            sum_deg -= 2
            if degree[u] == 0:
                connected_nodes.remove(u)
                full_nodes.append(u)
            if degree[v] == 0:
                connected_nodes.remove(v)
                full_nodes.append(v)
    return G
    
    
def euclid_distance(G,i,j):
    xi = G.node[i]['loc'][0]
    yi = G.node[i]['loc'][1]
    xj = G.node[j]['loc'][0]
    yj = G.node[j]['loc'][1]
    return ((xi-xj)**2 + (yi-yj)**2)**0.5
