import networkx as nx
import cPickle

# Wrapper class for networkx graph, whois data, and computed statistics
class ASNet():
  
  measures = ['degree', 
	      'closeness_centrality',
	      'betweenness_centrality',
	      #'current_flow_closeness_centrality', #Memory overflow on 8gb
	      #'current_flow_betweenness_centrality',  #Memory overflow on 8gb
	      #'eigenvector_centrality_numpy', #Memory overflow on 8gb
	      'graph_clique_number',
	      'graph_number_of_cliques',
	      'clustering']

  # Loads graph from real world data, including whois data
  def __init__(self, graphFile = "../data/2002-2010", asim = False, whois = True, graph = None):
    self.dates = set([])
    if graph != None: 
      self.graph = graph
    else:
      self.graph = nx.Graph()
      whoisGraph = cPickle.load(open('pickle/whoisInfoGraph.pck', 'r'))
      for line in open(graphFile).readlines():
	w = line.split()
	if w[1] == 'mkAS': 
	  self.graph.add_node(int(w[2]))
	  if (not asim) and whois: self.graph.node[int(w[2])] =  whoisGraph.node[int(w[2])]
	  self.graph.node[int(w[2])]['startDate'] = int(w[0])
	elif w[1] == 'rmAS': self.graph.node[int(w[2])]['endDate'] = int(w[0])
	elif w[1] == 'mkASedge': self.graph.add_edge(int(w[2]), int(w[3]), startDate = int(w[0]))
	elif w[1] == 'rmASedge': self.graph.edge[int(w[2])][int(w[3])]['endDate'] = int(w[0])
      # Add enddate to anyone who hasn't gotten one
      if asim:
	for node in self.graph.nodes(): 
	  if not hasattr(self.graph.node[node], 'endDate'): self.graph.node[node]['endDate'] = max(self.dates)
	  for edge in self.graph.edge[node]:
	    if not hasattr(self.graph.edge[node][edge], 'endDate'): self.graph.edge[node][edge]['endDate'] = max(self.dates)
      print "Loaded ", graphFile, " successfully"
    
    for node in self.graph.nodes():
      self.dates.add(self.graph.node[node]['startDate'])
      self.dates.add(self.graph.node[node]['endDate'])
    for edge in self.graph.edges():
      self.dates.add(self.graph.edge[edge[0]][edge[1]]['startDate'])
      self.dates.add(self.graph.edge[edge[0]][edge[1]]['endDate'])
      
  # Computes the value for the function called the first time, then stores it for later calls       
  def __getattr__(self, attr):
    if attr in self.measures: setattr(self, attr, getattr(nx, attr)(self.graph))
    else: raise AttributeError
    return getattr(self, attr) 

  # Gets snapshot for date
  def getSnapShot(self, date):
    g = nx.Graph()
    gs = self.graph
    g.add_nodes_from([(n, gs.node[n]) for n in gs.nodes() if gs.node[n]['startDate'] <= date < gs.node[n]['endDate']])
    g.add_edges_from([(e[0], e[1], gs.edge[e[0]][e[1]]) for e in gs.edges() if gs.edge[e[0]][e[1]]['startDate'] <= date < gs.edge[e[0]][e[1]]['endDate']])
    g.remove_nodes_from([n for n in g.nodes() if g.degree(n) == 0])
    return g
