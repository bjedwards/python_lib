import networkx as nx
from . import ASNet
import cPickle
import os
import glob

def save_graphs(graphDict,pathname='./'):
    """Saves a dictionary of graphs to pathname"""
    for G in graphDict:
        f = open(pathname + str(G) +'.pickle','w')
        cPickle.dump(graphDict[G],f)
        f.close()

def load_graphs(files=[],pathname=None):
    """Loads a list of filenames which refer to pickled graph objects"""
    if not pathname is None:
        for infile in glob.glob(os.path.join(pathname,'*.pickle')):
            files.append(infile)

    models = {}
    
    for infile in files:
        f = open(infile,'r')
        m_name = infile[infile.rfind('/') + 1:].rstrip('.pickle')
        models[m_name] = cPickle.load(f)
        f.close()
    return models

def load_BGP(dates = range(200205,201105,100),
             filename='../../data/internet/newData.pickle'):
    """Loads BGP data"""
    
    graphDict = {}
    f = open(filename,'r')
    new_data = cPickle.load(f)
    ASData = ASNet.ASNet(graph=new_data)
    for i in dates:
        graphDict[str(i)] = ASData.getSnapShot(i)
    return graphDict
