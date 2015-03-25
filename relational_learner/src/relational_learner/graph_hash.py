#!/usr/bin/env python

"""graph_hash.py: Computes graph hash for entire graph."""

__author__      = "Krishna Dubba"
__copyright__   = "Copyright 2014, University of Leeds"

from igraph import Graph
import warnings

def graph_hash(G, node_name_attribute='name', edge_name_attribute=None):
    """
    See Figure 4 in 'kLog: A Language for Logical and Relational Learning with Kernels'
    for the algorithm.
    
    Takes an igraph graph, node_name attribute and edge_name_attribute. Note that
    edge_name_attribute is optional i.e. for graphs without edge labels or to ignore edge labels, 
    edge_name_attribute is None.
    """
    
    #suppress Runtime Warnings regarding not being able to find a path through the graphs
    warnings.filterwarnings('ignore')
    actG = G
    G = Graph.copy(actG.graph)

    for node in G.vs:
        paths = G.get_shortest_paths(node)
        node_hashes = []
        for path in paths:
            if len(path) != 0:
                node_name = G.vs[path[-1]][node_name_attribute]
                if node_name == None:
                    node_name = repr(None)
                node_hashes.append((len(path), node_name))
        node_hashes.sort()
        node_hashes_string = ':'.join([repr(i) for i in node_hashes])
        node['hash_name'] = hash(node_hashes_string)
    warnings.filterwarnings('always')    
    edge_hashes = []
    if edge_name_attribute:
        edge_hashes = [(G.vs[edge.source]['hash_name'], G.vs[edge.target]['hash_name'], \
                                   edge[edge_name_attribute]) for edge in G.es]
    else:
        edge_hashes = [(G.vs[edge.source]['hash_name'], G.vs[edge.target]['hash_name']) \
                       for edge in G.es]
    edge_hashes.sort()
    edge_hashes_string = ':'.join([repr(i) for i in edge_hashes])
    return (hash(edge_hashes_string), actG)

if __name__ == '__main__':
    import itertools
    import time
    import cPickle as pickle
    
    start = time.asctime()
    for i in xrange(1):
        print i
        g = Graph([(0,1), (0,2), (2,3), (3,4), (4,2), (2,5), (5,0), (6,3), (5,6)],directed=True)
        # Give some names to vertices
        g.vs["name"] = ["Alice", "Bob", "Claire", "Dennis", "Esther", "Frank", "George"]
        # Give some names to edges. Note edge names are optional
        g.es["name"] = ["A", "B", "C", "D", "E", "F", "G", "H", "K"]
        #a = graph_hash(g, "name", "name")
    print '==========DONE========='
    print start
    print time.asctime()        
    g1 = pickle.load(open('/usr/not-backed-up/Dropbox/code/my_code/trunk/python/redvine/code_book.p'))    
    g2 = pickle.load(open('/usr/not-backed-up/Dropbox/code/my_code/trunk/python/redvine/graphlet.p'))
    print graph_hash(g1, "name")
    print '####################################'
    print graph_hash(g2, "name")
    print g1
    
    from generate_activity_graph_from_file import Activity_Graph
    
    episodes_file = '/usr/not-backed-up/data_sets/race/cornell_human_activities/CAD_120/relational_data/Subject5__having_meal__eating__0511141231__episodes.p'
    episodes_dict = pickle.load(open(episodes_file))
    episodes      = list(itertools.chain.from_iterable(episodes_dict.values()))
    
    print '####################################'
    activity_graphs = Activity_Graph(episodes, COLLAPSE_TEMPORAL_NODES=True)
    print time.asctime()
    valid_graphlets = activity_graphs.get_valid_graphlets(min_rows=None, max_rows=3)
    print time.asctime()
    
