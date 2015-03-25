#!/usr/bin/env python

"""Activity Graph handler code"""

__author__      = "Paul Duckworth"
__copyright__   = "Copyright 2015, University of Leeds"

import rospy
import os, sys, time
import itertools
import cPickle as pickle
import numpy as np
from relational_learner.Activity_Graph import Activity_Graph



def AG_setup(input_data, date, roi):
    params_str = (input_data['MIN_ROWS'], input_data['MAX_ROWS'], input_data['MAX_EPI'], input_data['num_cores'])
    params = []
    for x in params_str:
        params.append(int(x)) if x != 'None' else params.append(None)

    params_tag = map(str, params)
    params_tag = '_'.join(params_tag)
    tag = roi +'_'+ params_tag + date

    return params, tag


#**************************************************************#
#      Create Activity Graphs For Each Trajectory Instance     #
#**************************************************************#

def generate_graph_data(episodes, data_dir, params, tag,
                         __out=False, test=False):

    cnt=0
    activity_graphs = {}

    for episodes_file in episodes:  

        uuid, start, end = episodes_file.split('__')        
        if __out: rospy.loginfo('Processing for graphlets: ' + episodes_file)

        episodes_dict = episodes[episodes_file]
        episodes_list = list(itertools.chain.from_iterable(episodes_dict.values()))

        activity_graphs[episodes_file] = Activity_Graph(episodes_list, params)
        activity_graphs[episodes_file].get_valid_graphlets()

        if __out and cnt == 0: graph_check(activity_graphs, episodes_file) #print details of one activity graph
        cnt+=1
        if __out: print cnt
    
    if test: 
        return activity_graphs
    else:
        AG_out_file = os.path.join(data_dir + 'activity_graphs_' + tag + '.p')
        pickle.dump(activity_graphs, open(AG_out_file,'w')) 
        rospy.loginfo('4. Activity Graph Data Generated and saved to:\n' + AG_out_file) 
    return



def graph_check(gr, ep_file):
    """Prints to /tmp lots """
    gr[ep_file].graph2dot('/tmp/act_gr.dot', False)
    os.system('dot -Tpng /tmp/act_gr.dot -o /tmp/act_gr.png')
    print "graph: " + repr(ep_file)
    print gr[ep_file].graph

    gr2 = gr[ep_file].valid_graphlets
    for cnt_, i in enumerate(gr2[gr2.keys()[0]].values()):
        i.graph2dot('/tmp/graphlet.dot', False) 
        cmd = 'dot -Tpng /tmp/graphlet.dot -o /tmp/graphlet_%s.png' % cnt_
        os.system(cmd)



def generate_feature_space(data_dir, tag, __out=False):
    
    AG_out_file = os.path.join(data_dir + 'activity_graphs_' + tag + '.p')
    activity_graphs = pickle.load(open(AG_out_file))
    
    rospy.loginfo('5. Generating codebook')
    code_book, graphlet_book = [], []
    code_book_set, graphlet_book_set = set([]), set([])
    for episodes_file in activity_graphs: 

        #for window in activity_graphs[episodes_file].graphlet_hash_cnts:     #Loop through windows, if multiple windows
        window = activity_graphs[episodes_file].graphlet_hash_cnts.keys()[0]

        for ghash in activity_graphs[episodes_file].graphlet_hash_cnts[window]:
            if ghash not in code_book_set:
                code_book_set.add(ghash)
                graphlet_book_set.add(activity_graphs[episodes_file].valid_graphlets[window][ghash])
    code_book.extend(code_book_set)
    graphlet_book.extend(graphlet_book_set)   

    print "len of code book: " + repr(len(code_book))
    if len(code_book) != len(graphlet_book): 
        print "BOOK OF HASHES DOES NOT EQUAL BOOK OF ACTIVITY GRAPHS"
        sys.exit(1)

    rospy.loginfo('5. Generating codebook FINISHED')

    rospy.loginfo('5. Generating features')
    cnt = 0
    X_source_U = []
    #Histograms are Windowed dictionaries of histograms 
    for episodes_file in activity_graphs:
        if __out: print cnt, episodes_file
        histogram = activity_graphs[episodes_file].get_histogram(code_book)
        X_source_U.append(histogram)
        cnt+=1
        if __out and cnt ==1:
            key = activity_graphs[episodes_file].graphlet_hash_cnts.keys()[0]
            print "KEY = " + repr(key)
            print "hash counts: " + repr(activity_graphs[episodes_file].graphlet_hash_cnts[key].values())
            print "sum of hash counts: " + repr(sum(activity_graphs[episodes_file].graphlet_hash_cnts[key].values()))
            print "sum of histogram: " + repr(sum(histogram))
    
    rospy.loginfo('Generating features FINISHED')
    rospy.loginfo('Saving all experiment data')       
    
    feature_space = (code_book, graphlet_book, X_source_U)

    feature_space_out_file = os.path.join(data_dir + 'feature_space_' + tag + '.p')
    pickle.dump(feature_space, open(feature_space_out_file, 'w'))
    print "\nall graph and histogram data written to: \n" + repr(data_dir) 
    
    rospy.loginfo('Done')
    return feature_space


