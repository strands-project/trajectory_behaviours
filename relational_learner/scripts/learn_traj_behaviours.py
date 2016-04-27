#!/usr/bin/env python

"""compute_spatial_relations.py: Computes spatial relations."""

__author__      = "Paul Duckworth"
__copyright__   = "Copyright 2015, University of Leeds"

import rospy
import pymongo
import os, sys, time
import logging
import argparse
import itertools
import getpass

import numpy as np
from scipy import spatial
import cPickle as pickle

from geometry_msgs.msg import Pose, Quaternion
from human_trajectory.msg import Trajectory, Trajectories
from soma_trajectory.srv import TrajectoryQuery, TrajectoryQueryRequest, TrajectoryQueryResponse
from soma_geospatial_store.geospatial_store import *

import novelTrajectories.config_utils as util
import relational_learner.obtain_trajectories as ot
from relational_learner.graphs_handler import *
from novelTrajectories.traj_data_reader import *
from relational_learner.learningArea import *


def Mongodb_to_list(res):
    """Convert the EpisodeMsg into a list of episodes
       episodeMsg[]
           string obj1
           string obj1_type
           string obj2
           string obj2_type
           string spatial_relation
           int32 start_frame
           int32 end_frame
    """
    
    ep_list = []
    for i in res:
        ep = (str(i["obj1"]), str(i["obj1_type"]), str(i["obj2"]), \
              str(i["obj2_type"]), str(i["spatial_relation"]), \
              int(i["start_frame"]), int(i["end_frame"]))
        ep_list.append(ep)
    return ep_list


def run_all(turn_on_plotting=False):

    (directories, config_path, input_data, date) = util.get_learning_config()
    (data_dir, qsr, eps, activity_graph_dir, learning_area) = directories
    (soma_map, soma_config) = util.get_map_config(config_path)
    gs = GeoSpatialStoreProxy('geospatial_store','soma')
    msg_store = GeoSpatialStoreProxy('message_store', 'relational_episodes')


    #*******************************************************************#
    #                  Regions of Interest Knowledge                    #
    #*******************************************************************#
    rospy.loginfo('Getting Region Knowledge from roslog...') 
    roi_knowledge, roi_temp_list = region_knowledge(soma_map, soma_config, \
                                                 sampling_rate=10, plot=turn_on_plotting)
   
  
    #*******************************************************************#
    #                  Obtain Episodes in ROI                           #
    #*******************************************************************#
    rospy.loginfo("0. Running ROI query from message_store")   
    for roi in gs.roi_ids(soma_map, soma_config):
        str_roi = "roi_%s" % roi
        #if roi != '12': continue

        print 'ROI: ', gs.type_of_roi(roi, soma_map, soma_config), roi
        query = {"soma_roi_id" : str(roi)} 

        res = msg_store.find(query)

        all_episodes = {}
        trajectory_times = []
        for trajectory in res:
            all_episodes[trajectory["uuid"]] = Mongodb_to_list(trajectory["episodes"])   
            trajectory_times.append(trajectory["start_time"])
        print "Number of Trajectories in mongodb = %s. \n" % len(all_episodes)
       
        if len(all_episodes) < 12:
            print "Not enough episodes in region %s to learn model. \n" % roi
            continue

        #**************************************************************#
        #            Activity Graphs/Code_book/Histograms              #
        #**************************************************************#
        rospy.loginfo('Generating Activity Graphs')
        
        params, tag = AG_setup(input_data, date, str_roi)
        print "INFO: ", params, tag, activity_graph_dir

        generate_graph_data(all_episodes, activity_graph_dir, params, tag)

        #**************************************************************#
        #           Generate Feature Space from Histograms             #
        #**************************************************************#     
        rospy.loginfo('Generating Feature Space')
        feature_space = generate_feature_space(activity_graph_dir, tag)
        
        (code_book, graphlet_book, X_source_U) = feature_space
        print "code_book length = ", len(code_book)

        #print ">>>>>CODEBOOK 1:"
        #print code_book[1]
        #print graphlet_book[1].graph
        
        #for cnt, i in enumerate(graphlet_book):
        #    print cnt,  i.graph, "\n"

        #**************************************************************#
        #                    Learn a Clustering model                  #
        #**************************************************************#
        rospy.loginfo('Learning on Feature Space')
        params, tag = AG_setup(input_data, date, str_roi)

        smartThing=Learning(f_space=feature_space, roi=str_roi, vis=False)
        smartThing.kmeans(k=None) #Can pass k, or auto selects min(penalty)


        #*******************************************************************#
        #                    Temporal Analysis                              #
        #*******************************************************************#
        rospy.loginfo('Learning Temporal Measures')
        #print "traj times = ", trajectory_times, "\n"
        smartThing.time_analysis(trajectory_times, plot=turn_on_plotting)

           
        #Add the region knowledge to smartThing
        try:
            smartThing.methods["roi_knowledge"] = roi_knowledge[roi]
            smartThing.methods["roi_temp_list"] = roi_temp_list[roi]
        except KeyError:
            smartThing.methods["roi_knowledge"] = 0
            smartThing.methods["roi_temp_list"] = [0]*24

        smartThing.save(learning_area)
        print "Learnt models for: "
        for key in smartThing.methods:
            print "    ", key

    print "COMPLETED LEARNING PHASE"
    return




class Offline_Learning(object):

    def learn(self, turn_on_plotting=True):
    	r = run_all(turn_on_plotting)
	


if __name__ == "__main__":
    rospy.init_node("trajectory_learner")
    o = Offline_Learning()
    o.learn()


    #Test:
    data_dir = '/home/strands/STRANDS/'
    file_ = os.path.join(data_dir + 'learning/roi_12_smartThing.p')
    print file_

    #smartThing=Learning(load_from_file=file_)
    #print smartThing.methods
    #print smartThing.code_book

