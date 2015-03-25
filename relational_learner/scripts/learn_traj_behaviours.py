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

import relational_learner.obtain_trajectories as ot
from relational_learner.graphs_handler import *
from novelTrajectories.traj_data_reader import *
from relational_learner.learningArea import Learning

def check_dir(directory):
    if not os.path.isdir(directory):
        os.system('mkdir -p ' + directory)
    return


def qsr_setup(data_dir, params, date):
    params_tag = map(str, params)
    params_tag = '__'.join(params_tag)
    qsr_tag = params_tag + date

    qsr_dir = os.path.join(data_dir, 'qsr_dump/')
    check_dir(qsr_dir)
    return qsr_dir, qsr_tag




def run_all():
    """
       Change this to have multiple methods
       each method can thnn me interupted in OfflineLearning.action
    """
    global __out

    user = getpass.getuser()
    base_data_dir = os.path.join('/home/' + user + '/STRANDS/')
    check_dir(base_data_dir)
    check_dir(os.path.join('/home/' + user + '/STRANDS/qsr_dump/'))
    check_dir(os.path.join('/home/' + user + '/STRANDS/episodes_dump/'))
    check_dir(os.path.join('/home/' + user + '/STRANDS/AG_graphs/'))
    learning_area = os.path.join(base_data_dir, 'learning/')
    check_dir(learning_area)

    path = os.path.dirname(os.path.realpath(__file__))
    if path.endswith("/scripts"): 
        config_path = path.replace("/scripts", "/config.ini") 
    config_parser = ConfigParser.SafeConfigParser()
    print(config_parser.read(config_path))

    if len(config_parser.read(config_path)) == 0:
        raise ValueError("Options file not found, please provide a config.ini file as described in the documentation")
    config_section = "activity_graph_options"
    try:
        input_data={}
        date = config_parser.get(config_section, "date")
        input_data['MAX_ROWS'] = config_parser.get(config_section, "MAX_ROWS")
        input_data['MIN_ROWS'] = config_parser.get(config_section, "MIN_ROWS")
        input_data['MAX_EPI']  = config_parser.get(config_section, "MAX_EPI")
        input_data['num_cores'] = config_parser.get(config_section, "num_cores")
        #print input_data
    except ConfigParser.NoOptionError:
        raise    

    soma_map = 'uob_library'
    soma_config = 'uob_lib_conf'
    gs = GeoSpatialStoreProxy('geospatial_store','soma')
    msg = GeoSpatialStoreProxy('message_store', 'soma')
    rospy.loginfo("0. Running ROI query from geospatial_store")   
    
  
    #*******************************************************************#
    #             Obtain ROI, Objects and Trajectories                  #
    #*******************************************************************#
    __out = False
    roi_timepoints = {}
    roi_cnt = 0
    two_proxies = TwoProxies(gs, msg, soma_map, soma_config)
    for roi in gs.roi_ids(soma_map, soma_config):

        str_roi = "roi_%s" % roi
        if roi != '12': continue

        if __out: print 'ROI: ', gs.type_of_roi(roi, soma_map, soma_config), roi

        objects = two_proxies.roi_objects(roi)
        if objects == None: continue

        geom = two_proxies.gs.geom_of_roi(str(roi), soma_map, soma_config)
        if __out: print "  Number of objects in region =  " + repr(len(objects))
        if __out: print "  geometry of region= ", geom
	
        query = '''{"loc": { "$geoWithin": { "$geometry": 
        { "type" : "Polygon", "coordinates" : %s }}}}''' %geom['coordinates']
        if __out: print query
        #query = '''{"loc": { "$geoIntersects": { "$geometry": 
        #{ "type" : "Polygon", "coordinates" : %s }}}}''' %geom['coordinates']
        q = ot.query_trajectories(query)
        #roi_timepoints[str_roi] = q.trajectory_times #for Eris and debug
 
        if len(q.trajs)==0:
            print "No Trajectories in this Region"            
            continue
        else:
            print " number of unique traj returned = " + repr(len(q.trajs))

        #objects_per_trajectory = ot.trajectory_object_dist(objects, trajectory_poses)

        #LandMarks instead of Objects - need to select per ROI:
        #all_poses = list(itertools.chain.from_iterable(trajectory_poses.trajs.values()))
        #if __out: print "number of poses in total = " +repr(len(all_poses))
        #landmarks = ot.select_landmark_poses(all_poses)
        #pins = ot.Landmarks(landmarks)
        #if __out: print "landmark poses = " + repr(landmarks)
        #if __out: print pins.poses_landmarks

        #"""TO PLAY WITH LANDMARKS INSTEAD OF OBJECTS"""
        #object_poses = objects.all_objects
        #object_poses = pins.poses_landmarks


    #**************************************************************#
    #          Apply QSR Lib to Objects and Trajectories           #
    #**************************************************************#
    #Dependant on trajectories and objects
        __out = False
        rospy.loginfo('2. Apply QSR Lib')
        if __out: raw_input("Press enter to continue")

        reader = Trajectory_Data_Reader(config_filename = config_path, roi=str_roi)
        keeper = Trajectory_QSR_Keeper(objects=objects, 
                            trajectories=q.trajs, reader=reader)
        keeper.save(base_data_dir)
        #load_qsrs = 'roi_12_qsrs_qtcb__0_01__False__True__03_03_2015.p'
        #keeper= Trajectory_QSR_Keeper(reader=reader, load_from_file = load_qsrs, dir=base_data_dir) 

        #print keeper.reader.spatial_relations['7d638405-b2f8-55ce-b593-efa8e3f2ff2e'].trace[1].qsrs['Printer (photocopier)_5,trajectory'].qsr

    
    #**************************************************************#
    #             Generate Episodes from QSR Data                  #
    #**************************************************************#
    #Dependant on QSRs 
        __out = False
        rospy.loginfo('3. Generating Episodes')
        if __out: raw_input("Press enter to continue")

        ep = Episodes(reader=keeper.reader)
        ep.get_episodes(noise_thres=3, out=__out)
        ep.save(base_data_dir)
        if __out: print "episode test: " + repr(ep.all_episodes['5c02e156-493d-55bc-ad21-a4be1d9f95aa__1__22'])


    #**************************************************************#
    #            Activity Graphs/Code_book/Histograms              #
    #**************************************************************#
    #Dependant on Episodes
        __out = False
        rospy.loginfo('4. Generating Activity Graphs')
        if __out: raw_input("Press enter to continue")

        params, tag = AG_setup(input_data, date, str_roi)
        activity_graph_dir = os.path.join(base_data_dir, 'AG_graphs/')
        if __out: print params, tag, activity_graph_dir

        generate_graph_data(ep.all_episodes, activity_graph_dir, params, tag)
        if __out: print "Activity Graphs Done"


    #**************************************************************#
    #           Generate Feature Space from Histograms             #
    #**************************************************************#     
        __out = False
        rospy.loginfo('5. Generating Feature Space')
        if __out: raw_input("Press enter to continue")
        #feature_space is now a tuple
        feature_space = generate_feature_space(activity_graph_dir, tag)
        
    #**************************************************************#
    #                    Learn a Clustering model                  #
    #**************************************************************#
        __out = False
        rospy.loginfo('6. Learning on Feature Space')
        params, tag = AG_setup(input_data, date, str_roi)

        smartThing=Learning(f_space=feature_space, roi=str_roi, vis=__out)
        smartThing.kmeans(k=2) #Can pass k, or auto selects min(penalty)
        

    #*******************************************************************#
    #                    Region Knowledge                               #
    #*******************************************************************#
        #Only learn ROI Knowledge once for all regions. 
        if roi_cnt==0: 
            smartThing.region_knowledge(soma_map, soma_config)
            roi_knowledge = smartThing.methods["roi_knowledge"]
        else:
            smartThing.methods["roi_knowledge"] = roi_knowledge


    #*******************************************************************#
    #                    Temporal Analysis                              #
    #*******************************************************************#
        smartThing.time_analysis(q.trajectory_times)
   
        #print roi_knowledge[roi]
        #smartThing.time_plot(q.trajectory_times, roi_knowledge[roi], vis=True)
        
        smartThing.save(learning_area)
        print "Learnt models for: "
        for key in smartThing.methods:
            print "    ", key
            #print smartThing.methods[key]
  
        roi_cnt+=1

    return




class Offline_Learning(object):

    def learn(self):
    	r = run_all()
	







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

