#!/usr/bin/env python
import rospy
import sys, os, getpass, time
import ConfigParser
import itertools
import cPickle as pickle
from datetime import datetime
from scipy.spatial import distance

from relational_learner.msg import *
from relational_learner.srv import *

import novelTrajectories.config_utils as util
import relational_learner.graphs_handler as gh
import relational_learner.learningArea as la
from time_analysis.cyclic_processes import *
from relational_learner.Activity_Graph import Activity_Graph
from mongodb_store.message_store import MessageStoreProxy

#from std_msgs.msg import Header



class Importer(object):
    def __init__(self):
        rospy.loginfo("Connecting to mongodb...")
        self._client = pymongo.MongoClient(rospy.get_param("mongodb_host"),
                                           rospy.get_param("mongodb_port"))
        self._store_client = MessageStoreProxy(collection="relational_learner")

def episodesMsg_to_list(req):
    """Convert the EpisodesMsg into a list of episodes
       EpisodesMsg: 
            std_msgs/Header header
            string soma_roi_id
            string soma_map
            string soma_config
            int64 start_time
            relational_learner/episodeMsg[] episodes
                string obj1
                string obj1_type
                string obj2
                string obj2_type
                string spatial_relation
                int32 start_frame
                int32 end_frame
    """
    ep_list = []
    for i in req.episodes.episodes:
        ep = (i.obj1, i.obj1_type, i.obj2, i.obj2_type, \
             i.spatial_relation, i.start_frame, i.end_frame)
        ep_list.append(ep)
    all_episodes = {req.episodes.uuid : ep_list}
    return all_episodes



def handle_novelty_detection(req):

    """1. Get data from EpisodesMsg"""
    t0=time.time()
    uuid = req.episodes.uuid
    print "\nUUID = ", uuid
    roi = req.episodes.soma_roi_id
    print "ROI = ", roi
    eps_soma_map = req.episodes.soma_map
    eps_soma_config = req.episodes.soma_config
    start_time = req.episodes.start_time
    all_episodes = episodesMsg_to_list(req)

    episodes_file = all_episodes.keys()[0]
    print "Length of Episodes = ", len(all_episodes[episodes_file])
 
    (directories, config_path, input_data, date) = util.get_learning_config()
    (data_dir, qsr, eps, graphs, learning_area) = directories
    #(data_dir, config_path, params, date) = util.get_qsr_config()
    (soma_map, soma_config) = util.get_map_config(config_path)
    
    if eps_soma_map != soma_map: raise ValueError("Config file soma_map not matching published episodes")
    if eps_soma_config != soma_config: raise ValueError("Config file soma_config not matching published episodes")

    params, tag = gh.AG_setup(input_data, date, roi)

    print "params = ", params
    print "tag = ", tag

    """4. Activity Graph"""
    ta0=time.time()

    activity_graphs = gh.generate_graph_data(all_episodes, data_dir, \
            params, tag, test=True)

    #print "\n  ACTIVITY GRAPH: \n", activity_graphs[episodes_file].graph 
    ta1=time.time()
    
    """5. Load spatial model"""
    print "\n  MODELS LOADED :"
    file_ = os.path.join(data_dir + 'learning/roi_' + roi + '_smartThing.p')
    smartThing=la.Learning(load_from_file=file_)
    if smartThing.flag == False: return NoveltyDetectionResponse()

    print "code book = ", smartThing.code_book

    """6. Create Feature Vector""" 
    test_histogram = activity_graphs[episodes_file].get_histogram(smartThing.code_book)
    print "HISTOGRAM = ", test_histogram
  

    """6.5 Upload data to Mongodb"""
    """activityGraphMsg:
            std_msgs/Header header
            string uuid
            string soma_roi_id
            string soma_map
            string soma_config
            int64[] codebook
            float32[] histogram
    """
    #header = req.episodes.trajectory.header
    #meta = {"map":'uob_library'}

    #tm0 = time.time()
    #ag =  activityGraphMsg(
    #            header=header, uuid=req.episodes.uuid, roi=roi, \
    #            histogram = test_histogram, codebook = smartThing.code_book, \
    #            episodes=get_episode_msg(ep.all_episodes[episodes_file]))
   
    #query = {"uuid" : str(uuid)} 
    #p_id = Importer()._store_client.update(message=ag, message_query=query,\
    #                                       meta=meta, upsert=True)
    #tm1 = time.time()

    """7. Calculate Distance to clusters"""
    estimator = smartThing.methods['kmeans']
    closest_cluster = estimator.predict(test_histogram)
    
    print "INERTIA = ", estimator.inertia_
    #print "CLUSTER CENTERS = ", estimator.cluster_centers_

    a = test_histogram
    b = estimator.cluster_centers_[closest_cluster]
    dst = distance.euclidean(a,b)
    print "\nDISTANCE = ", dst

    mean = estimator.cluster_dist_means[closest_cluster[0]]
    std = estimator.cluster_dist_std[closest_cluster[0]]
    print "Mean & std = ",  mean, std

    if dst > mean+std:
        print ">>> NOVEL1\n"
        dst=1.0
    elif dst > mean + 2*std:
        print ">>> NOVEL2\n"
        dst=2.0
    elif  dst > mean + 3*std:
        print ">>> NOVEL3\n"
        dst=3.0
    else:
        print ">>> not novel\n"
        dst=0.0


    """8. Time Analysis"""
    fitting = smartThing.methods['time_fitting']
    dyn_cl = smartThing.methods['time_dyn_clst']

    pc = dyn_cl.query_clusters(start_time%86400)
    pf = fitting.query_model(start_time%86400)
    
    print "PC = ", pc
    print "PF = ", pf

    """9. ROI Knowledge"""
    try:
        region_knowledge = smartThing.methods['roi_knowledge']
        temporal_knowledge = smartThing.methods['roi_temp_list']
        print "Region Knowledge Score = ", region_knowledge
        print "Hourly score = ", region_knowledge

        t = datetime.fromtimestamp(start_time)
        print "Date/Time = ", t
        th = temporal_knowledge[t.hour]
        print "Knowledge per hour = ", th

    except KeyError:
        print "No Region knowledge in `region_knowledge` db"
        th = 0
 
    print "\n Service took: ", time.time()-t0, "  secs."
    print "  AG took: ", ta1-ta0, "  secs."
    #print "  Mongo upload took: ", tm1-tm0, "  secs."

    return NoveltyDetectionResponse(dst, [pc, pf], th)


def calculate_novelty():
    rospy.init_node('novelty_server')
                        #service_name       #serive_type       #handler_function
    s = rospy.Service('/novelty_detection', NoveltyDetection, handle_novelty_detection)
    print "Ready to detect novelty..."
    rospy.spin()



if __name__ == "__main__":
    calculate_novelty()

