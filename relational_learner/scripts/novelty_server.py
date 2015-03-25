#!/usr/bin/env python
import sys
import os
import rospy
import time
import getpass
import ConfigParser
from datetime import datetime
import itertools
from scipy.spatial import distance
import cPickle as pickle
from relational_learner.srv import *
from soma_geospatial_store.geospatial_store import * 
from relational_learner.Activity_Graph import Activity_Graph
import relational_learner.obtain_trajectories as ot
import novelTrajectories.traj_data_reader as tdr
import relational_learner.graphs_handler as gh
import relational_learner.learningArea as la
from time_analysis.cyclic_processes import *


class stitch_uuids(object):
    def __init__(self):
        self.uuid = ""
        self.stored_qsrs = []
        self.all_uuids = []

    def merge_qsr_worlds(self, uuid, data_reader):
        """Merges together trajectories which are published 
           with the same UUID. Merge after QSRs have been generated."""
        #If new UUID        
        if self.uuid != uuid:
            print "NEW ID"
            self.uuid = uuid
            #Initial QSRs of tractories:
            self.stored_qsrs = data_reader.spatial_relations[uuid].trace
            self.all_uuids.append(uuid)
            return data_reader
        #If same as previous UUID
        else:
            print "NOTE: QSRs stitched together"
            #print "NEW = ", reader.spatial_relations[uuid].trace
            #print "STORED = ", self.stored_qsrs
            len_of_stored = len(self.stored_qsrs)
            #print len_of_stored
            for (key,pose) in data_reader.spatial_relations[uuid].trace.items():
                self.stored_qsrs[key+len_of_stored] = pose
            #print self.stored_qsrs  #QSRs stitched together
            data_reader.spatial_relations[uuid].trace = self.stored_qsrs
            return data_reader



def get_poses(trajectory_message):
    traj = []    
    for entry in trajectory_message.trajectory:
        x=entry.pose.position.x
        y=entry.pose.position.y
        z=entry.pose.position.z
        traj.append((x,y,z))
    return traj

def handle_novelty_detection(req):
    """     1. Take the trajectory as input
            2. Query mongodb for the region and objects
            3. Pass to strands to QSRLib data parser
            4. Generate Activity Graph from Episodes()
            5. Load code_book and clustering centers
            6. Generate Feature
            7. Calculate distance to cluster centers
            8. Give a score based on distance (divided by number of clusters?)
    """
    t0=time.time()
    user = getpass.getuser()
    data_dir = os.path.join('/home/' + user + '/STRANDS/')

    path = os.path.dirname(os.path.realpath(__file__))
    if path.endswith("/scripts"): 
        config_path = path.replace("/scripts", "/config.ini") 
    config_parser = ConfigParser.SafeConfigParser()
    print(config_parser.read(config_path))

    if len(config_parser.read(config_path)) == 0:
        raise ValueError("Config file not found, please provide a config.ini file as described in the documentation")
    config_section = "soma" 
    try:
 
        soma_map = config_parser.get(config_section, "soma_map")
        soma_config = config_parser.get(config_section, "soma_config")
    except ConfigParser.NoOptionError:
         raise  

    """1. Trajectory Message"""
    uuid = req.trajectory.uuid
    start_time = req.trajectory.start_time.secs
    print "1. Analysing trajectory: %s" %uuid

    trajectory_poses = {uuid : get_poses(req.trajectory)}
    print "LENGTH of Trajectory: ", len(trajectory_poses[uuid])

    """2. Region and Objects"""  
    gs = GeoSpatialStoreProxy('geospatial_store', 'soma')
    msg = GeoSpatialStoreProxy('message_store', 'soma')
    two_proxies = TwoProxies(gs, msg, soma_map, soma_config)

    roi = two_proxies.trajectory_roi(req.trajectory.uuid, trajectory_poses[uuid])
    objects = two_proxies.roi_objects(roi)
    print "\n  ROI: ", roi
    print "\n  Objects: ", objects

    """3. QSRLib data parser"""
    qsr_reader = tdr.Trajectory_Data_Reader(objects=objects, \
                                        trajectories=trajectory_poses, \
                                        config_filename=config_path, \
                                        roi=roi)

    tr = qsr_reader.spatial_relations[uuid].trace
    #for i in tr:
    #    print tr[i].qsrs['Printer (photocopier)_5,trajectory'].qsr
    #print tr

    """3.5 Check the uuid is new"""
    stitching.merge_qsr_worlds(uuid, qsr_reader)

    """4. Episodes"""
    ep = tdr.Episodes(reader=qsr_reader)
    ep.get_episodes(noise_thres=3)
    print "\n  ALL EPISODES :"
    for t in ep.all_episodes:
        for o in ep.all_episodes[t]:
            print ep.all_episodes[t][o]

    """4. Activity Graph"""
    params = (None, 1, 3, 4)
    tag = 'None_1_3_4__19_02_2015'

    episodes_file = ep.all_episodes.keys()[0]
    uuid, start, end = episodes_file.split('__')        
    print episodes_file

    activity_graphs = gh.generate_graph_data(ep.all_episodes, data_dir, \
            params, tag, test=True)
    print "\n  ACTIVITY GRAPH: \n", activity_graphs[episodes_file].graph 


    """5. Load spatial model"""
    print "\n  MODELS LOADED :"
    file_ = os.path.join(data_dir + 'learning/roi_' + roi + '_smartThing.p')
    smartThing=la.Learning(load_from_file=file_)
    #print smartThing.methods
    print "code book = ", smartThing.code_book
    #print "\n  GRAPHLET: \n", smartThing.graphlet_book[0].graph
    
    """6. Create Feature Vector""" 
    test_histogram = activity_graphs[episodes_file].get_histogram(smartThing.code_book)
    print "HISTOGRAM = ", test_histogram
    
    """7. Calculate Distance to clusters"""
    estimator = smartThing.methods['kmeans']
    print "INERTIA = ", estimator.inertia_
    print "CLUSTER CENTERS = ", estimator.cluster_centers_
    
    closest_cluster = estimator.predict(test_histogram)
    a = test_histogram   
    b = estimator.cluster_centers_[closest_cluster]
    dst = distance.euclidean(a,b)
    print "\nDISTANCE = ", dst
 

    """8. Time Analysis"""
    fitting = smartThing.methods['time_fitting']
    dyn_cl = smartThing.methods['time_dyn_clst']

    model = fitting.models[np.argmin(fitting.bic)]
    pc = dyn_cl.query_clusters(start_time%86400)
    pf = fitting.query_model(start_time%86400)
    
    print "PC = ", pc
    print "PF = ", pf


    """9. ROI Knowledge"""
    knowledge = smartThing.methods['roi_knowledge'][roi]
    print "Region/Time Knowledge = ", knowledge
    t = datetime.fromtimestamp(start_time)
    th = knowledge[t.hour]
    print "knowledge score = ", th

    print "\n Service took: ", time.time()-t0, "  secs."
    return NoveltyDetectionResponse(dst, [pc, pf], th)


def calculate_novelty():
    
    rospy.init_node('novelty_server')

    #print stitching.uuid

                        #service_name       #serive_type       #handler_function
    s = rospy.Service('/novelty_detection', NoveltyDetection, handle_novelty_detection)
    print "Ready to detect novelty..."
    rospy.spin()



if __name__ == "__main__":
    stitching = stitch_uuids()
    calculate_novelty()

