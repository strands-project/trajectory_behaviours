#!/usr/bin/env python

#__author__      = "Paul Duckworth"
#__copyright__   = "Copyright 2015, University of Leeds"

import rospy
import sys, os, getpass, time
import ConfigParser
import itertools
from datetime import datetime

from soma_geospatial_store.geospatial_store import * 
import relational_learner.obtain_trajectories as ot
import novelTrajectories.traj_data_reader as tdr
import novelTrajectories.config_utils as util

from relational_learner.msg import *
from relational_learner.srv import *
from mongodb_store.message_store import MessageStoreProxy

#from std_msgs.msg import Header

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




class Importer(object):
    def __init__(self):
        rospy.loginfo("Connecting to mongodb...")
        self._client = pymongo.MongoClient(rospy.get_param("mongodb_host"),
                                           rospy.get_param("mongodb_port"))
        self._store_client = MessageStoreProxy(collection="relational_episodes")


def get_episode_msg(all_episodes):
    episodes = []
            
    for key, episode_list in all_episodes.items():
        #print key
        for cnt, ep in enumerate(episode_list):
            #print cnt, ep
            msg = episodeMsg()
            msg.obj1, msg.obj1_type, msg.obj2, msg.obj2_type, \
            msg.spatial_relation, msg.start_frame, msg.end_frame = ep[0:7]

            episodes.append(msg)
    return episodes



def get_poses(trajectory_message):
    traj = []    
    for entry in trajectory_message.trajectory:
        x=entry.pose.position.x
        y=entry.pose.position.y
        z=entry.pose.position.z
        traj.append((x,y,z))
    return traj



def handle_episodes(req):
    """     1. Take trajectory as input
            2. Query mongodb for the region and objects
            3. Pass to QSRLib data parser
            4. Generate Episodes
    """

    t0=time.time()

    """1. Trajectory Message"""
    uuid = req.trajectory.uuid
    start_time = req.trajectory.start_time.secs
    print "\n1. Analysing trajectory: %s" %uuid

    (data_dir, config_path) = util.get_path()
    (soma_map, soma_config) = util.get_map_config(config_path)
    
    trajectory_poses = {uuid : get_poses(req.trajectory)}
    print "LENGTH of Trajectory: ", len(trajectory_poses[uuid])

    """2. Region and Objects"""  
    gs = GeoSpatialStoreProxy('geospatial_store', 'soma')
    msg_store = GeoSpatialStoreProxy('message_store', 'soma')
    two_proxies = TwoProxies(gs, msg_store, soma_map, soma_config)

    roi = two_proxies.trajectory_roi(req.trajectory.uuid, trajectory_poses[uuid])
    if roi == None: return EpisodeServiceResponse(uuid=uuid)
        
    objects = two_proxies.roi_objects(roi)
    print "\nROI: ", roi
    #print "\n  Objects: ", objects
    if objects == None: return EpisodeServiceResponse(uuid=uuid)
    
    """2.5 Get the closest objects to the trajectory"""
    closest_objs_to_trajs = ot.trajectory_object_dist(objects, trajectory_poses)

    """3. QSRLib data parser"""#
    tq0=time.time()
    qsr_reader = tdr.Trajectory_Data_Reader(objects=objects, \
                                        trajectories=trajectory_poses, \
                                        objs_to_traj_map = closest_objs_to_trajs, \
                                        roi=roi)

    tr = qsr_reader.spatial_relations[uuid].trace

    #for i in tr:
    #   print tr[i].qsrs['Printer (photocopier)_2,trajectory'].qsr

    """3.5 Check the uuid is new (or stitch QSRs together)"""
    stitching.merge_qsr_worlds(uuid, qsr_reader)
    tq1=time.time()


    """4. Episodes"""
    te0=time.time()
    ep = tdr.Episodes(reader=qsr_reader)
    te1=time.time()
    #print "\n  ALL EPISODES :"
    #for t in ep.all_episodes:
    #    for o in ep.all_episodes[t]:
    #        print ep.all_episodes[t][o]
        
    episodes_file = ep.all_episodes.keys()[0] #This gives the ID of the Trajectory
    uuid, start, end = episodes_file.split('__')  #Appends the start and end frame #
    print episodes_file

    """6.5 Upload data to Mongodb"""
    tm0 = time.time()
    h = req.trajectory.header
    meta = {}

    msg = episodesMsg(header=h, uuid=uuid, soma_roi_id=str(roi),  soma_map=soma_map, \
                    soma_config=soma_config, start_time=start_time, \
                    episodes=get_episode_msg(ep.all_episodes[episodes_file]))

    #MongoDB Query - to see whether to insert new document, or update an existing doc.
    query = {"uuid" : str(uuid)} 
    p_id = Importer()._store_client.update(message=msg, message_query=query, meta=meta, upsert=True)

    tm1 = time.time()

    print "\nService took: ", time.time()-t0, "  secs."
    print "  Data Reader took: ", tq1-tq0, "  secs."
    print "  Episodes took: ", te1-te0, "  secs."
    print "  Mongo upload took: ", tm1-tm0, "  secs."

    return EpisodeServiceResponse(msg.header, msg.uuid, msg.soma_roi_id, msg.soma_map, \
                                  msg.soma_config, msg.start_time, msg.episodes)


def generate_episodes():
    
    rospy.init_node('episode_server')
                        #service_name      #service_type   #handler_function
    s = rospy.Service('/episode_service', EpisodeService, handle_episodes)
    print "Ready to service some episodes..."
    rospy.spin()



if __name__ == "__main__":
    stitching = stitch_uuids()
    generate_episodes()

