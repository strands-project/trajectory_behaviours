#!/usr/bin/env python

"""Queries Mongodb for Object and Trajectory data"""

__author__      = "Paul Duckworth"
__copyright__   = "Copyright 2015, University of Leeds"


import rospy
import pymongo
import os, sys, time, copy
import logging
import itertools
import numpy as np
import cPickle as pickle
import random
import landmark_utils  as lu
from scipy import spatial

from interactive_markers.interactive_marker_server import *
from visualization_msgs.msg import *

from geometry_msgs.msg import Point,Pose
from human_trajectory.msg import Trajectory,Trajectories
from soma_trajectory.srv import TrajectoryQuery, TrajectoryQueryRequest, TrajectoryQueryResponse
from soma_geospatial_store.geospatial_store import GeoSpatialStoreProxy

#**************************************************************#
#             Obtain Objects and Trajectories                  #
#**************************************************************#

class query_objects():

    def __init__(self):
        self.all_objects = dict()
        host = rospy.get_param("mongodb_host")
        port = rospy.get_param("mongodb_port")
        self._client = pymongo.MongoClient(host, port)
        self._retrieve_logs()

    def _retrieve_logs(self):
        logs = self._client.message_store.soma.find()

        for log in logs:
            for i, _id in enumerate(log['id']):

                x = log['pose']['position']['x']
                y = log['pose']['position']['y']
                z = log['pose']['position']['z']  
                obj_instance = log['type'] + '_' + log['id']

                if _id not in self.all_objects:
                    self.all_objects[obj_instance] = (x,y,z)
                else:
                    self.all_objects[obj_instance] = (x,y,z)
        return


    def check(self):
        print 'All objects in SOMA:'
        for i, key in enumerate(self.all_objects):
            print repr(i) + ",  " + repr(key) + ",  " + repr(self.all_objects[key]) 
        print repr(len(self.all_objects)) + ' objects Loaded.\n'




class QueryClient():
    def __init__(self):
        service_name = 'trajectory_query'
        rospy.wait_for_service(service_name)
        self.ser = rospy.ServiceProxy(service_name, TrajectoryQuery)

    def query(self, query, vis = False):
        try:
            req = TrajectoryQueryRequest()
            req.query = query
            req.visualize = vis
            res = self.ser(req)
            return res
        except rospy.ServiceException, e:
            rospy.logerr("Service call failed: %s"%e)





def trajectory_object_dist(objects, trajectory_poses):
    uuids=trajectory_poses.keys()
    object_ids=objects.keys()

    print repr(len(uuids)) + " trajectories.  "+ repr(len(object_ids)) + " objects. Selecting closest 4..."

    object_distances={}
    distance_objects={}
    for (uuid, obj) in itertools.product(uuids, object_ids):
        #object_distances[(uuid, obj)] = [] #No need for list, if only taking init_pose
        #print (uuid, obj)

        traj_init_pose = trajectory_poses[uuid][0][:2]  #Select the first trajectory pose for now
        object_pose = objects[obj][0:2]                 #Objects only have one pose
        dist = spatial.distance.pdist([traj_init_pose, object_pose], 'euclidean')

        if uuid not in object_distances:
            object_distances[uuid] = {}
            distance_objects[uuid]={}

        object_distances[uuid][obj] = dist[0]
        distance_objects[uuid][dist[0]]= obj
        if len(object_distances[uuid]) != len(distance_objects[uuid]):
            print "multiple objects exactly the same distance from trajectory: " + repr(uuid)
            print "object: " + repr(obj)
            continue

    closest_objects = {}
    for uuid, dist_objs in distance_objects.items():
        keys = dist_objs.keys()
        keys.sort()

        #select closest 4 objects or landmarks
        closest_dists = keys[0:4]
        closest_objects[uuid]=[]
    
        for dist in closest_dists:
            closest_objects[uuid].append(dist_objs[dist]) 
        
    return closest_objects




class query_trajectories():

    def __init__(self, query):
        client = QueryClient()
        self.res = client.query(query, True)
        self.get_poses()

        if self.res.error:
            rospy.logerr("Result: error: %s (Invalid query: %s)" % (res.error, query))
        else:      
            print "Query returned: %s trajectories. " % repr(len(self.res.trajectories.trajectories))


    def get_poses(self):
        self.trajs = {}
        self.trajectory_times = []

        for trajectory in self.res.trajectories.trajectories:
            self.trajs[trajectory.uuid] = []          
            self.trajectory_times.append(trajectory.start_time.secs) # Temporal Info
            for entry in trajectory.trajectory:
                x=entry.pose.position.x
                y=entry.pose.position.y
                z=entry.pose.position.z
                self.trajs[trajectory.uuid].append((x,y,z))


def convert_keys_to_string(dictionary):
    """Recursively converts dictionary keys to strings."""
    if isinstance(dictionary, unicode):
        return str(dictionary)
    elif isinstance(dictionary, list):
        return dictionary
    return dict((str(k), convert_keys_to_string(v)) 
        for k, v in dictionary.items())
    
    
if __name__ == "__main__":
    global __out
    __out = True
    rospy.init_node("trajectory_obtainer")   
    rospy.loginfo("Running trajectoy/ROI query ")

    gs = GeoSpatialStoreProxy('geospatial_store','soma')
    soma_map = 'uob_library'
    soma_config = 'uob_lib_conf'
    cnt=0

    for roi in gs.roi_ids(soma_map, soma_config):
        cnt+=1
        print 'ROI: ', gs.type_of_roi(roi, soma_map, soma_config), roi
        geom = gs.geom_of_roi(roi, soma_map, soma_config)
        res = gs.objs_within_roi(geom, soma_map, soma_config)
        if res == None:
            print "No Objects in this Region"            
            continue
        objects_in_roi = {}
        for i in res:
            key = i['type'] +'_'+ i['soma_id']
            objects_in_roi[key] = i['loc']['coordinates']       
            print key, objects_in_roi[key]

        #geom_str = convert_keys_to_string(geom)
        query = '''{"loc": { "$geoWithin": { "$geometry": 
        { "type" : "Polygon", "coordinates" : %s }}}}''' %geom['coordinates']

        #Resource room 12
        #query ='''{"loc": { "$geoWithin": { "$geometry":
        #{ "type" : "Polygon", "coordinates" : [ [ 
        #            [ -0.0002246355582968818, 
        #              -2.519034444503632e-05],
        #            [ -0.0002241486476179944, 
        #             -7.42736662147081e-05], 
        #            [ -0.000258645873657315, 
        #              -7.284014769481928e-05],
        #            [ -0.0002555339747090102, 
        #              -2.521782172948406e-05],
        #            [ -0.0002246355582968818, 
        #              -2.519034444503632e-05]
        #            ] ] }}}}'''

        #Library 20
        #query ='''{"loc": { "$geoWithin": { "$geometry":
        #{ "type" : "Polygon", "coordinates" : [ [ 
        #    [0.0001383168733184448, 5.836986395024724e-05], 
        #    [6.036547989651808e-05, 6.102209576397399e-05], 
        #    [5.951977148299648e-05, 0.0001788888702378699], 
         #   [-7.723460844033525e-05, 0.0001792680245529255], 
         #   [-7.442872255580824e-05, 0.0002988450114997931], 
          #  [0.0001391722775849757, 0.0003004005321542991], 
           # [0.0001383168733184448, 5.836986395024724e-05]
        #] ] }}}}'''


        q = query_trajectories(query)
        trajectory_poses = q.trajs

        if len(trajectory_poses)==0:
            print "No Trajectories in this Region"            
            continue
        else:
            print "number of unique traj returned = " + repr(len(trajectory_poses))
        raw_input("Press enter to continue")

        """Create Landmark pins at randomly selected poses from all the trajectory data"""      
        all_poses = list(itertools.chain.from_iterable(trajectory_poses.values()))
        print "number of poses in total = " +repr(len(all_poses))


        ##To Dump trajectories for testing
        #data_dir='/home/strands/STRANDS'
        #obj_file  = os.path.join(data_dir, 'obj_dump.p')
        #traj_file = os.path.join(data_dir, 'traj_dump.p')
        #pickle.dump(objects_in_roi, open(obj_file, 'w'))
        #pickle.dump(trajectory_poses, open(traj_file, 'w'))

        #pins = lu.Landmarks(select_landmark_poses(all_poses))
        #static_things = pins.poses_landmarks
        #static_things = objects_in_roi
        #objects_per_trajectory = trajectory_object_dist(static_things, trajectory_poses)

    print "running rospy.spin()"    
    rospy.spin()  
