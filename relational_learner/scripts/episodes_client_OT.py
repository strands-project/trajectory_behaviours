#!/usr/bin/env python

#__author__      = "Paul Duckworth"
#__copyright__   = "Copyright 2015, University of Leeds"

import sys
import rospy
from std_msgs.msg import String
from relational_learner.srv import *
from relational_learner.msg import *
from human_trajectory.msg import Trajectories
import relational_learner.obtain_trajectories as ot

class EpisodeClient(object):

    def __init__(self):
        self.ret = None
        self.uuid = ''
        #self.pose = None

        self.pub = rospy.Publisher("/trajectory_behaviours/episodes", episodesMsg, queue_size=10)
        rospy.Subscriber("/human_trajectories/trajectories/batch", Trajectories, self.callback)

    def episode_client(self, Trajectory):
        rospy.wait_for_service('/episode_service')
        proxy = rospy.ServiceProxy('/episode_service', EpisodeService)  
        req = EpisodeServiceRequest(Trajectory)
        ret = proxy(req)
        return ret

    def callback(self, msg):
        if len(msg.trajectories) > 0:
            self.uuid = msg.trajectories[0].uuid
            #self.pose = msg.trajectories[0].trajectory[-1].pose
            self.ret = self.novelty_client(msg.trajectories[0])


if __name__ == "__main__":
    rospy.init_node('episodes_client')

    ec = EpisodeClient()

    ### Query all ROI 12 to test (or just one dude) ###
    #query = '''{"uuid": "7d638405-b2f8-55ce-b593-efa8e3f2ff2e"}''' 
    ###geoIntersects? - Fix

    
    query ='''{"loc": { "$geoWithin": { "$geometry":
        { "type" : "Polygon", "coordinates" : [ [ 
                    [ -0.0002246355582968818, 
                      -2.519034444503632e-05],
                    [ -0.0002241486476179944, 
                     -7.42736662147081e-05], 
                    [ -0.000258645873657315, 
                      -7.284014769481928e-05],
                    [ -0.0002555339747090102, 
                      -2.521782172948406e-05],
                    [ -0.0002246355582968818, 
                      -2.519034444503632e-05]
                    ] ] }}}}'''
    
    q = ot.query_trajectories(query)

    test_list= [q.res.trajectories.trajectories[0],q.res.trajectories.trajectories[1],\
       q.res.trajectories.trajectories[1], q.res.trajectories.trajectories[2], \
       q.res.trajectories.trajectories[2], q.res.trajectories.trajectories[2],\
       q.res.trajectories.trajectories[0], q.res.trajectories.trajectories[0]]

    for cnt, i in enumerate(q.res.trajectories.trajectories):
    #for cnt, i in enumerate(test_list):
        print "\n",cnt, i.uuid

        ret = ec.episode_client(i)
        #print ret.header
        print ret.uuid, ret.soma_roi_id
        print "len = ", len(ret.episodes)
        ec.pub.publish(ret)
        rospy.sleep(1) 

    #rospy.spin()
























