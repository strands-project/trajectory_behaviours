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
        self.cnt = 0

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
            self.ret = self.episode_client(msg.trajectories[0])
            if self.ret.soma_roi_id != "":
                self.pub.publish(self.ret)
                print "\nuuid = %s Episodes uploaded to database" % self.uuid
                #print self.cnt
                self.cnt+=1
            else:
                print "\nuuid = %s is outside of all soma roi" % self.ret.uuid

if __name__ == "__main__":
    rospy.init_node('episodes_client')

    ec = EpisodeClient()

    rospy.spin()
























