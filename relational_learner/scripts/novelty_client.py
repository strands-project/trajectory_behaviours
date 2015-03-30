#!/usr/bin/env python

import sys
import rospy
from relational_learner.srv import *
#from relational_learner.msg import *
import relational_learner.obtain_trajectories as ot
from human_trajectory.msg import Trajectories
from std_msgs.msg import String


class NoveltyClient(object):

    def __init__(self):
        self.ret = None
        self.uuid = ''
        self.pose = None
        
        self.pub = rospy.Publisher("/trajectory_behaviours/novel_trajectory", String, queue_size=10)
        rospy.Subscriber("/human_trajectories/trajectories/batch", Trajectories, self.callback)

    def novelty_client(self, Trajectory):
        rospy.wait_for_service('/novelty_detection')
        proxy = rospy.ServiceProxy('/novelty_detection', NoveltyDetection)  
        req = NoveltyDetectionRequest(Trajectory)
        ret = proxy(req)
        return ret

    def callback(self, msg):
        if len(msg.trajectories) > 0:
            self.uuid = msg.trajectories[0].uuid
            self.pose = msg.trajectories[0].trajectory[-1].pose
            self.ret = self.novelty_client(msg.trajectories[0])


class NoveltyScoreLogic(object):
    def __init__(self):
        self.spatial_scores = {}
        self.temp_scores = {}
        self.published_uuids = []


    def test(self, uuid, ret):
        """Tests whether UUID is a novel trajectory"""
        self.uuid = uuid         
        spatial_novelty = ret.spatial_dist
        temp1 = ret.temporal_nov[0]
        temp2 = ret.temporal_nov[1]

        if spatial_novelty > 0: self.msg = "  >>> spatial novelty %s" % spatial_novelty
        elif ret.roi_knowledge > 0.5:
            if temp1 < 0.05: self.msg = "  >>> temporal novelty1"
            if temp2 < 0.05: self.msg = "  >>> temporal novelty2"
            self.msg=""
        else: self.msg = ""

        if self.msg!="":
            self.published_uuids.append(uuid)
            return True
        else: return False

    
if __name__ == "__main__":
    rospy.init_node('novelty_client')
    
    novlogic = NoveltyScoreLogic()
    nc = NoveltyClient()
    cnt=0
    while not rospy.is_shutdown():
        if nc.ret !=None:
            print "\n", cnt, nc.uuid
            cnt+=1
            print nc.ret
  
            if novlogic.test(nc.uuid, ret): 
                nc.pub.publish(nc.uuid)
            print novlogic.msg
            rospy.sleep(1)

    rospy.spin()
























