#!/usr/bin/env python

import sys
import rospy
from std_msgs.msg import String
from relational_learner.srv import *
from relational_learner.msg import *



class NoveltyClient(object):

    def __init__(self):
        self.ret = None
        self.uuid = ''
        
        self.pub = rospy.Publisher("/trajectory_behaviours/novel_trajectory", String, queue_size=10)
        rospy.Subscriber("/trajectory_behaviours/episodes", episodesMsg, self.callback)

    def novelty_client(self, msg):
        rospy.wait_for_service('/novelty_detection')
        proxy = rospy.ServiceProxy('/novelty_detection', NoveltyDetection)  
        req = NoveltyDetectionRequest(msg)
        ret = proxy(req)
        return ret

    def callback(self, msg):
        if len(msg.uuid) > 0:
            self.uuid = msg.uuid
            self.roi = msg.roi
            self.episodes = msg.episodes
            self.ret = self.novelty_client(msg)


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
    print "novelty client running..."

    novlogic = NoveltyScoreLogic()
    nc = NoveltyClient()
    cnt=0
    while not rospy.is_shutdown():
        if nc.ret !=None:
            print "\n", cnt, nc.uuid
            cnt+=1
            print nc.ret
  
            if novlogic.test(nc.uuid, nc.ret): 
                nc.pub.publish(nc.uuid)
            print novlogic.msg
            rospy.sleep(1)

    rospy.spin()
























