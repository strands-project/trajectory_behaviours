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

    ### Query all ROI 12 to test (or just one dude) ###
    #query = '''{"uuid": "328e2f8c-6147-5525-93c4-1b281887623b"}''' 
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

    #test_list= [q.res.trajectories.trajectories[0],q.res.trajectories.trajectories[1],\
    #    q.res.trajectories.trajectories[1], q.res.trajectories.trajectories[2], \
    #    q.res.trajectories.trajectories[2], q.res.trajectories.trajectories[2],\
    #    q.res.trajectories.trajectories[0], q.res.trajectories.trajectories[0]]

    for cnt, i in enumerate(q.res.trajectories.trajectories):
        if i.uuid in novlogic.published_uuids: continue
        print "\n",cnt, i.uuid

        ret = nc.novelty_client(i)
        print ret

        if novlogic.test(i.uuid, ret): 
            nc.pub.publish(i.uuid)
        print novlogic.msg

        rospy.sleep(1) 

    print novlogic.published_uuids



    #rospy.spin()
























