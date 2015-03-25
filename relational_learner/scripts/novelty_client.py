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
    
    def add(self, uuid, spatial_dist):
        self.spatial_scores[uuid] = spatial_dist


    def mean(self):
        """Return the sample arithmetic mean of data."""
        n = len(self.spatial_scores.values())
        if n < 1:
            raise ValueError('mean requires at least one data point')
        return sum(self.spatial_scores.values())/float(n)

    def _ss(self):
        """Return sum of square deviations of sequence data."""
        c = self.mean()
        ss = sum((x-c)**2 for x in self.spatial_scores.values())
        return ss

    def pstdev(self):
        """Calculates the population standard deviation."""
        n = len(self.spatial_scores.values())
        if n < 2:
            raise ValueError('variance requires at least two data points')
        ss = self._ss()
        pvar = ss/n # the population variance
        return pvar**0.5
    
if __name__ == "__main__":
    rospy.init_node('novelty_client')
    
    nsl = NoveltyScoreLogic()
    nc = NoveltyClient()
    
    while not rospy.is_shutdown():
        if nc.ret !=None:
            print "\nRESULTS = ", nc.ret
            nsl.add(i.uuid, ret.spatial_dist)

            values = nsl.spatial_scores.values()
            nc.pub.publish(i.uuid)

            print "mean of collection: ", nsl.mean()
            print "sum of square deviations: ", nsl._ss()
            if len(values)>1: print "population std dev: ",  nsl.pstdev()
            #print nsl.spatial_scores
            rospy.sleep(1)
    
    


    rospy.spin()
























