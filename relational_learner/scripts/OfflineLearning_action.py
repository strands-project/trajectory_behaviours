#! /usr/bin/env python

import sys
import rospy
import actionlib
from learn_traj_behaviours import Offline_Learning
from relational_learner.msg import OfflineLearningAction, OfflineLearningResult


class OfflineLearning_server(object):
    def __init__(self, turn_on_plotting=True):
        self.turn_on_plotting = bool(turn_on_plotting)

        # Start server
        rospy.loginfo("OfflineLearning starting an action server")
        self._as = actionlib.SimpleActionServer(
            "OfflineLearning",
            OfflineLearningAction,
            execute_cb=self.execute,
            auto_start=False
        )
        self._as.start()


    def cond(self):
        return self._as.is_preempt_requested()

    def execute(self, goal):
        ol = Offline_Learning()

	if not self.cond(): ol.learn(self.turn_on_plotting)
	#Split on multiple methods to allow system to stop the action server
	self._as.set_succeeded(OfflineLearningResult())

if __name__ == "__main__":
    rospy.init_node('OfflineLearning_server')

    turn_plotting = True
    if len(sys.argv) < 2:
        rospy.logerr(
            "usage: offlinelearning turn_on_plotting[0/1]"
        )
        sys.exit(1)

    OfflineLearning_server(int(sys.argv[1]))
    rospy.spin()
