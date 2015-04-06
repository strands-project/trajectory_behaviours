#! /usr/bin/env python

import rospy
import actionlib
from human_trajectory_classifier.msg import HMCAction, HMCGoal


def test():
    client = actionlib.SimpleActionClient(
        "human_movement_detection_server", HMCAction
    )
    rospy.loginfo("waiting for server...")
    client.wait_for_server()

    while not rospy.is_shutdown():
        goal = HMCGoal()
        request = raw_input("[update | accuracy | online | preempt]\n")
        if request != 'preempt':
            goal.request = request
            client.send_goal(goal)
            if request == 'update':
                client.wait_for_result()
                result = client.get_result()
                rospy.loginfo(str(result.updated) + " " + str(result.accuracy))
        else:
            client.cancel_goal()


if __name__ == '__main__':
    rospy.init_node("human_movement_detection_tester")
    test()
    rospy.spin()
