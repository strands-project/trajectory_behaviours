#!/usr/bin/env python

"""Utilities for landmarks"""

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
from scipy import spatial
from interactive_markers.interactive_marker_server import *
from visualization_msgs.msg import *
from geometry_msgs.msg import Point


def processFeedback(feedback):
    p = feedback.pose.position
    print feedback.marker_name + " is now at " + str(p.x) + ", " + str(p.y) + ", " + str(p.z)
   
    p = feedback.pose.orientation
    print p.x, p.y,p.z,p.w


def select_landmark_poses(input_data):
    """Input a list of all poses. Output a random selection of landmarks"""

    landmark_poses = []
    for i in xrange(10):
        pose= random.choice(input_data)
        landmark_poses.append(pose)

    return landmark_poses


class Landmarks():

    def __init__(self, poses):

        # create an interactive marker server on the topic namespace landmark_markers
        self._server = InteractiveMarkerServer("landmark_markers")
        self.poses = poses
        self.poses_landmarks = {}
        self.visualize_landmarks()

    def visualize_landmarks(self):
        """Create an interactive marker per landmark"""

        for i, pose in enumerate(self.poses):
            name = "landmark" + repr(i)
            self.poses_landmarks[name] = pose
            int_marker = self.create_marker(pose, name)

        
    def create_marker(self, pose, name, interactive=False):
        #print name, pose
        (x,y,z) = pose

        # create an interactive marker for our server
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "/map"
        int_marker.name = "%s" %name
        int_marker.description = "Simple 1-DOF Control"

        # create a grey box marker
        box_marker = Marker()
        box_marker.type = Marker.ARROW

        box_marker.scale.x = 1
        box_marker.scale.y = 0.1
        box_marker.scale.z = 0.1
    
        box_marker.pose.position = Point(x, y, 1)
        box_marker.pose.orientation.x = -8.02139854539e-10
        box_marker.pose.orientation.y = 0.695570290089
        box_marker.pose.orientation.z = 1.66903157961e-10
        box_marker.pose.orientation.w = 0.695570290089
        
        box_marker.color.r = 8.0
        box_marker.color.g = 0.1
        box_marker.color.b = 0.1
        box_marker.color.a = 0.7

        # create a non-interactive control which contains the box
        box_control = InteractiveMarkerControl()
        box_control.always_visible = True
        box_control.markers.append( box_marker )
        int_marker.controls.append( box_control )

        if interactive:
            rotate_control = InteractiveMarkerControl()
            rotate_control.name = "move_x"
            rotate_control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            int_marker.controls.append(rotate_control);


            #To create a control that rotates the arrows:
            control = InteractiveMarkerControl()
            control.orientation.x = 0
            control.orientation.y = 0
            control.orientation.z = 1
            control.orientation.w = 1
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            int_marker.controls.append(control)

        self._server.insert(int_marker, processFeedback)

        # 'commit' changes and send to all clients
        self._server.applyChanges()
        return int_marker


