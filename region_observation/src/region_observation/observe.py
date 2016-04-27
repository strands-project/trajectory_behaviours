#!/usr/bin/env python

import sys
import time
import rospy
import datetime
import argparse

from geometry_msgs.msg import Pose, Point
from soma_map_manager.srv import MapInfo
from visualization_msgs.msg import Marker
from soma_manager.srv import SOMA2QueryObjs
from region_observation.msg import RegionObservationTime
from mongodb_store.message_store import MessageStoreProxy
from region_observation.util import robot_view_cone, create_polygon, is_intersected


class OnlineRegionObservation(object):

    def __init__(
        self, name, soma_config, minute_increment=1, coll="region_observation"
    ):
        rospy.loginfo("Initializing region observation...")
        # get map info
        self.minute_increment = minute_increment
        soma_service = rospy.ServiceProxy("/soma2/map_info", MapInfo)
        soma_service.wait_for_service()
        self.soma_map = soma_service(1).map_name
        rospy.loginfo("Got soma map name %s..." % self.soma_map)
        # get region information from soma2
        soma_service = rospy.ServiceProxy("/soma2/query_db", SOMA2QueryObjs)
        soma_service.wait_for_service()
        result = soma_service(
            2, False, False, False, False, False, False,
            0, 0, 0, 0, 0, 0, 0, 0, [], [], ""
        )
        # create polygon for each regions
        self.soma_config = soma_config
        self.regions = dict()
        self.intersected_regions = list()
        rospy.loginfo("Total regions for this soma map are %d" % len(result.rois))
        for region in result.rois:
            if region.config == self.soma_config and region.map_name == self.soma_map:
                xs = [pose.position.x for pose in region.posearray.poses]
                ys = [pose.position.y for pose in region.posearray.poses]
                self.regions[region.roi_id] = create_polygon(xs, ys)
        # get robot sight
        rospy.loginfo("Subcribe to /robot_pose...")
        rospy.Subscriber("/robot_pose", Pose, self._robot_cb, None, 10)
        self.region_observation_duration = dict()
        # db for RegionObservation
        rospy.loginfo("Create collection db as %s..." % coll)
        self._db = MessageStoreProxy(collection=coll)
        # draw robot view cone
        self._pub = rospy.Publisher("%s/view_cone" % name, Marker, queue_size=10)

    def _robot_cb(self, pose):
        robot_sight, arr_robot_sight = robot_view_cone(pose)
        self.draw_view_cone(arr_robot_sight)
        intersected_regions = list()
        for roi, region in self.regions.iteritems():
            if is_intersected(robot_sight, region):
                intersected_regions.append(roi)
        self.intersected_regions = intersected_regions

    def observe(self):
        rospy.loginfo("Starting robot observation...")
        while not rospy.is_shutdown():
            # get the right starting time
            start_time = rospy.Time.now()
            st = datetime.datetime.fromtimestamp(start_time.secs)
            start_time = rospy.Time(time.mktime(st.timetuple()))
            if st.second != 0:
                rospy.sleep(0.1)
                continue
            if st.minute % self.minute_increment != 0:
                st = st - datetime.timedelta(
                    minutes=st.minute % self.minute_increment, seconds=st.second
                )
                start_time = rospy.Time(time.mktime(st.timetuple()))
            et = st + datetime.timedelta(minutes=self.minute_increment)
            end_time = rospy.Time(time.mktime(et.timetuple()))
            self._observe(start_time, end_time)

    def _observe(self, start_time, end_time):
        # get where the robot is initially
        prev_roi = self.intersected_regions
        rospy.loginfo("Robot sees regions %s" % (prev_roi))
        roi_start_time = {roi: start_time for roi in prev_roi}
        duration = dict()
        current_time = rospy.Time.now()
        while rospy.Time.now() < end_time:
            if rospy.is_shutdown():
                return
            current_roi = self.intersected_regions
            # for a new registered roi, store current time
            for roi in [i for i in current_roi if i not in prev_roi]:
                rospy.loginfo("Robot sees a new region %s" % roi)
                roi_start_time[roi] = current_time
            # for a registered roi that was just passed, calculate duration
            for roi in [i for i in prev_roi if i not in current_roi]:
                rospy.loginfo("Robot leaves region %s" % roi)
                if roi in duration:
                    duration[roi] += (current_time - roi_start_time[roi])
                else:
                    duration[roi] = current_time - roi_start_time[roi]
                del roi_start_time[roi]
            prev_roi = current_roi
            current_time = rospy.Time.now()
            rospy.sleep(0.05)

        for roi, st in roi_start_time.iteritems():
            duration[roi] = current_time - st
        self.save_observation(duration, start_time, end_time)

    def save_observation(self, duration, start_time, end_time):
        end_time = end_time - rospy.Duration(0, 1)
        rospy.loginfo(
            "Save observation within %s and %s..." % (str(start_time), str(end_time))
        )
        for roi, duration in duration.iteritems():
            self._db.insert(
                RegionObservationTime(
                    self.soma_map, self.soma_config, roi,
                    start_time, end_time, duration
                )
            )

    def draw_view_cone(self, view_cone):
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "view_cone"
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.scale.x = 0.05
        marker.color.a = 1.0
        marker.color.g = 1.0
        for ind, point in enumerate(view_cone+[view_cone[0]]):
            if ind == 2:
                marker.points.append(Point(point[0], point[1], 1.65))
            else:
                marker.points.append(Point(point[0], point[1], 0.1))
        self._pub.publish(marker)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="region_observation")
    parser.add_argument('soma_config', help="Soma Config")
    parser.add_argument('minute_increment', help="The Increment Minute Interval")
    args = parser.parse_args()

    rospy.init_node("region_observation")
    if 60 % int(args.minute_increment) != 0:
        rospy.loginfo("Please set minute_increment to a factor of 60")
        sys.exit(2)
    ro = OnlineRegionObservation(rospy.get_name(), args.soma_config, int(args.minute_increment))
    ro.observe()
    rospy.spin()
