#! /usr/bin/env python

import tf
import rospy
import datetime
import threading
import message_filters
from std_msgs.msg import Header
from shapely.geometry import Point
from robblog.msg import RobblogEntry
from scipy.spatial.distance import euclidean
from activity_checking.util import get_soma_info
from bayes_people_tracker.msg import PeopleTracker
from geometry_msgs.msg import PoseStamped, PoseArray
from mongodb_store.message_store import MessageStoreProxy


class PeopleCounter(object):

    def __init__(self, config, coll='activity_robblog'):
        rospy.loginfo("Starting activity checking...")
        self._lock = False
        # regions = {roi:polygon} and soma map info
        self.regions, self.soma_map = get_soma_info(config)
        self.reset()
        # tf stuff
        self._tfl = tf.TransformListener()
        # create db
        rospy.loginfo("Create database collection %s..." % coll)
        self._db = MessageStoreProxy(collection=coll)
        # subscribing to ubd topic
        subs = [
            message_filters.Subscriber(
                rospy.get_param(
                    "~ubd_topic", "/upper_body_detector/bounding_box_centres"
                ),
                PoseArray
            ),
            message_filters.Subscriber(
                rospy.get_param("~tracker_topic", "/people_tracker/positions"),
                PeopleTracker
            )
        ]
        ts = message_filters.ApproximateTimeSynchronizer(
            subs, queue_size=5, slop=0.15
        )
        ts.registerCallback(self.cb)

    def reset(self):
        self.uuids = {roi: list() for roi, _ in self.regions.iteritems()}
        self.people_poses = list()
        self._stop = False
        self._ubd_pos = list()
        self._tracker_pos = list()
        self._tracker_uuids = list()

    def cb(self, ubd_cent, pt):
        if not self._lock:
            self._lock = True
            self._tracker_uuids = pt.uuids
            self._ubd_pos = self.to_world_all(ubd_cent)
            self._tracker_pos = [i for i in pt.poses]
            self._lock = False

    def to_world_all(self, pose_arr):
        transformed_pose_arr = list()
        try:
            fid = pose_arr.header.frame_id
            for cpose in pose_arr.poses:
                ctime = self._tfl.getLatestCommonTime(fid, "/map")
                pose_stamped = PoseStamped(Header(1, ctime, fid), cpose)
                # Get the translation for this camera's frame to the world.
                # And apply it to all current detections.
                tpose = self._tfl.transformPose("/map", pose_stamped)
                transformed_pose_arr.append(tpose.pose)
        except tf.Exception as e:
            rospy.logwarn(e)
            # In case of a problem, just give empty world coordinates.
            return []
        return transformed_pose_arr

    def stop_check(self):
        self._stop = True

    def continuous_check(self, duration):
        rospy.loginfo("Start looking for people...")
        start_time = rospy.Time.now()
        end_time = rospy.Time.now()
        while (end_time - start_time) < duration and not self._stop:
            if not self._lock:
                self._lock = True
                for i in self._ubd_pos:
                    for ind, j in enumerate(self._tracker_pos):
                        # merge ubd with tracker pose
                        cond = euclidean(
                            [i.position.x, i.position.y],
                            [j.position.x, j.position.y]
                        ) < 0.2
                        # uuid must be new
                        is_new = True
                        for roi, uuids in self.uuids.iteritems():
                            if self._tracker_uuids[ind] in uuids:
                                is_new = False
                                break
                        cond = cond and is_new
                        # this pose must be inside a region
                        pose_inside_roi = ''
                        for roi, region in self.regions.iteritems():
                            if region.contains(
                                Point(i.position.x, i.position.y)
                            ):
                                pose_inside_roi = roi
                                break
                        cond = cond and (pose_inside_roi != '')
                        is_near = False
                        for pose in self.people_poses:
                            if euclidean(
                                pose, [i.position.x, i.position.y]
                            ) < 0.3:
                                is_near = True
                                break
                        cond = cond and (not is_near)
                        if cond:
                            # self.uuids.append(self._tracker_uuids[ind])
                            self.uuids[pose_inside_roi].append(
                                self._tracker_uuids[ind]
                            )
                            self.people_poses.append(
                                [i.position.x, i.position.y]
                            )
                            rospy.loginfo(
                                "%s is detected in region %s - (%.2f, %.2f)" % (
                                    self._tracker_uuids[ind], roi,
                                    i.position.x, i.position.y
                                )
                            )
                self._lock = False
            end_time = rospy.Time.now()
            rospy.sleep(0.1)
        rospy.loginfo("Storing location and the number of detected persons...")
        start_time = datetime.datetime.fromtimestamp(start_time.secs)
        end_time = datetime.datetime.fromtimestamp(end_time.secs)
        string_body = ''
        for roi, uuids in self.uuids.iteritems():
            string_body += "**Region %s**: %d person(s) were detected\n" % (
                roi, len(uuids)
            )
        entry = RobblogEntry(
            title="Activity check from %s to %s" % (start_time, end_time),
            body=string_body
        )
        self._db.insert(entry)
        self._stop = False


if __name__ == '__main__':
    rospy.init_node("activity_checker")
    soma_config = rospy.get_param("~soma_config", "activity_exploration")
    ac = PeopleCounter(soma_config)
    thread = threading.Thread(
        target=ac.continuous_check, args=(rospy.Duration(60),)
    )
    thread.start()
    rospy.sleep(10)
    ac.stop_check()
    thread.join()
