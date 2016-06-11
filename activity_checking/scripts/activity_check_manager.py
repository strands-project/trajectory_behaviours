#! /usr/bin/env python

import sys
import yaml
import time
import rospy
import roslib
import datetime
import threading
from dateutil import parser
from activity_checking.people_counter import PeopleCounter


class ActivityCheck(object):

    def __init__(self):
        region_name_path = ""
        config_path = rospy.get_param("~config_path", "")
        self.soma_config = rospy.get_param("~soma_config", "activity_exploration")
        self.collection_name = "tsc_blog"
        if config_path == "":
            config_path = roslib.packages.get_pkg_dir('activity_checking') + '/config/default.yaml'
            # region name path must be the same as config path
            region_name_path = roslib.packages.get_pkg_dir('activity_checking') + '/config/region_names.yaml'
        try:
            weekly_shift = yaml.load(open(config_path, 'r'))
            self.region_names = yaml.load(open(region_name_path, 'r'))
        except:
            rospy.logerr("default.yaml and region_names.yaml can not be found!")
            rospy.loginfo("Terminating...")
            sys.exit(2)
        self.weekly_shift = self._convert_weekly_shift(
            weekly_shift
        )

    def _convert_weekly_shift(self, weekly_shift):
        rospy.loginfo("Creating shifting times to check human activity...")
        result = list()
        for daily_check in weekly_shift:
            temp = list()
            for shift_check in daily_check:
                st, et = shift_check.split("-")
                st = parser.parse(st)
                et = parser.parse(et)
                temp.append((st, et))
            result.append(temp)
        return result

    def continuous_check(self):
        end_time = None
        thread = None
        while not rospy.is_shutdown():
            curr = rospy.Time.now()
            if end_time is None:
                curr = datetime.datetime.fromtimestamp(curr.secs)
                for st, et in self.weekly_shift[curr.weekday()]:
                    st = datetime.datetime(
                        curr.year, curr.month, curr.day,
                        st.hour, st.minute
                    )
                    et = datetime.datetime(
                        curr.year, curr.month, curr.day,
                        et.hour, et.minute
                    )
                    if curr >= st and curr < et:
                        end_time = rospy.Time(
                            time.mktime(et.timetuple())
                        )
                        thread = self._check(et, curr)
                        rospy.sleep(0.1)
            elif curr >= end_time:
                thread.join()
                end_time = None
                thread = None
            rospy.sleep(0.1)
        # if thread is not None and ac is not None:
        #     ac.stop_check()
        #     thread.join()
        #     rospy.sleep(0.1)

    def _check(self, et, curr):
        dur = rospy.Duration((et - curr).total_seconds())
        ac = PeopleCounter(
            self.soma_config, region_names=self.region_names, coll=self.collection_name
        )
        thread = threading.Thread(
            target=ac.continuous_check,
            args=(dur,)
        )
        thread.start()
        return thread


if __name__ == '__main__':
    rospy.init_node("activity_checking")
    ac = ActivityCheck()
    ac.continuous_check()
    rospy.spin()
