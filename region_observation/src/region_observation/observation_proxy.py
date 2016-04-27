#!/usr/bin/env python

import rospy
import datetime
from region_observation.msg import RegionObservationTime
from mongodb_store.message_store import MessageStoreProxy


class RegionObservationProxy(object):

    def __init__(
        self, soma_map, soma_config, coll="region_observation"
    ):
        rospy.loginfo("Initializing region observation proxy...")
        self.soma_map = soma_map
        self.soma_config = soma_config
        rospy.loginfo(
            "Soma map is %s with the configuration %s" %
            (soma_map, soma_config)
        )
        self._db = MessageStoreProxy(collection=coll)

    def load(self, start_time, end_time, roi="", minute_increment=1):
        # [roi[month[day[hour[minute:duration]]]]]
        rospy.loginfo(
            "Querying region observation from %s to %s" %
            (start_time.secs, end_time.secs)
        )
        end_time = end_time - rospy.Duration(minute_increment * 60, 0)
        query = {
            "soma": self.soma_map, "soma_config": self.soma_config,
            "start_from.secs": {"$gte": start_time.secs, "$lt": end_time.secs}
        }
        print query
        if roi != "":
            query.update({"region_id": roi})
        logs = self._db.query(RegionObservationTime._type, query)
        rospy.loginfo("Got %d entries..." % len(logs))
        roi_observation = dict()
        total_observation = rospy.Duration(0, 0)
        for log in logs:
            start = datetime.datetime.fromtimestamp(log[0].start_from.secs)
            end = log[0].until + rospy.Duration(0, 1)
            end = datetime.datetime.fromtimestamp(end.secs)
            if end.minute - start.minute == minute_increment:
                if log[0].region_id not in roi_observation:
                    roi_observation[log[0].region_id] = dict()
                if start.month not in roi_observation[log[0].region_id]:
                    roi_observation[log[0].region_id][start.month] = dict()
                if start.day not in roi_observation[log[0].region_id][start.month]:
                    roi_observation[log[0].region_id][start.month][start.day] = dict()
                if start.hour not in roi_observation[log[0].region_id][start.month][start.day]:
                    roi_observation[log[0].region_id][start.month][start.day][start.hour] = dict()
                key = "%s-%s" % (start.minute, end.minute)
                roi_observation[log[0].region_id][start.month][start.day][start.hour][key] = log[0].duration
                total_observation += log[0].duration

        return roi_observation, total_observation
