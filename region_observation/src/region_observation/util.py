#!/usr/bin/env python

import math
import itertools
import numpy as np
from shapely.geometry import Polygon
from tf.transformations import euler_from_quaternion


# def time_difference(self, tested_time):
#     in_datetime = datetime.datetime.fromtimestamp(tested_time.secs)
#     in_utcdatetime = datetime.datetime.utcfromtimestamp(tested_time.secs)
#     in_rostime = rospy.Time(time.mktime(in_datetime.timetuple()))
#     in_utcrostime = rospy.Time(time.mktime(in_utcdatetime.timetuple()))
#     delta = abs(tested_time.secs - in_rostime.secs)
#     return delta / 3600

# calculate polygon area using Shoelace Formula
# http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
def poly_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def is_intersected(poly1, poly2):
    return poly1.intersects(poly2)


def create_polygon(xs, ys):
    # if poly_area(np.array(xs), np.array(ys)) == 0.0:
    if Polygon(np.array(zip(xs, ys))).area == 0.0:
        xs = [
            [xs[0]] + list(i) for i in itertools.permutations(xs[1:])
        ]
        ys = [
            [ys[0]] + list(i) for i in itertools.permutations(ys[1:])
        ]
        areas = list()
        for ind in range(len(xs)):
            # areas.append(poly_area(np.array(xs[ind]), np.array(ys[ind])))
            areas.append(Polygon(np.array(zip(xs[ind], ys[ind]))))
        return Polygon(
            np.array(zip(xs[areas.index(max(areas))], ys[areas.index(max(areas))]))
        )
    else:
        return Polygon(np.array(zip(xs, ys)))


def robot_view_cone(pose):
    """
        let's call the triangle PLR, where P is the robot pose,
        L the left vertex, R the right vertex
    """
    d = 4       # max monitored distance: reasonably not more than 3.5-4m
    alpha = 1   # field of view: 57 deg kinect, 58 xtion, we can use exactly 1 rad (=57.3 deg)
    _, _, yaw = euler_from_quaternion(
        [0, 0, pose.orientation.z, pose.orientation.w]
    )
    # yaw = (yaw - (0.5 * math.pi)) % (2 * math.pi)
    lyaw = (yaw - (0.5 * alpha)) % (2 * math.pi)
    ryaw = (yaw + (0.5 * alpha)) % (2 * math.pi)

    Px = pose.position.x
    Py = pose.position.y
    Lx = Px + d * math.cos(lyaw)
    Ly = Py + d * math.sin(lyaw)
    Rx = Px + d * math.cos(ryaw)
    Ry = Py + d * math.sin(ryaw)

    return Polygon([[Lx, Ly], [Rx, Ry], [Px, Py]]), [[Lx, Ly], [Rx, Ry], [Px, Py]]


def robot_view_area(Px, Py, yaw):
    d = 3
    poses = list()
    degree = 45 / float(180) * math.pi

    for i in range(8):
        x = Px + d * (math.cos((yaw+(i * degree) % (2 * math.pi))))
        y = Py + d * (math.sin((yaw+(i * degree) % (2 * math.pi))))
        poses.append([x, y])
    # return [[Px + 4, Py], [Px, Py - 4], [Px - 4, Py], [Px, Py + 4]]
    return poses
