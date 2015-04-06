#!/usr/bin/env python

import sys
import random
import rospy
import math
import pymongo
import pylab
import matplotlib.pyplot as plt
from collections import namedtuple
from mpl_toolkits.axes_grid.axislines import SubplotZero
from human_trajectory.trajectory import Trajectory
from geometry_msgs.msg import Point, Quaternion, Pose, PoseStamped
from std_msgs.msg import Header


class KNNClassifier(object):

    def __init__(self):
        self.alpha = 0.6
        self.beta = 0.4
        self.k = 11
        self.accuracy = 0
        self.training_data = []
        self.test_data = []
        self.LabeledNormalizedPoses = namedtuple(
            "NormalizePoses", "uuid real normal"
        )

    # update training and test data from database
    def update_database(self):
        self.training_data = []
        self.test_data = []
        trajs = self._retrieve_logs()
        self._label_data(trajs)

    # splitting training data into training and test data
    def split_training_data(self, training_ratio):
        rospy.loginfo("Splitting data into test and training...")
        temp = []
        self.test_data = []
        for i in self.training_data:
            if random.random() < training_ratio:
                temp.append(i)
            else:
                self.test_data.append(i)
        self.training_data = temp

    # get k nearest values to a test data based on positions and velocities
    def _nearest_values_to(self, test):
        index = []
        nearest = []
        test_poses = test.normal
        for i, j in enumerate(self.training_data):
            dist = 0
            vel = 0
            for k, l in enumerate(j[0].normal):
                # delta distance calculation
                dx = test_poses[k].pose.position.x - l.pose.position.x
                dy = test_poses[k].pose.position.y - l.pose.position.y
                dist += math.hypot(dx, dy)
                # delta velocity calculation
                if k >= 1:
                    dx = l.pose.position.x - j[0].normal[k-1].pose.position.x
                    dy = l.pose.position.y - j[0].normal[k-1].pose.position.y
                    velo_l = math.hypot(dx, dy) / (
                        (l.header.stamp.secs -
                         j[0].normal[k-1].header.stamp.secs) +
                        (l.header.stamp.nsecs -
                         j[0].normal[k-1].header.stamp.nsecs) /
                        math.pow(10, 9)
                    )
                    dx = test_poses[k].pose.position.x - \
                        test_poses[k-1].pose.position.x
                    dy = test_poses[k].pose.position.y - \
                        test_poses[k-1].pose.position.y
                    velo_test = math.hypot(dx, dy) / (
                        (test_poses[k].header.stamp.secs -
                         test_poses[k-1].header.stamp.secs) +
                        (test_poses[k].header.stamp.nsecs -
                         test_poses[k-1].header.stamp.nsecs) /
                        math.pow(10, 9)
                    )
                    vel += abs(velo_l - velo_test)
            if nearest != []:
                dist = (self.alpha * dist) + (self.beta * vel)
                max_val = max(nearest)
                if max_val > dist and len(nearest) >= self.k:
                    temp = nearest.index(max_val)
                    nearest[temp] = dist
                    index[temp] = i
                elif max_val > dist and len(nearest) < self.k:
                    nearest.append(dist)
                    index.append(i)
            else:
                nearest.append(dist)
                index.append(i)

        sort_data = sorted(zip(nearest, index), key=lambda i: i[0])
        return [self.training_data[i[1]] for i in sort_data]

    # predict the class of the test data
    def predict_class_data(self, test_data):
        rospy.loginfo("Predicting class for %s", test_data.uuid)
        result = None
        nn = self._nearest_values_to(test_data)
        human = [i for i in nn if i[1] == 'human']
        nonhuman = [i for i in nn if i[1] == 'non-human']
        rospy.loginfo("Vote: %d, %d", len(human), len(nonhuman))
        if len(human) > len(nonhuman):
            result = 'human'
        else:
            result = 'non-human'

        rospy.loginfo("%s belongs to %s", test_data.uuid, result)
        return (result, human[:1], nonhuman[:1])

    # get accuracy of the overall prediction with 5-fold-cross validation
    def get_accuracy(self, queue=None):
        rospy.loginfo("Getting the overall accuracy...")
        # dividing training data into k
        k_fold = 5
        length = len(self.training_data) / k_fold
        k_fold_list = []
        preempt = False
        for i in range(k_fold):
            ind = i * length
            k_fold_list.append(self.training_data[ind:ind+length])

        # measure the accuracy
        accuracy = 0
        for j in k_fold_list:
            rospy.loginfo("Total testing data is %d", len(j))
            self.training_data = []
            for i in k_fold_list:
                if i != j:
                    self.training_data.extend(i)

            counter = 0
            for i in j:
                if queue is not None and not queue.empty():
                    preempt = queue.get()['preempt']
                    break
                result = self.predict_class_data(i[0])
                rospy.loginfo("The actual class is %s", i[1])
                if result[0] == i[1]:
                    counter += 1
            accuracy += float(counter) / float(len(j))
            rospy.loginfo("Accuracy for this data is %d", accuracy)
            if preempt:
                break

        if not preempt:
            self.accuracy = accuracy/float(k_fold)
        return self.accuracy

    # label data and put them into training set
    def _label_data(self, trajs):
        rospy.loginfo("Splitting data into chunk...")
        for uuid, traj in trajs.iteritems():
            traj.validate_all_poses()
            chunked_traj = self.create_chunk(
                uuid, list(zip(*traj.humrobpose)[0])
            )
            label = 'human'
            start = traj.humrobpose[0][0].header.stamp
            end = traj.humrobpose[-1][0].header.stamp
            delta = float((end-start).secs + 0.000000001 * (end-start).nsecs)
            if delta != 0.0:
                avg_vel = traj.length[-1] / delta
            else:
                avg_vel = 0.0
            guard = traj.length[-1] < 0.1 or avg_vel < 0.5 or avg_vel > 1.5
            if guard:
                label = 'non-human'
            for i in chunked_traj:
                self.training_data.append((i, label))

    # normalize poses so that the first pose becomes (0,0)
    # and the second pose becomes the base for the axis
    # with tangen, cos and sin
    def get_normalized_poses(self, poses):
        dx = poses[1].pose.position.x - poses[0].pose.position.x
        dy = poses[1].pose.position.y - poses[0].pose.position.y
        if dx < 0.00001:
            dx = 0.00000000000000000001
        rad = math.atan(dy / dx)
        for i, j in enumerate(poses):
            if i > 0:
                dx = j.pose.position.x - poses[0].pose.position.x
                dy = j.pose.position.y - poses[0].pose.position.y
                if dx < 0.00001:
                    dx = 0.00000000000000000001
                rad2 = math.atan(dy / dx)
                delta_rad = rad2 - rad
                if rad2 == 0:
                    r = dx / math.cos(rad2)
                else:
                    r = dy / math.sin(rad2)
                x = r * math.cos(delta_rad)
                y = r * math.sin(delta_rad)
                poses[i].pose.position.x = x
                poses[i].pose.position.y = y

        poses[0].pose.position.x = poses[0].pose.position.y = 0
        return poses

    # chunk data for each trajectory
    def create_chunk(self, uuid, poses, chunk=20):
        i = 0
        chunk_trajectory = []
        while i < len(poses) - (chunk - 1):
            normalized = list()
            # can not just do poses[i:i+chunk], need to rewrite
            for j in range(chunk):
                position = Point(
                    poses[i + j].pose.position.x,
                    poses[i + j].pose.position.y,
                    poses[i + j].pose.position.z
                )
                orientation = Quaternion(
                    poses[i + j].pose.orientation.x,
                    poses[i + j].pose.orientation.y,
                    poses[i + j].pose.orientation.z,
                    poses[i + j].pose.orientation.w
                )
                pose = Pose(position, orientation)
                header = Header(
                    poses[i + j].header.seq,
                    poses[i + j].header.stamp,
                    poses[i + j].header.frame_id
                )
                normalized.append(PoseStamped(header, pose))
            normalized = self.get_normalized_poses(normalized)
            chunk_trajectory.append(
                self.LabeledNormalizedPoses(uuid, poses[i:i+chunk], normalized)
            )
            i += chunk

        return chunk_trajectory

    # retrieve trajectory from mongodb
    def _retrieve_logs(self):
        client = pymongo.MongoClient(
            rospy.get_param("datacentre_host", "localhost"),
            rospy.get_param("datacentre_port", 62345)
        )
        rospy.loginfo("Retrieving data from mongodb...")
        trajs = dict()
        rospy.loginfo("Constructing data from people perception...")
        for log in client.message_store.people_perception.find():
            for i, uuid in enumerate(log['uuids']):
                if uuid not in trajs:
                    trajs[uuid] = Trajectory(uuid)
                header = Header(
                    log['header']['seq'],
                    rospy.Time(log['header']['stamp']['secs'],
                               log['header']['stamp']['nsecs']),
                    log['header']['frame_id']
                )
                human_pose = Pose(
                    Point(log['people'][i]['position']['x'],
                          log['people'][i]['position']['y'],
                          log['people'][i]['position']['z']),
                    Quaternion(log['people'][i]['orientation']['x'],
                               log['people'][i]['orientation']['y'],
                               log['people'][i]['orientation']['z'],
                               log['people'][i]['orientation']['w'])
                )
                robot_pose = Pose(
                    Point(log['robot']['position']['x'],
                          log['robot']['position']['y'],
                          log['robot']['position']['z']),
                    Quaternion(log['robot']['orientation']['x'],
                               log['robot']['orientation']['y'],
                               log['robot']['orientation']['z'],
                               log['robot']['orientation']['w']))
                trajs[uuid].append_pose(human_pose, header, robot_pose)
        return trajs


# create a visualisation graph in cartesian coordinate between test data,
# one of the nearest training data (human), and training data (non_human)
def visualize_test_between_class(test, human, non_human):
    fig = plt.figure("Trajectories for Test, Human, and Non-Human")
    ax = SubplotZero(fig, 111)
    fig.add_subplot(ax)
    line_style = ['r.-', 'gx-', 'bo-']

    # plotting test data
    x = [i.pose.position.x for i in test]
    y = [i.pose.position.y for i in test]
    ax.plot(x, y, line_style[0], label="Test")
    # plotting human data
    x = [i.pose.position.x for i in human]
    y = [i.pose.position.y for i in human]
    ax.plot(x, y, line_style[1], label="Human")
    # plotting non-human data
    x = [i.pose.position.x for i in non_human]
    y = [i.pose.position.y for i in non_human]
    ax.plot(x, y, line_style[2], label="Non-human")

    ax.margins(0.05)
    ax.legend(loc="lower right", fontsize=10)
    plt.title("Chunks of Trajectories")
    plt.xlabel("Axis")
    plt.ylabel("Ordinate")

    for direction in ["xzero", "yzero"]:
        ax.axis[direction].set_axisline_style("-|>")
        ax.axis[direction].set_visible(True)

    for direction in ["left", "right", "bottom", "top"]:
        ax.axis[direction].set_visible(False)

    pylab.grid()
    plt.show()


# getting True Positive Rate and True Negative Rate from different
# configurations (variation of k, alpha, and beta) from classifier
def get_tpr_tnr(classifier, k, alpha, beta):
    rospy.loginfo("Constructing tpr, tnr...")
    tp = fn = fp = tn = 0
    classifier.k = k
    classifier.alpha = alpha
    classifier.beta = beta

    for i, t in enumerate(classifier.test_data):
        prediction = classifier.predict_class_data(t[0])
        rospy.loginfo("The actual class is %s", t[1])
        if prediction[0] == 'human' and t[1] == 'human':
            tp += 1
        elif prediction[0] == 'human' and t[1] == 'non-human':
            fp += 1
        elif prediction[0] == 'non-human' and t[1] == 'non-human':
            tn += 1
        else:
            fn += 1
    tpr = tp / float(tp + fn)
    tnr = tn / float(fp + tn)
    print "TPR: %0.5f, TNR: %0.5f" % (tpr, tnr)
    return (tpr, tnr)


if __name__ == '__main__':
    rospy.init_node("knn_trajectory_classifier")

    if len(sys.argv) < 3:
        rospy.logerr(
            "usage: predictor train_ratio [roc|score|test_graph]"
        )
        sys.exit(2)

    lsp = KNNClassifier()
    lsp.update_database()

    if sys.argv[2] == 'score':
        rospy.loginfo("The overall accuracy is " + str(lsp.get_accuracy()))
    elif sys.argv[2] == 'roc':
        lsp.split_training_data(float(sys.argv[1]))
        alpha = [0.0, 0.5, 1.0]
        beta = [1.0, 0.5, 0.0]
        K = [5, 7, 9, 11]
        tpr = list()
        tnr = list()

        for i in range(len(alpha)):
            for k in K:
                print "K: %d, alpha: %0.2f, beta: %0.2f" % (k, alpha[i], beta[i])
                temp = get_tpr_tnr(lsp, k, alpha[i], beta[i])
                tpr.append(temp[0])
                tnr.append(temp[1])
        print str([round(i, 5) for i in tpr])
        print str([round(i, 5) for i in tnr])
    else:
        lsp.split_training_data(float(sys.argv[1]))
        human_data = None
        while not rospy.is_shutdown():
            human_data = lsp.test_data[
                random.randint(0, len(lsp.test_data)-1)
            ]
            prediction = lsp.predict_class_data(human_data[0])
            rospy.loginfo("The actual class is %s", human_data[1])
            if len(prediction[1]) != 0 and len(prediction[2]) != 0:
                lsp.visualize_test_between_class(
                    human_data[0].normal,
                    prediction[1][0][0].normal,
                    prediction[2][0][0].normal
                )
