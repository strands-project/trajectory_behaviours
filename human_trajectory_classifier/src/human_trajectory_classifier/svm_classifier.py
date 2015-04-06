#!/usr/bin/env python

import sys
import rospy
import math
import pymongo
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from sklearn import svm as SVM
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc
from geometry_msgs.msg import Point, Quaternion, Pose
from human_trajectory.trajectory import Trajectory
from std_msgs.msg import Header


class SVCClassifier(object):

    def __init__(self):
        self.svm = SVM.SVC(cache_size=1000, C=1000, gamma=10)
        self.accuracy = 0
        self.training = list()
        self.test = list()
        self.label_train = list()
        self.label_test = list()

    def get_tpr_tnr(self):
        rospy.loginfo("Constructing tpr, tnr...")
        all_tpr = dict()
        all_tnr = dict()
        # calculate accuracy of all combination of C and gamma
        for i in [0.1, 1, 10, 100, 1000]:
            temp = dict()
            temp2 = dict()
            for j in [0.01, 0.1, 1, 10, 100]:
                self.svm = SVM.SVC(cache_size=2000, C=i, gamma=j)
                self._fitting_classifier()
                tp = 0
                fp = 0
                tn = 0
                fn = 0
                for k, v in enumerate(self.test):
                    prediction = self.predict_class_data(v)
                    if prediction[-1] == 1 and self.label_test[k] == 1:
                        tp += 1
                    elif prediction[-1] == 1 and self.label_test[k] == 0:
                        fp += 1
                    elif prediction[-1] == 0 and self.label_test[k] == 0:
                        tn += 1
                    else:
                        fn += 1
                tpr = tp / float(tp + fn)
                tnr = tn / float(fp + tn)
                print "C: %0.1f, Gamma:%0.2f, TPR: %0.5f, TNR: %0.5f" % (i, j, tpr, tnr)
                temp[j] = tpr
                temp2[j] = tnr
            all_tpr[i] = temp
            all_tnr[i] = temp2

        return all_tpr, all_tnr

    # produce roc curve by varying C and gamma
    def get_roc_curve(self):
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        # calculate accuracy of all combination of C and gamma
        for i in [0.1, 1, 10, 100, 1000]:
            for j in [0.01, 0.1, 1, 10, 100]:
                self.svm = SVM.SVC(cache_size=2000, C=i, gamma=j)
                self._fitting_classifier(True)
                # predict with probability, to be fed to roc_curve
                prediction = self.svm.predict_proba(self.test)
                fpr, tpr, threshold = roc_curve(
                    self.label_test, prediction[:, 1]
                )
                mean_tpr += interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
                roc_auc = auc(fpr, tpr)
                # plot the result
                plt.plot(
                    fpr, tpr, lw=1,
                    label='C=%0.2f, Gamma=%0.3f (area = %0.2f)' % (
                        i, j, roc_auc
                    )
                )

        # calculate the average roc value
        mean_tpr /= 25
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(
            mean_fpr, mean_tpr, 'k--',
            label='Mean ROC (area = %0.2f)' % mean_auc, lw=2
        )

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Human Trajectory')
        plt.legend(loc='lower right')
        plt.show()

    # fitting the training data with label and print the result
    def _fitting_classifier(self, probability=False):
        rospy.loginfo("Fitting the training data...")
        if probability:
            self.svm.probability = True
        else:
            self.svm.probability = False
        print self.svm.fit(self.training, self.label_train)

    # predict the class of the test data
    def predict_class_data(self, test):
        return self.svm.predict(test)

    # get the mean accuracy of the classifier
    def get_accuracy(self, queue=None):
        preempt = False
        rospy.loginfo("Getting the accuracy...")
        if queue is not None and not queue.empty():
            preempt = queue.get()['preempt']
        if not preempt:
            self.accuracy = self.svm.score(self.test, self.label_test)
        return self.accuracy

    # update training and test data from database
    def update_database(self, fitting=False):
        rospy.loginfo("Updating database...")
        self.training = []
        self.test = []
        trajs = self._retrieve_logs()
        self._label_data(trajs)
        if fitting:
            self._fitting_classifier()

    # splitting training data into training and test data
    def split_training_data(self, training_ratio):
        rospy.loginfo("Splitting data into test and training...")
        (a, b, c, d) = cross_validation.train_test_split(
            self.training, self.label_train,
            train_size=training_ratio, random_state=0
        )
        self.training = a
        self.test = b
        self.label_train = c
        self.label_test = d
        self._fitting_classifier()

    # labeling data
    def _label_data(self, trajs):
        rospy.loginfo("Splitting data into chunk...")
        for uuid, traj in trajs.iteritems():
            traj.validate_all_poses()
            chunked_traj = self.create_chunk(list(zip(*traj.humrobpose)[0]))
            label = 1
            start = traj.humrobpose[0][0].header.stamp
            end = traj.humrobpose[-1][0].header.stamp
            delta = float((end-start).secs + 0.000000001 * (end-start).nsecs)
            if delta != 0.0:
                avg_vel = traj.length[-1] / delta
            else:
                avg_vel = 0.0
            guard = traj.length[-1] < 0.1 or avg_vel < 0.5 or avg_vel > 1.5
            if guard:
                label = 0
            for i in chunked_traj:
                self.training.append(i)
                self.label_train.append(label)

    # normalize poses so that the first pose becomes (0,0)
    # and the second pose becomes the base for the axis
    # with tangen, cos and sin
    def get_normalized_poses(self, poses):
        dx = poses[1][0] - poses[0][0]
        dy = poses[1][1] - poses[0][1]
        if dx < 0.00001:
            dx = 0.00000000000000000001
        rad = math.atan(dy / dx)
        for i, j in enumerate(poses):
            if i > 0:
                dx = j[0] - poses[0][0]
                dy = j[1] - poses[0][1]
                if dx < 0.00001:
                    dx = 0.00000000000000000001
                rad2 = math.atan(dy / dx)
                delta_rad = rad2 - rad
                if rad2 == 0:
                    r = dx / math.cos(rad2)
                else:
                    r = dy / math.sin(rad2)
                poses[i][0] = r * math.cos(delta_rad)
                poses[i][1] = r * math.sin(delta_rad)

        poses[0][0] = poses[0][1] = 0
        return poses

    # chunk data for each trajectory, return x, y position for both
    # original and normalized poses
    def create_chunk(self, poses, chunk=20):
        i = 0
        chunk_trajectory = list()
        while i < len(poses) - (chunk - 1):
            temp = list()
            for j in range(chunk):
                temp.append([
                    poses[i + j].pose.position.x,
                    poses[i + j].pose.position.y
                ])
            temp = self.get_normalized_poses(temp)
            normalized = list()
            for k in temp:
                normalized.append(k[0])
                normalized.append(k[1])
            chunk_trajectory.append(normalized)
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


if __name__ == '__main__':
    rospy.init_node("svm_trajectory_classifier")

    if len(sys.argv) < 3:
        rospy.logerr(
            "usage: classifier train_ratio [score | roc]"
        )
        sys.exit(2)

    svmc = SVCClassifier()
    svmc.update_database()
    svmc.split_training_data(float(sys.argv[1]))

    if sys.argv[2] == 'score':
        rospy.loginfo("The overall accuracy is " + str(svmc.get_accuracy()))
    else:
        svmc.get_roc_curve()
