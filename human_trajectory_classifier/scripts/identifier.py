#! /usr/bin/env python

import rospy
import actionlib
from multiprocessing import Process, Queue
from human_trajectory.trajectory import Trajectory
from human_trajectory.msg import Trajectories
from human_trajectory_classifier.classifier import KNNClassifier
from human_trajectory_classifier.svm_classifier import SVCClassifier
from human_trajectory_classifier.msg import HMCAction, HMCResult
from human_trajectory_classifier.msg import HumanIdentifier


class IdentifierServer(object):

    def __init__(self, name, classifier="SVM"):
        self._action_name = name
        self._classifier = classifier
        if classifier == "SVM":
            self.classifier = SVCClassifier()
        else:
            self.classifier = KNNClassifier()
        self.trajs = list()

        # Start server
        rospy.loginfo("%s is starting an action server", name)
        self._as = actionlib.SimpleActionServer(
            self._action_name,
            HMCAction,
            execute_cb=self.execute,
            auto_start=False
        )
        self._as.start()
        self._pub = rospy.Publisher(
            self._action_name+'/detections', HumanIdentifier, queue_size=10
        )
        rospy.loginfo("%s is ready", name)

    # get trajectory data
    def traj_callback(self, msg):
        self.trajs = []
        for i in msg.trajectories:
            traj = Trajectory(i.uuid)
            traj.humrobpose = zip(i.trajectory, i.robot)
            traj.length.append(i.trajectory_length)
            traj.sequence_id = i.sequence_id
            self.trajs.append(traj)

    def _knn_prediction(self, trajs):
        for i in trajs:
            human_counter = 0
            chunked_traj = self.classifier.create_chunk(
                i.uuid, list(zip(*i.humrobpose)[0])
            )
            for j in chunked_traj:
                if self._as.is_preempt_requested():
                    break
                result = self.classifier.predict_class_data(j)
                if result[0] == 'human':
                    human_counter += 1
            if self._as.is_preempt_requested():
                rospy.loginfo("The online prediction is preempted")
                break
            if len(chunked_traj) > 0:
                conf = human_counter/float(len(chunked_traj))
                human_type = 'human'
                if conf < 0.5:
                    conf = 1.0 - conf
                    human_type = 'non-human'
                self._pub.publish(HumanIdentifier(i.uuid, human_type, conf))

    def _svm_prediction(self, trajs):
        for i in trajs:
            human_counter = 0
            chunked_traj = self.classifier.create_chunk(
                list(zip(*i.humrobpose)[0])
            )
            for j in chunked_traj:
                if self._as.is_preempt_requested():
                    break
                result = self.classifier.predict_class_data(j)
                if result[-1] == 1:
                    human_counter += 1
            if self._as.is_preempt_requested():
                rospy.loginfo("The online prediction is preempted")
                break
            if len(chunked_traj) > 0:
                conf = human_counter/float(len(chunked_traj))
                human_type = 'human'
                if conf < 0.5:
                    conf = 1.0 - conf
                    human_type = 'non-human'
                self._pub.publish(HumanIdentifier(i.uuid, human_type, conf))
            rospy.sleep(0.05)

    def get_online_prediction(self):
        # Subscribe to trajectory publisher
        rospy.loginfo(
            "%s is subscribing to human_trajectories/trajectories/batch",
            self._action_name
        )
        s = rospy.Subscriber(
            "human_trajectories/trajectories/batch", Trajectories,
            self.traj_callback, None, 30
        )

        while not self._as.is_preempt_requested():
            trajs = self.trajs
            if self._classifier == "SVM":
                self._svm_prediction(trajs)
            else:
                self._knn_prediction(trajs)
        self._as.set_preempted()
        s.unregister()

    # update classifier database
    def update_db(self):
        rospy.loginfo("%s is updating database", self._action_name)
        if self._classifier == "SVM":
            self.classifier.update_database(True)
        else:
            self.classifier.update_database()
        rospy.loginfo("%s is ready", self._action_name)

    # get the overal accuracy using 5-fold cross validation
    def get_accuracy(self):
        queue = Queue()
        t = Process(target=self.classifier.get_accuracy, args=(queue,))
        t.daemon = True
        t.start()
        preempt = False
        while t.is_alive():
            if self._as.is_preempt_requested():
                queue.put({'preempt': True})
                preempt = True
                break
            rospy.sleep(0.1)
        t.join()

        if not preempt:
            rospy.loginfo("The overal accuracy is %d",
                          self.classifier.accuracy)
            self._as.set_succeeded(HMCResult(False, self.classifier.accuracy))
        else:
            rospy.loginfo("The overall accuracy request is preempted")
            self._as.set_preempted()

    # execute call back for action server
    def execute(self, goal):
        if goal.request == 'update':
            self.update_db()
            self._as.set_succeeded(HMCResult(True, 0))
        elif goal.request == 'accuracy':
            self.get_accuracy()
        else:
            self.get_online_prediction()


if __name__ == '__main__':
    rospy.init_node("human_movement_detection_server")
    sv = IdentifierServer(rospy.get_name())
    rospy.spin()
