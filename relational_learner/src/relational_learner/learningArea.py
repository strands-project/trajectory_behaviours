#!/usr/bin/env python

"""learningArea.py: File with Learing class."""

__author__      = "Paul Duckworth"
__copyright__   = "Copyright 2015, University of Leeds"

import os, sys
import rospy
import datetime, time
import math
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial import distance
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from soma_geospatial_store.geospatial_store import *
from tf.transformations import euler_from_quaternion

from relational_learner.Activity_Graph import Activity_Graph
from time_analysis.cyclic_processes import *

from relational_learner.msg import *
from mongodb_store.message_store import MessageStoreProxy

class RegionKnowledgeImporter(object):
    def __init__(self):
        rospy.loginfo("Connecting to mongodb...")
        self._client = pymongo.MongoClient(rospy.get_param("mongodb_host"),
                                           rospy.get_param("mongodb_port"))
        self._db = "message_store"
        self._collection = "region_knowledge"
        self._store_client = MessageStoreProxy(collection=self._collection)

    def find(self, query_json):
        return self._client[self._db][self._collection].find(query_json)


class Learning():
    '''
    Unsupervised Learning Class:
    Accepts a feature space, where rows are instances, and columns are features.
    '''

    def __init__(self, f_space=None, roi="",
                    vis=False, load_from_file=None):

        if load_from_file is not None and load_from_file != "":
            self.load(load_from_file)
        else:

            (self.code_book, self.graphlet_book, \
                      self.feature_space) = f_space
            self.methods = {}
            self.roi = roi
            self.visualise = vis
            self.roi_knowledge = {}
            self.roi_temp_list = {}
            self.roi_temp_know = {}

    def save(self, dir):
        print("Saving...")
        print self.roi
        filename = os.path.join(dir, self.roi + '_smartThing.p')

        foo = { "ROI": self.roi, "feature_space": self.feature_space, \
                "code_book": self.code_book, "graphlet_book": self.graphlet_book, \
                "learning_methods": self.methods}
        print(filename)
        with open(filename, "wb") as f:
            pickle.dump(foo, f)
        print("success")

    
    def load(self, filename):
        print("Loading Learning from", filename)
        try:
            with open(filename, "rb") as f:
                foo = pickle.load(f)
            self.roi = foo["ROI"]
            self.methods = foo["learning_methods"]
            self.code_book = foo["code_book"]
            self.graphlet_book = foo["graphlet_book"]
            self.feature_space = foo["feature_space"]
            self.flag = True
            print "Loaded: " + repr(self.methods.keys())
            print("success")

        except:
            print "Loading of learnt model failed. Cannot test novelty)."
            self.flag = False


    def kmeans(self, k=None):
        np.random.seed(42)
        X = np.array(self.feature_space)        
        #scaler = StandardScaler()
        #scaler.fit(X)
        #X_s = scaler.transform(X)        
        data = X #_s
        if k!=None:
            (estimator, penalty) = self.kmeans_util(data, k=k)
        else:
            print "Automatically selecting k"
            #self.visualise = True
            min_k = 2
            #loop from k=2 until a third of the datapoints
            for k in xrange(min_k, int(len(data)/3)+1):
                (estimator, penalty) = self.kmeans_util(data, k) 
                if k==min_k: 
                    (best_e, best_p, best_k) = estimator, penalty, k
                if penalty < best_p:
                    (best_e, best_p, best_k) = estimator, penalty, k
            estimator, penalty, k = (best_e, best_p, best_k)

            print "k = %d has minimum inertia*penalty" %k
        
    
        estimator = self.kmeans_cluster_radius(data, estimator)
        
        self.methods["kmeans"] = estimator
        if self.visualise: plot_pca(data, k)
        rospy.loginfo('Done\n')


    def kmeans_cluster_radius(self, data, estimator):
        n_samples, n_features = data.shape
        print "sum of inertias = ", estimator.inertia_
        #print "CLUSTER CENTERS = ", estimator.cluster_centers_
        print "sample labels = ", estimator.labels_
    
        cluster_radius = {}
        for i, sample in enumerate(data):
            #print "Sample = ", i
            label = estimator.labels_[i]
            clst = estimator.cluster_centers_[label]
            dst = distance.euclidean(sample,clst)

            if label not in cluster_radius:
                cluster_radius[label] = [dst]
            else:
                cluster_radius[label].append(dst)

        means, std = {}, {}
        for label in cluster_radius:
            means[label] = np.mean(cluster_radius[label])
            std[label] = np.std(cluster_radius[label])
        print "avg distance to clusters", means
        print "std distance to clusters", std
        
        estimator.cluster_dist_means = means
        estimator.cluster_dist_std = std

        return estimator

    def kmeans_util(self, data, k=None):
        n_samples, n_features = data.shape
        if self.visualise:
            print("n_samples %d, \t n_features %d, \t n_clusters %d"
                  % (n_samples, n_features, k))
            print(40 * '-')
            print('% 9s' % 'init' '         time  inertia   *Penalty')


        (estimator, pen) = self.bench_k_means(KMeans(init='k-means++', n_clusters=k, n_init=10),
                      name="k-means++", data=data, k=k)
        #self.bench_k_means(KMeans(init='random', n_clusters=k, n_init=10),
        #              name="random", data=data, k=k)

        # in this case the seeding of the centers is deterministic, hence we run the
        # kmeans algorithm only once with n_init=1
        #pca = PCA(n_components=k).fit(data)
        #self.bench_k_means(KMeans(init=pca.components_, n_clusters=k, n_init=1),
        #              name="PCA-based", data=data, k=k)
        if  self.visualise: print(40 * '-')
        return (estimator, pen)




    def bench_k_means(self, estimator, name, data, k):
        t0 = time.time()
        estimator.fit(data)
        #penalty = estimator.inertia_*math.sqrt(k)
        penalty = estimator.inertia_*k
        if  self.visualise: print('% 9s   %.2fs    %i     %i'
                % (name, (time.time() - t0), estimator.inertia_, penalty))
        return (estimator, penalty)

         

    def time_analysis(self, time_points, plot=False, interval=1800):
        """Number of seconds in a day = 86400"""

        first_day = int(min(time_points)/86400)

        dyn_cl = dynamic_clusters()
        for t in time_points:
            day = int(t/86400)-first_day+1
            #print day
            time_in_day = t%86400   #in seconds
            dyn_cl.add_element(day,time_in_day) 

        timestamps_vec = time_wrap(time_points)[0]    
        fitting = activity_time(timestamps_vec, interval=interval)

        self.methods["time_dyn_clst"] = dyn_cl
        self.methods["time_fitting"] = fitting
        if plot: self.temporal_plot(vis=False)
        rospy.loginfo('Done\n')

    def temporal_plot(self, plot_interval=900.0, period=86400, \
                        vis=False):
        pc = []
        pf = []
        #query model at these points to generate graph:
        timestamps = np.arange(0,period,plot_interval) 
        for v in timestamps:
            pc.append(self.methods["time_dyn_clst"].query_clusters(v))
            pf.append(self.methods["time_fitting"].query_model(v))
            
        plot_vec = [t/3600.0 for t in timestamps]
        fig = plt.figure()
        t_ax = fig.add_subplot(111)
        width=0.01
        plt.bar(plot_vec,pc,width, color='r', edgecolor='r',\
              label='dynamic clustering')
        plt.plot(plot_vec,pf,label='GMM fitting')

        plt.xlim([0,24])
        t_ax.set_xticks(np.arange(0,24))

        plt.xlabel('time of day')
        plt.ylabel('probability')
        plt.legend()
        filename='/tmp/temporal_plot_%s.jpg' % self.roi
        plt.savefig(filename, bbox_inches='tight', dpi=100)



def region_knowledge(map, config, interval=3600.0, period = 86400.0,\
                    sampling_rate=10, plot=False):

    """Returns the ROIs the robot has montitor at each logged robot pose"""
    t0 = time.time()
    n_bins = int(period/interval)

    ##Get info stored in Mongodb Region Knowledge Store
    ks = RegionKnowledgeImporter()
    roi_knowledge = {}
    roi_temp_list = {}

    query = {"soma_roi_id"  : {"$exists": "true"}, 
            "roi_knowledge" : {"$exists": "true"}}
    for region in ks.find(query):
        roi_knowledge[str(region["soma_roi_id"])] = int(region["roi_knowledge"])
        roi_temp_list[str(region["soma_roi_id"])] = region["roi_knowledge_hourly"]

    print "existing knowledge = ", roi_knowledge
    print "existing hourly knowledge = ", roi_temp_list

    ##Query the Robot Poses from roslog
    gs = GeoSpatialStoreProxy('geospatial_store','soma')
    ms = GeoSpatialStoreProxy('message_store','soma_roi')
    roslog = GeoSpatialStoreProxy('roslog','robot_pose')

    query = {"_id": {"$exists": "true"}}
    print "sampling rate =", sampling_rate

    ##Loop through the robot poses for the day
    for cnt, p in enumerate(roslog.find(query)):
        if cnt % sampling_rate != 0: continue   #Take 1/10 of the roslog poses
        timepoint = cnt/sampling_rate

        #print p
        pose = p['position']
        ro, pi, yaw = euler_from_quaternion([0, 0, \
                p['orientation']['z'], p['orientation']['w'] ])

        inserted_at = p['_meta']['inserted_at']
        hour = inserted_at.time().hour

        coords = robot_view_cone(pose['x'], pose['y'], yaw)
        lnglat = []
        for pt in coords:
            lnglat.append(gs.coords_to_lnglat(pt[0], pt[1]))
        #add first points again to make it a complete polygon
        lnglat.append(gs.coords_to_lnglat(coords[0][0], coords[0][1]))

        for i in gs.observed_roi(lnglat, map, config):
            region = str(i['soma_roi_id'])

            #Region Knowledge
            if region in roi_knowledge:
                roi_knowledge[region]+=1
            else:
                roi_knowledge[region]=1
                
            #Region Knowledge per hour. Bin them by hour.
            if region in roi_temp_list: 
                roi_temp_list[region][hour]+=1
            else:
                roi_temp_list[region]=[0]*24
                roi_temp_list[region][hour] = 1

    print "roi_knowledge = ", roi_knowledge
    print "roi_temporal_knowledge = ", roi_temp_list

    #update mongodb (as the roslog/robot_pose data is removed at the end of the day)
    for roi, score in roi_knowledge.items():
        region_type = gs.type_of_roi(roi, map, config)

        msg = RegionKnowledgeMsg(soma_roi_id = roi, type = str(region_type), \
                   roi_knowledge = score, roi_knowledge_hourly = roi_temp_list[roi])

        query = {"soma_roi_id" : roi}
        #print "MESSAGE = \n", msg
        #print "query = ", query
        p_id = ks._store_client.update(message=msg, message_query=query, meta={}, upsert=True)

    print "Knowledge of Regions takes: ", time.time()-t0, "  secs."
    if plot: knowledge_plot(roi_temp_list, n_bins)

    rospy.loginfo('Done')
    return roi_knowledge, roi_temp_list


def knowledge_plot(roi_temp_list, n_bins):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    z = 0
    cl = ['r', 'g', 'b', 'y']
    regions=[]
    for (roi, k) in roi_temp_list.items():
        #print k
        regions.append(roi)
        cls = [cl[z%4]]*n_bins
        ax.bar(range(n_bins),k, zs=z, zdir='y', color=cls, alpha = 0.8)
        z = z+1
    ax.set_ylabel("ROI")
    ax.set_xlabel("time")
    ax.set_zlabel("observation (secs)/area (m^2)")
    ax.set_xticks([0,3,6,9,12,15,18,21,24])
    ax.set_yticks(range(1,len(regions)+1))
    ax.set_yticklabels(regions)
    date = time.strftime("%x").replace("/","_")
    filename='/tmp/roi_knowledge__%s.jpg' % date
    plt.savefig(filename, bbox_inches='tight', dpi=100)

    
def robot_view_cone( Px, Py, yaw):
    """ let's call the triangle PLR, where P is the robot pose, 
        L the left vertex, R the right vertex"""
    d = 4 # max monitored distance: reasonably not more than 3.5-4m
    alpha = 1 # field of view: 57 deg kinect, 58 xtion, we can use exactly 1 rad (=57.3 deg)
    Lx = Px + d * (math.cos((yaw-alpha)/2))
    Ly = Py + d * (math.cos((yaw-alpha)/2))
    Rx = Px + d * (math.cos((yaw+alpha)/2))
    Ry = Py + d * (math.cos((yaw+alpha)/2))
    return [ [Lx, Ly], [Rx, Ry], [Px, Py] ]



def plot_pca(data, k):
    ###############################################################################
    # Visualize the results on PCA-reduced data
    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
    y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')


    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    centroids = kmeans.cluster_centers_         # Plot the centroids as a white X
    print centroids
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the Resource Room/trajectories data (PCA-reduced)\n'
             'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())  
    plt.show()



if __name__ == "__main__":
    rospy.init_node('learningArea')

    data_dir = '/home/strands/STRANDS/'
    file_ = os.path.join(data_dir + 'learning/roi_12_smartThing.p')
    print file_

    smartThing=Learning(load_from_file=file_)
    print smartThing.methods
    print smartThing.code_book

