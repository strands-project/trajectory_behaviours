#!/usr/bin/env python

"""learningArea.py: File with Learing class."""

__author__      = "Paul Duckworth"
__copyright__   = "Copyright 2015, University of Leeds"

import os, sys
import rospy
import time
import math
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from soma_geospatial_store.geospatial_store import *
from tf.transformations import euler_from_quaternion

from relational_learner.Activity_Graph import Activity_Graph
from time_analysis.cyclic_processes import *

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
        except:
            print "Loading of learnt model failed. Cannot test novelty)."

        self.roi = foo["ROI"]
        self.methods = foo["learning_methods"]
        self.code_book = foo["code_book"]
        self.graphlet_book = foo["graphlet_book"]
        self.feature_space = foo["feature_space"]
        print "Loaded: " + repr(self.methods.keys())
        print("success")


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
            for k in xrange(1, len(data)/3):
                (estimator, penalty) = self.kmeans_util(data, k) 
                if k==1: 
                    best_pen = penalty
                    best_k = 1

                if penalty < best_pen:
                    best_pen = penalty
                    best_k = k
            k = best_k
            print "k = %d has minimum inertia*penalty" %k
            
        self.methods["kmeans"] = estimator
        if self.visualise: plot_pca(data, k)
        rospy.loginfo('6. Done')


    def kmeans_util(self, data, k=None):
        n_samples, n_features = data.shape
        if self.visualise:
            print("n_samples %d, \t n_features %d, \t n_clusters %d"
                  % (n_samples, n_features, k))
            print(40 * '-')
            print('% 9s' % 'init' '         time  inertia   *Penalty')


        (estimator, pen) = self.bench_k_means(KMeans(init='k-means++', n_clusters=k, n_init=10),
                      name="k-means++", data=data, k=k)
        self.bench_k_means(KMeans(init='random', n_clusters=k, n_init=10),
                      name="random", data=data, k=k)

        # in this case the seeding of the centers is deterministic, hence we run the
        # kmeans algorithm only once with n_init=1
        pca = PCA(n_components=k).fit(data)
        self.bench_k_means(KMeans(init=pca.components_, n_clusters=k, n_init=1),
                      name="PCA-based", data=data, k=k)
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

         

    def time_analysis(self, time_points, interval=1800):
        timestamps_vec = time_wrap(time_points)[0]

        dyn_cl = dynamic_clusters()
        for t in range(len(timestamps_vec)):
            dyn_cl.add_element(t+1,timestamps_vec[t])
        fitting = activity_time(timestamps_vec, interval=interval)
        
        #plot_options: title, hist_colour, curve_colour
        #stop = fitting.display_indexes(['trajectories','g','b'],dyn_cl,[]) 
        self.methods["time_dyn_clst"] = dyn_cl
        self.methods["time_fitting"] = fitting

    def region_knowledge(self, map, config, \
                        interval=3600.0, period = 86400.0):
        """Returns the ROIs the robot can montitor at each pose"""
        t0 = time.time()
        n_bins = int(period/interval)

        gs = GeoSpatialStoreProxy('geospatial_store','soma')
        ns = GeoSpatialStoreProxy('message_store','monitored_nav_events')
        ms = GeoSpatialStoreProxy('message_store','soma_roi')

        twoproxies = TwoProxies(gs, ns, map, config)

        query = {"_id": {"$exists": "true"}}     
        for p in twoproxies.msg.find(query):
            timepoint = p['event_start_time']['secs']
            pose = p['event_start_pose']['position']
            ro, pi, yaw = euler_from_quaternion([0, 0, \
                    p['event_start_pose']['orientation']['z'], \
                    p['event_start_pose']['orientation']['w'] ])
       
            coords = robot_view_cone(pose['x'], pose['y'], yaw)
            lnglat = []
            for pt in coords:
                lnglat.append(twoproxies.gs.coords_to_lnglat(pt[0], pt[1]))
            lnglat.append(twoproxies.gs.coords_to_lnglat(\
                            coords[0][0], coords[0][1]))

            regions = []           
            for i in twoproxies.gs.observed_roi(lnglat, map, config):
                if i['soma_roi_id'] in self.roi_knowledge:
                    self.roi_knowledge[i['soma_roi_id']]+=1
                    self.roi_temp_list[i['soma_roi_id']].append(timepoint)
                else:
                    self.roi_knowledge[i['soma_roi_id']]=1
                    self.roi_temp_list[i['soma_roi_id']]=[timepoint]


        for roi in self.roi_temp_list:
            a = ms.area_of_roi(str(roi), map, config)
            view_ts, ind = time_wrap(self.roi_temp_list[roi])
            binned_timepoints = binning(view_ts, n_bins, interval)

            self.roi_temp_know[roi] = [b/a for b in binned_timepoints[:-1]]

        print "Knowledge of Regions takes: ", time.time()-t0, "  secs."
        self.knowledge_plot(n_bins)
        self.methods["roi_knowledge"] = self.roi_knowledge
        self.methods["roi_knowledge"] = self.roi_temp_know

        

    
    def time_plot(timestamps_vec, knowledge, interval=3600, period=86400, \
                        vis=False):
        pc = []
        pf = []
        for v in timestamps_vec:
            pc.append(self.methods["time_dyn_clst"].query_clusters(v))
            pf.append(self.methods["time_fitting"].query_model(v))

        plt.plot(timestamps_vec,pc,label='dynamic clustering')
        plt.plot(timestamps_vec,pf,label='GMM fitting')
        plt.plot(np.arange(0,period+1,interval), knowledge, label='knowledge')
        plt.xlabel('samples')
        plt.ylabel('probability')
        plt.legend()
        plt.savefig('/home/strands/STRANDS/learning/roi12.jpg', \
                bbox_inches='tight', dpi=100)



    def knowledge_plot(self, n_bins):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        z = 0
        cl = ['r', 'g', 'b', 'y']
        regions=[]
        for (roi, k) in self.roi_temp_know.items():
            print k
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
        plt.savefig('/home/strands/STRANDS/learning/roi_knowledge.jpg', \
                bbox_inches='tight', dpi=100)

    
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

