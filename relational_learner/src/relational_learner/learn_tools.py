
import os, sys
import numpy as np
import scipy.io
import cPickle as pickle
import itertools
import warnings

from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn import svm
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score


def generate_weka_file(features, labels, out_file):
    """ Generate weka arff file from list of features and labels
    """
    weka_file_handle = open(out_file, 'w')
    classes = set(labels)
    num_of_features = len(features[0])
    weka_file_handle.write('@RELATION cad120\n')
    weka_file_handle.write('\n')
    for i in range(num_of_features):
        attrib_string = '@ATTRIBUTE %d NUMERIC\n' %(i)
        weka_file_handle.write(attrib_string)
    class_attrib_string = '@ATTRIBUTE class {' + ','.join(classes) + '}\n'
    weka_file_handle.write(class_attrib_string)
    weka_file_handle.write('@DATA\n')
    for i in range(len(features)):
        features_str = map(str, features[i])
        instance_str = ','.join(features_str) + ',' + labels[i] + '\n'
        weka_file_handle.write(instance_str)
    weka_file_handle.close()    

    return



def cosineSimilarityAndAF(X, labels_true):   
    from sklearn.cluster import AffinityPropagation
    from sklearn import metrics
    
    
    ##############################################################################
    # Generate sample data
    centers = [[1, 1], [-1, -1], [1, -1]]
    
    ##############################################################################
    # Compute Affinity Propagation
    af = AffinityPropagation(preference=-50).fit(X)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    
    n_clusters_ = len(cluster_centers_indices)
    
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels, metric='sqeuclidean'))
    
    ##############################################################################
    # Plot result
    import matplotlib.pyplot as plt
    from itertools import cycle
    
    plt.close('all')
    plt.figure(1)
    plt.clf()
    
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        cluster_center = X[cluster_centers_indices[k]]
        plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
        for x in X[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
    
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
 
    return
    

        """ PCA TEST """
        """
        pca = PCA(n_components=0.95, whiten=False)
        pca.fit(all_max_X_source[X_test],  X_source_labels[X_test])     
        x_train = pca.transform(all_max_X_source[X_test])
        #pca.components_
        
        #pca_graphbook = pca.transform(graphlet_books[X_test])
        #weka_path = os.path.join('/'.join(exp_data_file.split('/')[:-2]) + '/weka/'+ X_test + "330PCA.arff")
        #generate_weka_file(X, X_source_labels[X_test], weka_path)
        """
    
    
def feature_selection_univariate(train_features, train_labels, selector, percent=50):
    selector = SelectPercentile(f_classif, percentile=percent)
    new_train_features = selector.fit_transform(train_features, train_labels)   
     
    return new_train_features, selector


