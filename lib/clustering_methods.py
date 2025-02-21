import numpy as np
import numpy.ma as ma
from sklearn.cluster import MiniBatchKMeans

def kmeans_knn_partition(tseries,n_seeds,batchsize=None,return_centers=False):
    if batchsize==None:
        batchsize = n_seeds*5
    if ma.count_masked(tseries)>0:
        labels = ma.zeros(tseries.shape[0],dtype=int)
        labels.mask = np.any(tseries.mask,axis=1)
        kmeans = MiniBatchKMeans(batch_size=batchsize,n_clusters=n_seeds).fit(ma.compress_rows(tseries))
        labels[~np.any(tseries.mask,axis=1)] = kmeans.labels_
    else:
        kmeans = MiniBatchKMeans(batch_size=batchsize,n_clusters=n_seeds).fit(tseries)
        labels=kmeans.labels_
    if return_centers:
        return labels,kmeans.cluster_centers_
    return labels


def kmedoids_clustering(sample,n_clusters,tol = .001):
    '''
    Compute the labels of each point, the location of the medoids, and the largest epsilon
    Using Chebyshev metric (maxnorm)
    '''
    from pyclustering.cluster.kmedoids import kmedoids
    from pyclustering.utils.metric import distance_metric,type_metric

    
    if ma.count_masked(sample)>0:
        labels = ma.zeros(sample.shape[0],dtype=int)
        labels.mask= np.any(sample.mask,axis=1)
        data = ma.compress_rows(sample)
        

        metric = distance_metric(type_metric.CHEBYSHEV)
        initial_medoid_indices = np.random.choice(np.arange(len(data)),n_clusters,replace=False)
        # Create instance of K-Medoids algorithm.
        kmedoids_instance = kmedoids(data, initial_medoid_indices, metric=metric,tolerance = tol)
        # Run cluster analysis and obtain results.
        kmedoids_instance.process()
        clusters = kmedoids_instance.get_clusters()
        medoids = kmedoids_instance.get_medoids()

        labels_data = np.arange(len(data))
        for kc,cluster in enumerate(clusters):
            labels_data[cluster] = kc

        max_epsilon = np.mean([np.max(np.abs(data[clusters[kc]]-data[medoids[kc]])) for kc in range(len(clusters))])

        labels[~np.any(sample.mask,axis=1)] = labels_data
        
    else:
        metric = distance_metric(type_metric.CHEBYSHEV)
        initial_medoid_indices = np.random.choice(np.arange(len(sample)),n_clusters,replace=False)
        # Create instance of K-Medoids algorithm.
        kmedoids_instance = kmedoids(sample, initial_medoid_indices, metric=metric,tolerance = tol)
        # Run cluster analysis and obtain results.
        kmedoids_instance.process()
        clusters = kmedoids_instance.get_clusters()
        medoids = kmedoids_instance.get_medoids()

        labels = np.arange(len(sample))
        for kc,cluster in enumerate(clusters):
            labels[cluster] = kc

        max_epsilon = np.mean([np.max(np.abs(sample[clusters[kc]]-sample[medoids[kc]])) for kc in range(len(clusters))])

    return labels,medoids,max_epsilon