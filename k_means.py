''' k means'''

import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import  sklearn.metrics.pairwise
from sklearn.neighbors import NearestNeighbors
import copy
import time

import json
import os
os.system('cls')

# load csv data
# the file have 3 columns: URI,name,text for many peopls
wiki = pd.read_csv('people_wiki.csv')
#insert a new column as an indicator
wiki['id'] = np.array([i for i in range(wiki.shape[0])])

# load the tf_idf file
loader = np.load('people_wiki_tf_idf.npz')
#### result is [shape, data, indices, indptr]
# shape = [59071 547979] which is the number of pages(text) and number of words
shape = loader['shape']
# all nonzero elements and len(data) = NNZ = 10379283
data = loader['data']
# columns which NZ element are there so it is between 0,...,547979
# and len(indices) == NNZ
indices = loader['indices']
# it is comulative sum of NZ elements in each raw
# len(indptr) = # texts +1 = 59072
indptr = loader['indptr']
# convert compress sparse raw(csr) to non compress
tf_idf = csr_matrix( (data, indices, indptr), shape)


# k means is formulated apon euclidean distance rather than cosine distane
# however, euclidean pays attention to the lentgh of the article and penalize \
#long articles unfairly
# to disregard the lentgh of the article, we normalize vectors
# it can be proved that euclidean distance between two unit vectors is\
# proportional to their cosine distance
tf_idf = normalize(tf_idf)


# initialize cluster centers
def get_initial_centroids(data, k, seed = None):
    ''' randomly choose k data points as initial centroids'''
    if seed is not None:
        np.random.seed(seed)
    # choose k random integers from[0 , # of documents]
    rand_indices = np.random.randint(0, data.shape[0], k)
    # convert to dense format
    initial_centroids = data[ rand_indices].toarray()
    return initial_centroids

def assign_clusters( data, centroids):
    ''' for a fixed sets of centroids, find assignements that
    minimaize sum(over j)(sum(over x belongs to jth cluster)(centroids(j) - xi))
    '''
    # compute distance between each data and centroid
    distance = sklearn.metrics.pairwise.euclidean_distances\
    (data, centroids, squared = True)
    # for each data point find the centroid with the least distance
    # and assign data point to that cluster
    z = np.argmin(distance, axis=1)
    return z


def revise_centroids(z, data, k):
    ''' for a fixed set of assignments find cluster centers to minimize
     sum(over j)(sum(over x belongs to jth cluster)(centroids(j) - xi))
    k = number of clusters
    '''
    new_centroids = np.zeros((k, data.shape[1]))
    for j in range(k):
        # select datas that are assigned to that cluster
        members_j = data[ z==j ]
        # and average over columns
        new_centroid_j = members_j.mean(axis=0 )
        new_centroids[j] = new_centroid_j
    return new_centroids

def compute_heterogeneity(data, z, centroids, k):
    ''' compute heterogeneity ( dissimilarities between datas in each group)
    this is the objective function of k means cluster
    '''
    quality = 0
    for j in range(k):
        # for every cluster select data that are assigned to that cluster
        members_j = data[z == j]
        if members_j.shape[1] > 0:
            # if at least there is one data point in thet cluster
            # ompute distance between all data points & centroid of thet cluster
            distance_ji = sklearn.metrics.pairwise.euclidean_distances\
            (members_j, centroids[j].reshape(1, data.shape[1]), squared=True)
            # sum distance over all data points in the cluster
            distance_j = np.sum(distance_ji)
            # sum over all clusters
            quality = quality + distance_j
    return quality

#main function
def kmeans( data, k, initial_centroids, max_iter, verbose = False, heterogeneity_history = None):
    ''' function to compute cluster centers and assignments to each cluster
    Input:
    max_iter: maximum number of iteration till stopping
    verbose : if true print heterogeneity number of changes in assignments for
    each iteration
    Outout:
    z and centroids : assinments to clusters and  center of clusters
    heterogeneity_history and number_changes for observing convergence
    '''

    # copy initial_centroids
    centroids = initial_centroids[:]
    # initial assignments
    z = np.zeros(data.shape[0])
    heterogeneity_history = []
    number_changes = []

    for iter in range(max_iter):
        if verbose :
            print "iteration", iter
        # make a copy of assinments
        old_assignments = z[:]
        # find new assignments
        z = assign_clusters( data, centroids)
        # find new centriods
        centroids = revise_centroids(z, data, k)
        # number of changes in assinments
        number_changes.append(np.sum(z!=old_assignments))
        if verbose:
            print 'elements changed their cluster assignment', number_changes[iter]
        if heterogeneity_history is not None:
            # compute heterogeneity
            quality = compute_heterogeneity(data, z, centroids, k)
            # make a list of heterogeneity over iteration
            heterogeneity_history.append(quality)
            if verbose:
                print '  heterogeneity.', quality

        # check for stopping condition
        # when there is no change in assignments
        if np.array_equal(z, old_assignments):
            break
    return z, centroids

# k = 3
# initial_centroids = get_initial_centroids(tf_idf, k, seed = 0)
# (a,b,c,d) = kmeans( tf_idf, k, initial_centroids, 15)
# for i in range(k):
#     print sum(a == i)
# # plot "number of changes in assignments" & heterogeneity across iteration
# plt.plot(c, linewidth = 4)
# plt.xlabel('# of iteration')
# plt.ylabel('heterogeneity')
# plt.figure()
# plt.plot(d, linewidth = 4)
# plt.xlabel('iteration')
# plt.ylabel('number of changes in assignments')
# plt.show()

# k = 10
# start = time.time()
# b = []
# heterogeneity = []
# for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
#     initial_centroids = get_initial_centroids(tf_idf, k, seed )
#     z, centroids= kmeans( tf_idf, k, initial_centroids, max_iter =400 )
#     quality = compute_heterogeneity(tf_idf, z, centroids, k)
#     print " seed : % 10d , final heterogeneity : %10f " %( seed, quality )
#     heterogeneity.append(quality)
#     a = []
#     for i in range(k):
#         a.append(sum(z == i))
#     b.append((min(a), max(a)))
# stop = time.time()
# print (stop - start)
# print b


def smart_initialization(data, k , seed = None):
    if seed is not None:
        np.random.seed(seed)
    # as first centroid, randomly select one of the data points
    c1 = np.random.randint(0,data.shape[0], 1 )
    # make a list of indices of data that are considered as centroids
    c = []
    # because of shape of c1
    c.append(c1[0])

    for j in range(1,k):
        # for each x compute distance to all cluster
        dis_ij = sklearn.metrics.pairwise.euclidean_distances(data, data[c], squared = True)
        # for each data x compute distance to nearest clusters
        dis_nearest_cluster = np.amin(dis_ij, axis=1)
        # normalizes distance**2 since they are considered as probability
        dis_normalized = (dis_nearest_cluster/ np.sum(dis_nearest_cluster))
        # first change the shape of dis_normalized from (D,1) to (1,D)
        # choose a centroid from datapoints with probability proportional to\
        # (distance from nearest centroids)^2
        # thus, dtp which are far from all centroids are likely to be chosen
        c_new = np.random.choice\
        (data.shape[0], p =(dis_normalized.reshape(1,data.shape[0]))[0])
        # append the index of chosen data/centroid to list of indices
        c.append(c_new)
    initial_centroids = data[c].toarray()
    return initial_centroids

# print "-"*70
# print " results with smart initialization"
# k = 10
# c = []
# heterogeneity_smart = []
# start = time.time()
# for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
#     initial_centroids = smart_initialization(tf_idf, k, seed )
#     z, centroids = kmeans( tf_idf, k, initial_centroids, max_iter =400 )
#     quality = compute_heterogeneity(tf_idf, z, centroids, k)
#     print " seed : % 10d , final heterogeneity : %10f " %( seed,quality )
#     heterogeneity_smart.append(quality)
#     a = []
#     for i in range(k):
#         a.append(sum(z == i))
#     c.append((min(a), max(a)))
# stop = time.time()
# print (stop - start)
# print c

# plt.boxplot([heterogeneity, heterogeneity_smart], vert=False)
# plt.yticks([1, 2], ['k-means', 'k-means++'])
# plt.show()

def kmeans_multiple_runs( data, k, num_runs, max_iter, seed_list = None, verbose = False):
    ''' run kmeans for multiple seeds with smart initialization and select\
    the result of the one leadinig to less heterogeneity
    verbose: print heterogeneity in each run if True
    '''
    # initialize variables
    best_seed = None
    best_assignments = None
    best_centroids = None
    min_heterogeneity = float('inf')
    # if seed list is provided choose min of (length of list and num_runs) as
    # number of runs
    if seed_list is  not None:
        if len(seed_list) < num_runs:
            print " length of given seed_list is less than num_runs"
            print " the program will run for len(seed_list) instead of num_runs"
            num_runs = len(seed_list)

    for iter in range(num_runs):
        # use UTC timer if no list is provided for seed
        if seed_list is not None:
            seed = seed_list[iter]
        else:
            seed = int(time.time())

        # initize centroids in a smart way
        initial_centroids = smart_initialization(data, k , seed = seed)

        # run knn and find assinments and centers of clusters
        z, centroids = kmeans( data, k, initial_centroids, max_iter,\
         verbose = False, heterogeneity_history = None)

        # compute heterogeneity
        heterogeneity = compute_heterogeneity(data, z, centroids, k)

        if verbose:
            print " seed : % 15f , heterogeneity : %15s " %(seed, heterogeneity)

        # look for minimum heterogeneity and store best seed and clusters
        if heterogeneity < min_heterogeneity:
            best_assignments = z
            best_centroids = centroids
            best_seed = seed
            min_heterogeneity = heterogeneity

    return(best_assignments, best_centroids, min_heterogeneity)

# start = time.time()
# k_list = [2, 10, 25, 50, 100]
# seed_list = [0, 20000, 40000, 60000, 80000, 100000, 120000]
# centroids_dic = {}
# assinments_dic = {}
# heterogeneity_values =[]
# for k in k_list:
#     (z,c,h) = kmeans_multiple_runs( tf_idf, k, len(seed_list), 400, seed_list = seed_list, verbose = False)
#     centroids_dic[k] = c
#     assinments_dic[k] = z
#     heterogeneity_values.append(h)
#     print " k : % 10d , heterogeneity : %15s " %(k,h)
#
# plt.plot(k_list, heterogeneity_values)
# stop = time.time()
# print " it took to complete code", (stop - start)
# plt.show()

# with precomputed numpy array do the same: z,center, heterogeneity
k_list = [2, 10, 25, 50, 100]
arrays = np.load('kmeans-arrays.npz')
print arrays.keys()
cluster_assinmen = {}
centroids ={}
heterogeneity_values = []
for k in k_list:
    cluster_assinmen[k] = arrays['cluster_assignment_{:d}'.format(k)]
    centroids[k] = arrays['centroids_{:d}'.format(k)]
    quality = compute_heterogeneity(tf_idf,cluster_assinmen[k], centroids[k], k)
    heterogeneity_values.append(quality)

plt.plot(k_list, heterogeneity_values, linewidth = 4)
plt.title('heterogeneity VS number of clusters')
plt.xlabel(' number of clusters')
plt.ylabel('heterogeneity')


k = 10

model = NearestNeighbors(metric= 'euclidean' , algorithm = 'brute')
model.fit(tf_idf[cluster_assinmen[k] == 1])
distances, indices = model.kneighbors((centroids[1])[1], n_neighbors = 10)
print wiki.filter(items = indices, axis =0)
