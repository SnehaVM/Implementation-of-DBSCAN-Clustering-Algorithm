import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

filename = 'train.dat.txt'
with open(filename, "r") as fh:
     data = fh.readlines()

train_arr = []
train_arr = [l.split() for l in data]


#build a csr matrix from the data
#no.of rows in the csr matrix will be exactly same as the length of the array.
nrows = len(train_arr)
#what we need to figure out is the number of columns (which would be the max value present in the entire dataset) and the mappings -
#for that we need to track the indices and the corresponding values in the dataset.

idx_val = []
data_val = []

for item in train_arr:
    data_index_temp = []
    data_value_temp = []
    #all values in odd loc of the dataset are indices, add them to a temp array
    for i in range(0,len(item),2):
        data_index_temp.append(item[i])
    #all values in the even loc of dataset are values, add them to another temp array
    for j in range(1,len(item),2):
        data_value_temp.append(item[j])
    idx_val.append(data_index_temp)
    data_val.append(data_value_temp)

idx = {}
tid = 0
nnz = 0

#loop through each documents indices
for item in idx_val:
    #no.of features present in a doc = number of non-zeros(nnz) = length of an item
    nnz += len(item)
    for i in item:
        if i not in idx:
            idx[i] = tid
            tid += 1
#print len(idx)

# set up memory
val = np.zeros(nnz, dtype=np.double)
ind = np.zeros(nnz, dtype=np.int)
ptr = np.zeros(nrows+1, dtype=np.int)

i = 0 # document ID / row counter
n = 0 # non-zero counter

for i,(j,k) in enumerate(zip(idx_val,data_val)):
        length = len(j)
        for l in range(length):
            ind[l+n] = j[l]
            val[l+n] = k[l]
        ptr[i+1] = ptr[i] + length
        n += length
        i += 1

#ncols = max(ind)
ncols = max(ind)+1

mat2 = []
#mat2 = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
mat2 = csr_matrix((val, ind, ptr), dtype=np.double)


# scale matrix - method from activity-data3
from collections import defaultdict

#Create a TF-IDF matrix from CSR
def csr_idf(mat, copy=False, **kargs):
    r""" Scale a CSR matrix by idf.
    Returns scaling factors as dict. If copy is True,
    returns scaled matrix and scaling factors.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # document frequency
    df = defaultdict(int)
    for i in ind:
        df[i] += 1

    # print df

    # inverse document frequency
    for k, v in df.items():
        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
    # scale by idf
    for i in range(0, nnz):
        val[i] *= df[ind[i]]

    return df if copy is False else mat

mat4 = csr_idf(mat2, copy=True)
#print train_arr[12]
#print mat4[12]

from sklearn.preprocessing import normalize
mat_normalized = normalize(mat4, norm='l2')

from sklearn.decomposition import PCA, KernelPCA
# from sklearn.metrics.pairwise import linear_kernel
kpca_arr = PCA(n_components=150)
X_mat= kpca_arr.fit_transform(mat_normalized)
# xnorm = np.linalg.norm(X_mat, axis=1)
# X_mat = X_mat/xnorm.reshape(X_mat.shape[0], 1)
# # compute cosine dist
# x_mat_dist = 1. - linear_kernel(X_mat)
print X_mat.shape[1]

v_mat = X_mat.dot(X_mat.T)
v_mat_out = 1 - v_mat

x_mat_dist = v_mat_out

#DBSCAN implementation
def dbScan(D, eps, T):
    min_pts = T
    core_points = []
    border_points = []
    noise_points = []
    scanned_points = []
    all_neighbors_mapping = []
    clusters = []

    core_point_mat = []

    for p in range(D.shape[0]):
        all_neighbors = []
        if p not in scanned_points:
            scanned_points.append(p)
            all_neighbors = findAllNeighborsWithinEps(p, eps)
            if len(all_neighbors) == 0:
                # a noise point
                noise_points.append(p)
            elif len(all_neighbors) > 0 and len(all_neighbors) < min_pts:
                # a border point
                border_points.append(p)
            else:
                # a core point
                core_points.append(p)
                all_neighbors_mapping.append(all_neighbors)

    print "noise: " + str(len(noise_points))
    print "border: " + str(len(border_points))
    print "core: " + str(len(core_points))

    #     # find border points thar are not clustered yet - using bitwise exclusive or operator
    #     non_common_points = [j for i in all_neighbors_mapping for j in i]
    #     unclustered_borders = set(non_common_points) ^ set(border_points) ^ set(core_points)
    #     # unclustered_borders = set(non_common_points) ^ set(border_points)

    #     for b_point in unclustered_borders:
    #         core_of_brd = findCoreOfBorder(b_point, eps, core_points)
    #         # print "core_of_brd: " + str(core_of_brd)
    #         for i_map in range(len(all_neighbors_mapping)):
    #             if core_of_brd in all_neighbors_mapping[i_map]:
    #                 all_neighbors_mapping[i_map].append(b_point)

    #     non_common_points1 = [j for i in all_neighbors_mapping for j in i]
    #     unclustered_borders1 = set(non_common_points1) ^ set(border_points) ^ set(core_points)

    core_points_temp = []
    core_points_temp = list(core_points)

    clusters = []

    for c in core_points:
        core_neighbors = []
        flag = 0
        core_neighbors = findConnectedCorePoints(c, eps, core_points_temp)
        # print core_neighbors
        clusters.append(core_neighbors)
        core_points_temp.remove(c)  # reduces the computation by .5

    clusters_new = []

    clusters_new = list(connected_graph(clusters))
    clusters_new.append(noise_points)


    print "\nno.of initial clusters (core+noise): "
    print len(clusters_new)
    return clusters_new, border_points

 #find intersection of the points, and merge clusters
def mergeClusters(clusters_new):
    cluster_odd=[]
    cluster_merged=[]
    for a in range(0,len(clusters_new),2):
        if (a == len(clusters_new)-1) and (len(clusters_new) % 2 != 0): #check length is odd
            cluster_odd.append(clusters_new[a])
            print "here"
        else:
            inters = set(clusters_new[a]) & set(clusters_new[a+1])
            if len(inters) > 0:
                cluster_merged.append(list((set(clusters_new[a]).union(set(clusters_new[a+1])))))
    return cluster_merged,cluster_odd


from collections import defaultdict

#find points within epsilon
def findAllNeighborsWithinEps(p,eps):
    #neighbors = set()
    neighbors = []
    i = 0
    #print p,i
    for i in range(len(x_mat_dist)):
         if p != i: #self similarity
            if x_mat_dist[p][i] <= eps:
                #neighbors.add((p,i))
                neighbors.append(i)

    return neighbors


clusters_tmp = []


#find connected core points
def findConnectedCorePoints(c_point, eps, core_pts_temp):
    core_neighbors1 = []
    core_neighbors1.append(c_point)
    for i in range(0, len(core_pts_temp)):
        if (c_point != core_pts_temp[i]) and (x_mat_dist[c_point][core_pts_temp[i]] <= eps):
            core_neighbors1.append(core_pts_temp[i])
    return core_neighbors1


#find border points
def findCoreOfBorder(b_point, eps, core_points):
    dist = []
    sorted_dist = []
    for i in range(0, len(core_points)):
        tmp = []
        if x_mat_dist[b_point][core_points[i]] != 0:
            tmp.append(core_points[i])
            tmp.append(x_mat_dist[b_point][core_points[i]])
            dist.append(tmp)
    sorted_dist = sorted(dist, key=lambda x: int(x[1]))
    if sorted_dist[0][1] > eps:
        return sorted_dist[0][0]

#build a connected graph of core points
def connected_graph(clust):
    traversed = set()
    comps = defaultdict(set)
    for clt in clust:
        for c in clt:
            comps[c].update(clt)

    def component(point):
        check = traversed.add
        points = set([point])
        next_point = points.pop
        while points:
            point = next_point()
            check(point)
            points.update(points | (comps[point] - traversed))
            # do not return - start from the point after yield next
            yield point

    for point in comps:
        if point not in traversed:
            # do not return - start from the point after yield next
            yield sorted(component(point))

clust, border_points = dbScan(X_mat, eps=0.82, T=230)

#verify the count --for testing
sum_c=0
for l in clust:
    print len(l)
    sum_c += len(l)
print "\n"
print str(sum_c)
simv = []
def findClusterIdOfB(b):
    sim_vals = []
    for j in range(len(clust)):
        s_val = []
        for i in range(len(clust[j])):
            if x_mat_dist[b][clust[j][i]] != 0:
                s_val.append(x_mat_dist[b][clust[j][i]])
        sim_vals.append(min(s_val))
    return sim_vals, sim_vals.index(min(sim_vals))


for b in border_points:
    sim_vals, c = findClusterIdOfB(b)
    # print c, sim_vals
    simv = list(sim_vals)
    clust[c].append(b)

#Save the ouput
file_name = open("/Users/sneha/desktop/output.txt", 'w');
count = 0
cluster_ids = [0] * 8580
for cluster in clust:
    count = count + 1
    for clt in cluster:
        cluster_ids[clt] = count

for id in cluster_ids:
    file_name.write(str(id))
    file_name.write('\n')
file_name.close()

