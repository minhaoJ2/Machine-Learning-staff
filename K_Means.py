import numpy as np
import random
import matplotlib.pyplot as plt

file = open("./breast-cancer-wisconsin.data", "r")
total = file.read().splitlines()
dataset = []
for i in range(len(total)):
    invalid = 0
    temp = total[i].split(",")
    t = []
    for j in range(1, len(temp) - 1):
        if temp[j] == '?':
            invalid = 1
            break
        t.append(int(temp[j]))
    if not invalid:
        dataset.append(t)
        
# find the closest centroid
def choose_cluster(clusters, data_point):
    m = 1000000
    c = -1
    for i in range(len(clusters)):
        distance = 0
        for j in range(len(data_point)):
            distance += (data_point[j] - clusters[i][j])**2
        if distance < m:
            m = distance
            c = i
    return c

# find the new centroid in each cluster
def cluster_means(clusterID, c):
    temp = []
    for key in clusterID:
        if clusterID[key] == c:
            temp.append(dataset[key])
    if len(temp) == 0:
        for _ in range(len(dataset[0])):
            temp.append(0)
        return temp
    else:
        return list(np.average(temp, axis=0))

def reassign(clusterID, clusters):
    for i in range(len(clusters)):
        t = cluster_means(clusterID, i)
        clusters[i] = t
    return clusters

# calculate the 2-norm between two lists
def distance(x, y):
    distance = 0
    for i in range(len(x)):
        distance += (x[i] - y[i])**2
    return distance

# final potential function
def potential(clusters, clusterID):
    potential = 0
    for i in range(len(dataset)):
        potential += distance(dataset[i], clusters[clusterID[i]])
    return potential
max_iter = 300

# k-means main loop
def fit(k):
    clusters = random.sample(dataset, k)
    clusterID = {}
    for _ in range(max_iter):
        for i in range(len(dataset)):
            clusterID[i] = choose_cluster(clusters, dataset[i])
        temp = clusters.copy()
        clusters = reassign(clusterID, clusters)
        # converges
        if (clusters == temp):
            break
    return potential(clusters, clusterID)

if __name__ == "__main__":
    k_values = [2, 3, 4, 5, 6, 7, 8]
    result = []
    for k in k_values:
        result.append(fit(k))
    plt.xlabel("k values")
    plt.ylabel("Potential Value")
    plt.plot(k_values, result)
