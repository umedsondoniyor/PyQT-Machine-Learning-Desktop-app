# -*- coding: utf-8 -*-
"""
Created on Tue Dec 05 02:29:39 2017

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans

X = np.array([[1, 2],
              [5, 8],
              [1.5, 1.8],
              [8, 8],
              [1, 0.6],
              [9, 11],
              [15, 7],
              [7.5, 2.8],
              [7, 5],
              [10, 4.6],
              [8, 1]
              
              
              
              ])

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids , "Merkezlerin bilgisi")
print(labels ,"Her değer için kümesi bilgisi")

colors = ["g.","r.","b.","y.", " c."]

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)

print centroids

plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=70, linewidths=3, zorder=10)

plt.show()
