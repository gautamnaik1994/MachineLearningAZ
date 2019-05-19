# -*- coding: utf-8 -*-
"""
Created on Tue May 14 22:47:21 2019

@author: Gautam
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

from sklearn.cluster import KMeans
wccs = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=
                    10,random_state=0)
    kmeans.fit(X)
    wccs.append(kmeans.inertia_)
plt.plot(range(1,11),wccs)
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCCS")
plt.show()

kmeans = KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=
            10,random_state=0)
y_kmeans=kmeans.fit_predict(X)

plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100, c= 'red', label= 'Carefull')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100, c= 'blue', label=
            'Standard')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100, c= 'green', label= 'Target')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100, c= 'cyan', label=
            'Careless')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100, c= 'magenta', label=
            'Sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300, c=
            'yellow', label= 'Centroids')
plt.title("Clusters")
plt.xlabel('Annual Income')
plt.ylabel('Spending Score (1,100)')
plt.legend()
plt.show()

