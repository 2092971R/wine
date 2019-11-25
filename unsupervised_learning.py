# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 15:41:56 2018

@author: Alexandra Russell
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn import preprocessing

wine = pd.read_csv("winequality-red.csv")

#scale the values
min_max_scaler = preprocessing.MinMaxScaler()
wine_scaled = min_max_scaler.fit_transform(wine)
wine_2 = pd.DataFrame(wine_scaled)

mapping_2={0: "fixed acididty (g/dm^3)", 
                              1: "volatile acididty (g/dm^3)",
                              2: "citric acid (g/dm^3)",
                              3: "residual sugar (g/dm^3)",
                              4: "chlorides (g/dm^3)",
                              5: "free sulfur dioxide (mg/dm^3)",
                              6: "total sulfur dioxide (mg/dm^3)",
                              7: "density(g/cm^3)",
                              8: "pH",
                              9: "sulphates (g/dm^3)",
                              10: "alcohol (vol.%)",
                              11: "quality"}
wine_2 = wine_2.rename(columns=mapping_2)
wine_2 = wine_2.drop(['quality'], axis = 1)
kmeans = KMeans(n_clusters=7, random_state=0).fit(wine_2)
print("For 7 clusters, comparing them to the wine quality:")
print("Silhouette score", metrics.silhouette_score(wine_2, kmeans.labels_))
print("Completeness score", metrics.completeness_score(wine["quality"], kmeans.labels_))
print("Homogeneity score", metrics.homogeneity_score(wine["quality"], kmeans.labels_))

#testing cluster sizes
store = []
for i in range(3,10):
   kmeans = KMeans(n_clusters=i, random_state=0).fit(wine_2)
   store.append((metrics.silhouette_score(wine_2, kmeans.labels_), i)) 
   
plt.scatter([s[1]for s in store],
                 [s[0] for s in store])
    
plt.xlabel("Clusters")
plt.ylabel("Silhouette score")
plt.savefig("clusters.png")
plt.close()

#graphs showing the groupings obtained
data = wine_2.drop(["fixed acididty (g/dm^3)", "volatile acididty (g/dm^3)","citric acid (g/dm^3)", "total sulfur dioxide (mg/dm^3)","free sulfur dioxide (mg/dm^3)","density(g/cm^3)", "pH","sulphates (g/dm^3)"], axis = 1)
labels = KMeans(6, random_state=0).fit_predict(data)

wine = pd.read_csv("winequality-red.csv")
mapping={"fixed acidity": "fixed acididty (g/dm^3)", 
                              "volatile acidity": "volatile acididty (g/dm^3)",
                              "citric acid": "citric acid (g/dm^3)",
                              "residual sugar": "residual sugar (g/dm^3)",
                              "chlorides": "chlorides (g/dm^3)",
                              "free sulfur dioxide": "free sulfur dioxide (mg/dm^3)",
                              "total sulfur dioxide": "total sulfur dioxide (mg/dm^3)",
                              "density": "density(g/cm^3)",
                              "pH": "pH",
                              "sulphates": "sulphates (g/dm^3)",
                              "alcohol": "alcohol (vol.%)"}
wine = wine.rename(columns=mapping)
plt.scatter(wine["alcohol (vol.%)"], wine["residual sugar (g/dm^3)"], c=labels,s=50, cmap='viridis');
plt.xlabel("alcohol (vol.%)")
plt.ylabel("residual sugar (g/dm^3)")

plt.savefig("alh_res.png")
plt.show()

plt.scatter(wine["residual sugar (g/dm^3)"], wine["chlorides (g/dm^3)"], c=labels,s=50, cmap='viridis');
plt.xlabel("chlorides (g/dm^3)")
plt.xlabel("residual sugar (g/dm^3)")
plt.savefig("alh_chl.png")