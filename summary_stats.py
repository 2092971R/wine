#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 13:16:02 2018

@author: Alexandra Russell
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

#correlation heatmap
corr = wine.corr()
ax = sns.heatmap(corr)
ax.figure.tight_layout()
plt.savefig("heatmap.png")
plt.show()

#add in the units of attributes
wine = wine.rename(columns=mapping)

#max, min and mean values of the attributes
(wine.drop(['quality'], axis = 1)).describe().to_csv("describe.csv")

#box plots for all numberic variables
(wine.drop(['quality'], axis = 1)).plot(kind='box', subplots=True, layout=(7,3), figsize=(10,25))
plt.savefig("boxplots.png")
plt.show()

wine["residual sugar (g/dm^3)"].plot.density()
plt.xlabel('residual sugar (g/dm^3)')
plt.savefig("density.png")
plt.show()

#normalise and produce the interquartile ranges
from sklearn import preprocessing
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
                              10: "alcohol (vol.%)"}
wine_2 = wine_2.rename(columns=mapping_2)
(wine_2.drop([11], axis = 1)).describe().to_csv("normalised.csv")

#residual sugars
print("Median for residual sugar values", np.median(wine["residual sugar (g/dm^3)"]))

#bar chart of the wine qualities
wine.groupby("quality").size().plot(kind = 'bar')
plt.xlabel("Wine Quality")
plt.ylabel("Number of wines")
plt.savefig("barchart.png")
plt.show()