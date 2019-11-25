# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 16:34:31 2018

@author: Alexandra Russell 
"""

import pandas as pd

wine = pd.read_csv("winequality-red.csv")


if (wine.isnull().values.any()):
    print("There are some null values in the dataset")
else:
    print("There are no null values in the dataset")
    
    
print("There are", wine['quality'].count(), "entries in the dataset" )

print("The variables in the dataset are", list(wine.columns.values))