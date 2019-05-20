# -*- coding: utf-8 -*-
"""
Created on Sun May 19 13:17:54 2019

@author: Gautam
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('./Market_Basket_Optimisation.csv',header=None)

transactions=[]
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

from apyori import apriori
rules = apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)

results = list(rules)
