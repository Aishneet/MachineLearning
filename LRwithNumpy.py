# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 19:09:12 2018

@author: Aishneet
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
cost=[]
steps=[]


dataset=pd.read_csv("Salary_Data.csv")
X=dataset.iloc[:,-2].values
Y=dataset.iloc[:,-1].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size = 1/3,random_state = 0)

theta0=0.5
theta1=0.5
alpha=0.001

n=len(X_train)
for i in range(10000):
    y_predict=theta0+theta1*X_train
    cost.append(np.sum((y_predict-Y_train)**2)/2*n)
    steps.append(i)
    loss=np.sum(y_predict-Y_train)/n
    loss1=np.sum(np.multiply(y_predict-Y_train,X_train))/n
    theta0=theta0-alpha*loss
    theta1=theta1-alpha*loss1
    
    
plt.scatter(cost,steps)
plt.plot(cost)
plt.show()


y_bar=np.sum(Y_train)/n
SST=np.sum((y_bar-Y_train)**2)
SSE=np.sum((y_predict-Y_train)**2)
R_squ=1-SSE/SST