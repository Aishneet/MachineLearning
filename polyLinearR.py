# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 19:11:00 2018

@author: Aishneet
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data=pd.read_csv("Salaries.csv")
X=data.iloc[:,1:2].values
Y=data.iloc[:,2:].values

regressor = LinearRegression()
regressor.fit(X,Y)
Y_predict=regressor.predict(X)

plt.scatter(X,Y,color="red")
plt.plot(X,regressor.predict(X),color = "blue")
plt.title("Salary vs level")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

#polynomial becuse we didnt get best fit line
from sklearn.preprocessing import PolynomialFeatures
pf= PolynomialFeatures(degree=2)
newx=pf.fit_transform(X)

new_regressor = LinearRegression()
new_regressor.fit(newx,Y)
newy=new_regressor.predict(newx)

plt.scatter(X,Y,color="red")
plt.plot(X,new_regressor.predict(newx),color = "blue")
plt.title("Salary vs level")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

#degree increases and graph get improve
poly= PolynomialFeatures(degree=4)
new1x=poly.fit_transform(X)

new1_regressor = LinearRegression()
new1_regressor.fit(new1x,Y)
new1y=new1_regressor.predict(new1x)

plt.scatter(X,Y,color="red")
plt.plot(X,new1_regressor.predict(new1x),color = "blue")
plt.title("Salary vs level")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

#converting discrete to continuous to get smooth graph
new2x=np.arange(min(X),max(X),0.1)
#converting list into matrix
new2x=new2x.reshape(len(new2x),1)

plt.scatter(X,Y,color="red")
plt.plot(new2x,new1_regressor.predict(poly.transform(new2x)),color = "blue")
plt.title("Salary vs level")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

