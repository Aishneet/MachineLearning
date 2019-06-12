# -*- coding: utf-8 -*-

#import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("Salary_Data.csv")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size = 1/3,random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train,Y_train)
Y_predict=regressor.predict(X_test)






"""plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color = "blue")
plt.title("Salary vs Years of experince(Training set)")
plt.xlabel("Years")
plt.ylabel("Salary")
plt.show()"""



plt.scatter(X_test,Y_test,color="red")
plt.plot(X_train,regressor.predict(X_train),color = "blue")
plt.title("Salary vs Years of experince(Training set)")
plt.xlabel("Years")
plt.ylabel("Salary")
plt.show()
