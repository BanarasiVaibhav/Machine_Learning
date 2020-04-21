# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 01:00:42 2020

@author: Vaibhav
"""

# Machine learning Template
import numpy  as np
import matplotlib.pyplot as plt
import pandas as pd


# importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values


# Splitting dataset to training set and test set
"""from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)"""


# feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""


# Fitting linear regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)


# Fitting polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=3)
x_poly = poly_reg.fit_transform(x)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

#Visualise the linear regression results
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title("Truth or Bluff (linear regression)")
plt.xlabel('Salary')
plt.ylabel("Years")
plt.show()

#Visualising polynomial regressor model
x_grid= np.arange(min(x),max(x),0.1)
x_grid= x_grid.reshape((len(x_grid)),1)

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color='blue')
plt.title("Truth or Bluff (linear regression)")
plt.xlabel("salary")
plt.ylabel("Years")
plt.show()

