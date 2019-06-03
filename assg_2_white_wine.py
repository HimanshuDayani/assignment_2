#!/usr/bin/env python
# coding: utf-8

# In[73]:


import pandas as pd
import numpy as np
w_wine=pd.read_csv("/home/himanshu/Downloads/winequality-white.csv",delimiter=';')
w_wine[:10]


# In[74]:


m=w_wine.shape[0]
n=w_wine.shape[1]
x=np.ones((m,1))
x.shape
w_wine = (w_wine - w_wine.mean())/w_wine.std()
w_wine[:10]


# In[75]:


y=w_wine[['quality']].values
X=w_wine.drop(labels='quality',axis=1)
X=np.concatenate((x,X),1)
X


# In[76]:


theta=np.random.randn(n,1)
h=X.dot(theta)
theta.shape


# In[77]:


iteration=1000
alpha=0.01        #learning rate is like step size. It indicates how fast are our parameters changing. If the
#learning is too small, gradient descent can be slow and if the learning rate is too large, gradient descent can
#overshoot the minimum. It may fail to converge also.
cost=np.zeros(iteration)
#Derivative denotes the slope at a particular point. So, here derivative of cost function w.r.t. to theta parameter
#is done which explains how fast is our cost function changing with change in parameters.
for i in range(iteration) :
    theta=theta-(alpha/m)*(np.dot(X.T,h-y))
    h=X.dot(theta)
    cost[i]=(1/(2*m))*(np.sum(np.square(h-y)))
    
cost[:20]
#Cost function is a mathematical formula which helps to determine how well is our algorithm performing on the data
#provided to it. Our main aim is to minimise the cost function and determine the parameters.Basically it determines
#the difference between our predicted value and the the actual output.


# In[78]:


import matplotlib.pyplot as plt
fig,ax=plt.subplots()
ax.plot(np.arange(iteration),cost,'r')
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')


# In[70]:


cost[cost.shape[0]-1]


# In[ ]:




