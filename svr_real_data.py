#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import svm
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


data = pd.read_csv('/Users/aaronhuang/Desktop/ngram_table.csv')


# In[3]:


vix = pd.read_csv('/Users/aaronhuang/Desktop/VIX month data.csv')


# In[6]:


data = data.T
vix = vix.T


# In[132]:


data = data[::-1]
vix = vix[::-1]


# In[133]:


vix


# In[134]:


data


# In[193]:


date = data.iloc[140:156,0:1].values
date_test = date[:,0]


# In[156]:


x_test = data.iloc[140:156,1:3550].values
x_train = data.iloc[1:140,1:3550].values


# In[165]:


y_test = vix.iloc[139:155,:].values
y_train = vix.iloc[0:139,:].values


# In[168]:


vix_train = svm.SVR()
vix_train.fit(x_train, y_train)
SVR(C=1e3, cache_size=2000, coef0=0.0, degree=3, epsilon=0.1, gamma=0.01,
    kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)


# In[190]:


vix_test = vix_train.predict(x_test)


# In[192]:


vix_train.score(x_train,y_train)


# In[184]:


plt.plot(date_test,y_test,color = 'r', label="y_test")
plt.plot(date_test,vix_test,color = 'g', label="vix_predict")
plt.title("Predict", x=0.5, y=1.03)
plt.legend(loc = "best", fontsize=10)
plt.show()


# In[189]:


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(date_test, y_test, color='red', label='data_test')
ax1.set_ylabel('y_test')
ax1.set_xlabel('Date')
ax1.set_ylim(10, 30)
plt.legend(loc='lower left')


ax2 = ax1.twinx()
ax2.plot(date_test, vix_test, color='blue', label='vix_predict')
ax2.set_ylabel('VIX_predict')
ax2.set_ylim(17, 19)
plt.legend(loc='upper right')
plt.show()

