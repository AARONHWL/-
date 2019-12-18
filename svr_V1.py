#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import svm
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


data = pd.read_csv('/Users/aaronhuang/Desktop/CNBC_freq_table_rowdata.csv',index_col=1).iloc[:,1:].T.astype(float)


# In[3]:


vix = pd.read_csv('/Users/aaronhuang/Desktop/VIX month data.csv',index_col=0).T


# In[4]:


data = pd.concat([vix,data],axis=1)


# In[5]:


nonzero_data = pd.DataFrame((data != 0).sum(0))
nonzero_data.T


# In[8]:


data.loc["nonzero_data"]=list(nonzero_data.values.reshape(-1))


# In[9]:


data


# In[10]:


data1 = data.sort_values(by=['nonzero_data'],axis=1,ascending=False)


# In[12]:


droplist=list(data1.loc["nonzero_data"][data1.loc["nonzero_data"]<=4].index)
data1=data1.drop(droplist,axis=1)


# In[13]:


data1


# In[14]:


data2 = data1.drop('nonzero_data',axis=0)


# In[15]:


data2


# In[16]:


data2.corr()['VIX']


# In[18]:


data2.corr()['VIX'].min()


# In[19]:


data2 = data2[::-1]


# In[20]:


data2


# In[32]:


x_test = data2.iloc[140:156,1:].values
x_train = data2.iloc[:140,1:].values


# In[34]:


y_test = data2.iloc[140:156,:1].values
y_train = data2.iloc[:140,:1].values


# In[48]:


date = data2.index.values
date_test = date[140:156]


# In[54]:


vix_train = svm.SVR()
vix_train.fit(x_train, y_train)
SVR(C=1e3, cache_size=2000, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
    kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)


# In[55]:


vix_test = vix_train.predict(x_test)


# In[56]:


vix_train.score(x_train,y_train)


# In[57]:


plt.plot(date_test,y_test,color = 'r', label="y_test")
plt.plot(date_test,vix_test,color = 'g', label="vix_predict")
plt.title("Predict", x=0.5, y=1.03)
plt.legend(loc = "best", fontsize=10)
plt.show()


# In[58]:


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


# In[ ]:




