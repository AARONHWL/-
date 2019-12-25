#!/usr/bin/env python
# coding: utf-8

# In[296]:


from sklearn import svm
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd


# In[641]:


data = pd.read_csv('/Users/aaronhuang/Desktop/ngram_table.csv',index_col=0).iloc[:,0:].T.astype(float)


# In[642]:


data


# In[643]:


vix = pd.read_csv('/Users/aaronhuang/Desktop/VIX month data.csv',index_col=0).T


# In[644]:


vix


# In[645]:


data = data[::-1]
vix = vix[::-1]


# In[646]:


nonzero_data = pd.DataFrame((data != 0).sum(0))
nonzero_data.T
data.loc["nonzero_data"]=list(nonzero_data.values.reshape(-1))


# In[647]:


droplist=list(data.loc["nonzero_data"][data.loc["nonzero_data"]<=4].index)
data=data.drop(droplist,axis=1)


# In[648]:


droplist2=list(data.loc["nonzero_data"][data.loc["nonzero_data"]>=150].index)
data=data.drop(droplist2,axis=1)


# In[649]:


data=data.drop('nonzero_data',axis=0)


# In[650]:


data = pd.concat([vix,data],axis=1)


# In[651]:


x_train_withVIX = data.iloc[:120,:]


# In[652]:


corr=x_train_withVIX.corr()['VIX']
corr.T


# In[653]:


data.loc["corr"]=list(corr.values.reshape(-1))


# In[654]:


droplist_1=list(data.loc["corr"][data.loc["corr"]<=0].index)
data=data.drop(droplist_1,axis=1)


# In[655]:


data=data.dropna(axis=1)


# In[656]:


data=data.drop('corr',axis=0)


# In[657]:


x_test = data.iloc[120:156,1:].values
x_train = data.iloc[:120,1:].values


# In[658]:


X=data.iloc[:,1:].values


# In[659]:


date = data.index.values
date_test = date[120:156]
date_train = date[:120]


# In[660]:


y_test = vix.iloc[120:156,:1].values
y_train = vix.iloc[:120,:1].values


# In[661]:


y=vix.iloc[:,:1].values


# In[662]:


vix_train = svm.SVR(C=1e3, cache_size=2000, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
vix_train.fit(x_train, y_train)


# In[663]:


vix_test = vix_train.predict(x_test)


# In[664]:


vix_real = vix_train.predict(x_train)


# In[665]:


plt.plot(date_test,y_test,color = 'r', label="y_test")
plt.plot(date_test,vix_test,color = 'g', label="vix_predict")
plt.title("Predict", x=0.5, y=1.03)
plt.legend(loc = "best", fontsize=10)
plt.show()


# In[666]:


vix_train.score(x_train,y_train)


# In[667]:


#图片像素
plt.rcParams['savefig.dpi'] = 300 
#分辨率
plt.rcParams['figure.dpi'] = 200 


# In[707]:



plt.plot(date_train,y_train,color = 'r',label="VIX")
plt.plot(date_train,vix_real,color = 'b', label="nvix_train")
plt.plot(date_test,y_test,color = 'r',ls='None',marker='.',label="VIX")
plt.plot(date_test,vix_test,color = 'b',ls='None',marker='.', label="NVIX")
plt.xticks(fontsize=5)
plt.title("Predict", x=0.5, y=1)
plt.legend(loc = "best", fontsize=10)
plt.show()


# In[577]:


vix_test1=vix_test.reshape(1,-1)
vix_test1=pd.DataFrame(vix_test1)
y_test1=pd.DataFrame(y_test)
vix_test1=vix_test1.T
result = pd.concat([vix_test1,y_test1],axis=1)
print(result.corr())
vix_train.score(x_test,y_test)


# In[691]:


vix_train.coef_






