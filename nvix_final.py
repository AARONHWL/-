#!/usr/bin/env python
# coding: utf-8

# In[296]:


from sklearn import svm
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd


# In[517]:


data = pd.read_csv('/Users/aaronhuang/Desktop/ngram_table.csv',index_col=0).iloc[:,0:].T.astype(float)


# In[518]:


data


# In[519]:


vix = pd.read_csv('/Users/aaronhuang/Desktop/VIX month data.csv',index_col=0).T


# In[520]:


vix


# In[521]:


data = data[::-1]
vix = vix[::-1]


# In[522]:


nonzero_data = pd.DataFrame((data != 0).sum(0))
nonzero_data.T
data.loc["nonzero_data"]=list(nonzero_data.values.reshape(-1))


# In[523]:


droplist=list(data.loc["nonzero_data"][data.loc["nonzero_data"]<=4].index)
data=data.drop(droplist,axis=1)


# In[524]:


droplist2=list(data.loc["nonzero_data"][data.loc["nonzero_data"]>=150].index)
data=data.drop(droplist2,axis=1)


# In[525]:


data=data.drop('nonzero_data',axis=0)


# In[526]:


data = pd.concat([vix,data],axis=1)


# In[527]:


x_train_withVIX = data.iloc[:120,:]


# In[528]:


corr=x_train_withVIX.corr()['VIX']
corr.T


# In[529]:


data.loc["corr"]=list(corr.values.reshape(-1))


# In[530]:


droplist_1=list(data.loc["corr"][data.loc["corr"]<=0].index)
data=data.drop(droplist_1,axis=1)


# In[532]:


data=data.dropna(axis=1)


# In[533]:


data=data.drop('corr',axis=0)


# In[534]:


x_test = data.iloc[120:156,1:].values
x_train = data.iloc[:120,1:].values


# In[535]:


date = data.index.values
date_test = date[120:156]
date_train = date[:120]


# In[536]:


y_test = vix.iloc[120:156,:1].values
y_train = vix.iloc[:120,:1].values


# In[537]:


vix_train = svm.SVR(C=1e3, cache_size=2000, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
vix_train.fit(x_train, y_train)


# In[538]:


vix_test = vix_train.predict(x_test)


# In[539]:


vix_real = vix_train.predict(x_train)


# In[540]:


plt.plot(date_test,y_test,color = 'r', label="y_test")
plt.plot(date_test,vix_test,color = 'g', label="vix_predict")
plt.title("Predict", x=0.5, y=1.03)
plt.legend(loc = "best", fontsize=10)
plt.show()


# In[541]:


vix_train.score(x_train,y_train)


# In[542]:


#图片像素
plt.rcParams['savefig.dpi'] = 300 
#分辨率
plt.rcParams['figure.dpi'] = 200 


# In[543]:


plt.plot(date_train,y_train,color = 'r', label="y_train")
plt.plot(date_train,vix_real,color = 'g', label="vix_train")
plt.plot(date_test,y_test,color = 'r', label="y_test")
plt.plot(date_test,vix_test,color = 'g', label="vix_predict")
plt.title("Predict", x=0.5, y=1)
plt.legend(loc = "best", fontsize=10)
plt.show()


# In[554]:


vix_test1=vix_test.reshape(1,-1)
vix_test1=pd.DataFrame(vix_test1)
y_test1=pd.DataFrame(y_test)
vix_test1=vix_test1.T
result = pd.concat([vix_test1,y_test1],axis=1)
print(result.corr())
vix_train.score(x_test,y_test)

