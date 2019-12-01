from sklearn import svm
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd

#讀入檔案
vix = pd.read_csv('/Users/aaronhuang/Desktop/VIX.csv')

#切割 Vix_data 為 y_train, y_test
y_test = vix.iloc[-100:, 4:5].values
y_train = vix.iloc[:-100, 4:5].values

#沒第三階段的data，自己生數字
X = np.sort(5 * np.random.rand(3501, 1), axis=0)

#切割 X_data 為 x_train, x_test
x_train = X[:-100]
x_test = X[-100:]

#開SVR
vix_train = svm.SVR()
vix_train.fit(x_train, y_train)
SVR(C=1e3, cache_size=2000, coef0=0.0, degree=3, epsilon=0.1, gamma=0.01,
    kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

#畫出train的圖
plt.subplot(2,2,1)
plt.scatter(x_train, y_train, c='r', label='data')
plt.plot(x_train,y_train, c='k' , label='linear model')
plt.xlabel('train_x_data')
plt.ylabel('VIX_train')
plt.title('VIX_train')
plt.legend()
plt.show()

#畫出test的圖
plt.subplot(2,2,2)
vix_test = vix_train.predict(x_test)
plt.scatter(vix_test,y_test, c='g', label='test_data')
plt.plot(vix_test,y_test, c='k' , label='linear model')
plt.xlabel('test_x_data')
plt.ylabel('test_y_data')
plt.title('VIX_test')
plt.legend()
plt.show()

#畫出預測圖
plt.subplot(2,2,3)
plt.plot(y_train)
plt.plot(vix_train.predict(x_train))





