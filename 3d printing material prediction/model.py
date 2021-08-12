import pandas as  pd

import numpy as np

import matplotlib.pyplot as plt

# UPLOADING DATA

ds=pd.read_csv(r"data_set.csv")

ds.isnull().any()

from sklearn.preprocessing import LabelEncoder

lb=LabelEncoder()

ds=ds.iloc[:,:].values

ds[:,3]=lb.fit_transform(ds[:,3])

ds[:,7]=lb.fit_transform(ds[:,7])

da=pd.DataFrame(ds)

y=ds[:,7]

y=y.astype("int")

da.drop(columns=7,inplace=True)

x=da.iloc[:,:].values

x

# TRAIN TEST SPLIT

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# FEATURE SCALING

from sklearn.preprocessing import MinMaxScaler

sc=MinMaxScaler()

x_train

x_train=sc.fit_transform(x_train)

x_train

x_test=sc.transform(x_test)

x_test

"""# decision tree

# training
"""

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(criterion='entropy')

dt.fit(x_train,y_train)

"""# predicting"""

y_pred_dt=dt.predict(x_test)

"""# accuracy"""

import sklearn.metrics as metrics

fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred_dt)

roc_auc_DT=metrics.auc(fpr,tpr)

roc_auc_DT

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred_dt)

plt.plot(fpr,tpr,label='AUC = %0.2f' % roc_auc_DT)
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title("roc_curve")
plt.legend()



