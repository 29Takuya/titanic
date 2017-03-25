
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[94]:

df_test = pd.read_csv('test.csv')


# In[95]:

df_train = df_test[['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']]


# In[96]:

df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())
df_train['Fare'] = df_train['Fare'].fillna(df_train['Fare'].mean())
df_train['Cabin'] = df_train['Cabin'].fillna(-1)
df_train['Embarked'] = df_train['Embarked'].fillna(-1)


# In[97]:

for field in ['Sex','Embarked']:
    df_dummy = pd.get_dummies(df_train[field],prefix = field)
    df_train = pd.concat([df_train,df_dummy],axis  = 1)
    df_train = df_train.drop(field,axis = 1)


# In[98]:

df_train = df_train.drop('Cabin', axis = 1)


# In[99]:

cols = df_train.columns.tolist()
cols.insert(7,'Embarked_-1')
df_train_p = pd.DataFrame(df_train,columns=cols)
df_train_p['Embarked_-1'] = df_train_p['Embarked_-1'].fillna(0)


# In[109]:

df_train_p


# In[100]:

x_np = df_train_p.as_matrix().astype(np.float32)


# In[101]:

from sklearn.model_selection import train_test_split
import chainer
from chainer import training, Chain
from chainer.training import extensions
from chainer import links as L
from chainer import functions as F
from chainer.datasets import TupleDataset
from chainer import serializers


# In[102]:

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[136]:

class MyChain4(Chain):
    def __init__(self):
        super(MyChain4,self).__init__(
            l1 = L.Linear(None, 100),
            l2 = L.Linear(None, 100),
            l3 = L.Linear(None,100),
            l4 = L.Linear(None,2),
            bn1 = L.BatchNormalization(size = 100),
            bn2 = L.BatchNormalization(size = 100),
            bn3 = L.BatchNormalization(size = 100)
        )
        
    def __call__(self, x):
        h = F.dropout(x,ratio = 0.2)
        h = F.dropout(F.relu(self.bn1(self.l1(x))))
        h = F.dropout(F.relu(self.bn2(self.l2(h))))
        h = F.dropout(F.relu(self.bn3(self.l3(h))))
        y = self.l4(h)
        
        return y


# In[131]:

class MyChain(Chain):
    def __init__(self):
        super(MyChain,self).__init__(
            l1 = L.Linear(None, 1000),
            l2 = L.Linear(None,100),
            l3 = L.Linear(None,2),
            bn1 = L.BatchNormalization(size = 1000)
        )
        
    def __call__(self, x):
        h = F.relu(self.bn1(self.l1(x)))
        h = F.dropout(F.relu(self.l2(h)))
        y = self.l3(h)
        
        return y


# In[142]:

class MyChain5(Chain):
    def __init__(self):
        super(MyChain5,self).__init__(
            l1 = L.Linear(None, 100),
            l2 = L.Linear(None, 100),
            l3 = L.Linear(None,100),
            l4 = L.Linear(None,2),
            bn1 = L.BatchNormalization(size = 100),
            bn2 = L.BatchNormalization(size = 100),
            bn3 = L.BatchNormalization(size = 100)
        )
        
    def __call__(self, x):
        h = F.dropout(x,ratio = 0.2)
        h = F.dropout(F.relu(self.bn1(self.l1(x))))
        h = F.dropout(F.relu(self.bn2(self.l2(h))))
        h = F.dropout(F.relu(self.bn3(self.l3(h))))
        y = self.l4(h)
        
        return y


# In[143]:

model = L.Classifier(MyChain4())
serializers.load_npz('model/model_5.npz',model)


# In[144]:

pred = np.argmax(F.softmax(model.predictor(x_np)).data,axis = 1)


# In[145]:

df_out =pd.DataFrame(pred, columns=['Survived'])


# In[146]:

df_out = pd.concat([df_test[['PassengerId']],df_out],axis=1)


# In[147]:

df_out.to_csv('submit/model_5.csv',index=False)


# In[ ]:



