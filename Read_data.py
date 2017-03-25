
# coding: utf-8

# In[217]:

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[218]:

df_dev = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_gender = pd.read_csv('gender_submission.csv')


# In[219]:

df_dev.describe()


# In[220]:

df_train = df_dev[['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']]


# In[221]:

df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())
df_train['Cabin'] = df_train['Cabin'].fillna(-1)
df_train['Embarked'] = df_train['Embarked'].fillna(-1)


# In[222]:

for field in ['Sex','Embarked']:
    df_dummy = pd.get_dummies(df_train[field],prefix = field)
    df_train = pd.concat([df_train,df_dummy],axis  = 1)
    df_train = df_train.drop(field,axis = 1)


# In[223]:

df_nan = df_train.isnull()
df_nan.describe()


# In[224]:

df_train = df_train.drop('Cabin', axis = 1)


# In[225]:

df_train


# In[226]:

df_target = df_dev[['Survived']]


# In[227]:

df_target


# In[228]:

x_np = df_train.as_matrix().astype(np.float32)


# In[249]:

df_train.describe()


# In[229]:

y_np = df_target.as_matrix().astype(np.int32)


# In[230]:

from sklearn.model_selection import train_test_split
import chainer
from chainer import training, Chain
from chainer.training import extensions
from chainer import links as L
from chainer import functions as F
from chainer.datasets import TupleDataset
from chainer import serializers


# In[192]:

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[396]:

X_train,X_test,y_train,y_test = train_test_split(x_np,y_np,test_size=0.2)


# In[397]:

train = TupleDataset(X_train,y_train.reshape(1,-1)[0])
test = TupleDataset(X_test,y_test.reshape(1,-1)[0])


# In[398]:

train_iter = chainer.iterators.SerialIterator(train, 100)
test_iter = chainer.iterators.SerialIterator(test, 100,repeat=False, shuffle=False)


# In[399]:

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


# In[400]:

model = L.Classifier(MyChain4())
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)


# In[406]:

updater = training.StandardUpdater(train_iter, optimizer, device=-1)
trainer = training.Trainer(updater, (10000, 'epoch'), out="result")
trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
#trainer.extend(extensions.ProgressBar())


# In[407]:

trainer.run()


# In[410]:

serializers.save_npz('model/model_5.npz',model)


# In[408]:

pred = np.argmax(F.softmax(model.predictor(X_test)).data,axis = 1)


# In[409]:

print classification_report(y_test,pred)


# In[193]:

classifier = SVC()


# In[194]:

classifier.fit(X_train,y_train)


# In[195]:

pd = classifier.predict(X_test)


# In[196]:

confusion_matrix(y_test,pd)


# In[197]:

print classification_report(y_test,pd)


# In[198]:

print accuracy_score(y_test,pd)


# In[ ]:



