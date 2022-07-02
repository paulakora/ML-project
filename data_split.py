#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# In[2]:


X = joblib.load('train_data_2.pkl')


# In[3]:


y = pd.read_csv('/Users/paulakoralewska/Downloads/train_labels.csv')


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify=y)


# In[5]:


y_train.value_counts()


# In[6]:


y_test.value_counts()


# In[7]:


print(X.shape)
print(y.shape)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[8]:


scaler = MinMaxScaler(clip=True, feature_range=(-1.0 , 1.0))
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[9]:


dummy_clf = DummyClassifier(strategy="most_frequent")

dummy_clf.fit(X_train, y_train)

y_pred_dummy = dummy_clf.predict(X_test)

dummy_clf.score(X_test, y_test)


# In[10]:


cf_matrix_dummy = confusion_matrix(y_test, y_pred_dummy)
print(cf_matrix_dummy)


# In[13]:


labels = ['TN','FP','FN','TP']
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix_dummy, annot=labels, fmt='', cmap='Blues')


# In[14]:


def judge_model(model, name, plot=False):
    print(name)
    print('-'*20)
    
    print('Training Performance')
    print('-> Acc:', accuracy_score(y_train, model.predict(X_train)) )
    print('-> AUC:', roc_auc_score(y_train, model.predict_proba(X_train)[:, 1] ))
    
    print('Testing Performance')
    print('-> Acc:', accuracy_score(y_train, model.predict(X_train)) )
    print('-> AUC:', roc_auc_score(y_test, model.predict_proba(X_test)[:, 1] ))
    print()
    
    if plot:
        fpr, tpr, thres = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        plt.figure(figsize=(4, 4))
        plt.plot(fpr, tpr, label='Test')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.show()


# In[15]:


judge_model(dummy_clf, 'Dummy Classifier', plot=True)


# In[16]:


dt_clf = DecisionTreeClassifier(random_state=0)

dt_clf.fit(X_train, y_train)

y_pred_dt = dt_clf.predict(X_test)

dt_clf.score(X_test, y_test)


# In[41]:


cross_val_score(dt_clf, X_test, y_test, cv=10)


# In[58]:


cf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
print(cf_matrix_dt)


# In[59]:


labels = ['TN','FP','FN','TP']
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix_dt, annot=labels, fmt='', cmap='Blues')


# In[60]:


judge_model(dt_clf, 'Decision Tree', plot=True)


# In[19]:


model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    }
}


# In[21]:


scores = []

for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(X, y)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })


# In[22]:


df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df


# In[ ]:




