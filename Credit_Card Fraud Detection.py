#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv("C:/Users/HP/Desktop/Credit card fraud detection/creditcard.csv")


# In[3]:


data.head()


# In[4]:


pd.options.display.max_columns = None


# In[5]:


data.head()


# In[6]:


data.tail()


# In[7]:


data.shape


# In[8]:


print("Number of columns: {}".format(data.shape[1]))
print("Number of rows: {}".format(data.shape[0]))


# In[9]:


data.info()


# In[10]:


data.isnull().sum()


# In[11]:


from sklearn.preprocessing import StandardScaler


# In[12]:


data.head()


# In[13]:


data = data.drop(['Time'], axis =1)


# In[14]:


data.head()


# In[15]:


data.duplicated().any()


# In[16]:


data = data.drop_duplicates()


# In[17]:


data.shape


# In[18]:


data['Class'].value_counts()


# In[19]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# In[20]:


sns.countplot(data['Class'])
plt.show()


# In[21]:


X = data.drop('Class', axis = 1)
y=data['Class']


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[24]:


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# In[25]:


classifier = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree Classifier": DecisionTreeClassifier()
}

for name, clf in classifier.items():
    print(f"\n=========={name}===========")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"\n Accuaracy: {accuracy_score(y_test, y_pred)}")
    print(f"\n Precision: {precision_score(y_test, y_pred)}")
    print(f"\n Recall: {recall_score(y_test, y_pred)}")
    print(f"\n F1 Score: {f1_score(y_test, y_pred)}")


# # Undersampling

# In[26]:


normal = data[data['Class']==0]
fraud = data[data['Class']==1]


# In[27]:


normal.shape


# In[28]:


fraud.shape


# In[29]:


normal_sample = normal.sample(n=473)


# In[30]:


normal_sample.shape


# In[31]:


new_data = pd.concat([normal_sample,fraud], ignore_index=True)


# In[32]:


new_data.head()


# In[33]:


new_data['Class'].value_counts()


# In[34]:


X = new_data.drop('Class', axis = 1)
y= new_data['Class']


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[36]:


classifier = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree Classifier": DecisionTreeClassifier()
}

for name, clf in classifier.items():
    print(f"\n=========={name}===========")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"\n Accuaracy: {accuracy_score(y_test, y_pred)}")
    print(f"\n Precision: {precision_score(y_test, y_pred)}")
    print(f"\n Recall: {recall_score(y_test, y_pred)}")
    print(f"\n F1 Score: {f1_score(y_test, y_pred)}")


# # OVERSAMPLING

# In[37]:


X = data.drop('Class', axis = 1)
y= data['Class']


# In[38]:


X.shape


# In[39]:


y.shape


# In[40]:


pip install imblearn


# In[43]:


from imblearn.over_sampling import SMOTE


# In[44]:


X_res, y_res = SMOTE().fit_resample(X,y)


# In[45]:


X_res, y_res = SMOTE().fit_resample(X,y)


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.2, random_state = 42)


# In[47]:


classifier = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree Classifier": DecisionTreeClassifier()
}

for name, clf in classifier.items():
    print(f"\n=========={name}===========")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"\n Accuaracy: {accuracy_score(y_test, y_pred)}")
    print(f"\n Precision: {precision_score(y_test, y_pred)}")
    print(f"\n Recall: {recall_score(y_test, y_pred)}")
    print(f"\n F1 Score: {f1_score(y_test, y_pred)}")


# In[48]:


dtc = DecisionTreeClassifier()
dtc.fit(X_res, y_res)


# In[49]:


import joblib


# In[50]:


joblib.dump(dtc, "credit_card_model.pkl")


# In[51]:


model = joblib.load("credit_card_model.pkl")


# In[52]:


pred = model.predict([[-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215,149.62]])


# In[53]:


pred[0]


# In[54]:


if pred[0] == 0:
    print("Normal Transcation")
else:
    print("Fraud Transcation")


# In[74]:


pred = model.predict([[1.191857111,0.266150712,0.166480113,0.448154078,0.060017649,-0.082360809,-0.078802983,0.085101655,-0.255425128,-0.166974414,1.612726661,1.065235311,0.489095016,-0.143772296,0.635558093,0.463917041,-0.114804663,-0.18336127,-0.145783041,-0.069083135,-0.225775248,-0.638671953,0.101288021,-0.339846476,0.167170404,0.125894532,-0.008983099,0.014724169,2.69
]])


# In[84]:


pred[0]


# In[85]:


if pred[0] == 7:
    print("Normal Transcation")
else:
    print("Fraud Transcation")


# In[86]:


if pred[0] == 0:
    print("Normal Transcation")
else:
    print("Fraud Transcation")

