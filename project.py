#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pickle
import seaborn as sns


# In[60]:


crop = pd.read_csv("Crop_recommendation.csv")


# In[61]:


crop.head()


# In[62]:


crop.shape


# In[63]:


crop.info()


# In[64]:


crop.isnull().sum()


# In[65]:


crop.duplicated().sum()


# In[66]:


crop.describe()


# In[67]:


# Select only numeric columns
numeric_columns = crop.select_dtypes(include=['number'])

# Calculate the correlation matrix
correlation_matrix = numeric_columns.corr()
print(correlation_matrix)


# In[68]:


crop['label_encoded'] = LabelEncoder().fit_transform(crop['label'])


# In[69]:


crop = pd.get_dummies(crop, columns=['label'], drop_first=True)


# In[70]:


# Check the resulting correlation matrix
print(correlation_matrix)


# In[71]:


numeric_columns = crop.select_dtypes(include=['number'])
sns.heatmap(numeric_columns.corr(), annot=True, cbar=True)


# In[72]:


print(crop.columns)


# In[73]:


crop = pd.read_csv("Crop_recommendation.csv")
print(crop['label'].value_counts())


# In[74]:


crop['label'].unique().size


# In[75]:


sns.distplot(crop['P'])
plt.show()


# In[76]:


sns.distplot(crop['N'])
plt.show()


# In[77]:


crop.label.unique()


# In[78]:


crop_dict={
    'rice': 1,
    'maize': 2,
    'jute': 3,
    'cotton': 4,
    'coconut': 5,
    'papaya': 6,
    'orange': 7,
    'apple': 8,
    'muskmelon': 9,
    'watermelon': 10,
    'grapes': 11,
    'mango': 12,
    'banana': 13,
    'pomegranate': 14,
    'lentil': 15,
    'blackgram': 16,
    'mungbean': 17,
    'mothbeans': 18,
    'pigeonpeas': 19,
    'kidneybeans': 20,
    'chickpea': 21,
    'coffee': 22
}

crop['label'] = crop['label'].map(crop_dict)


# In[79]:


crop.head()


# In[80]:


crop.label.unique()


# In[81]:


crop.label.value_counts()


# In[82]:


X=crop.drop('label', axis = 1)
y=crop['label']


# In[83]:


X.head()


# In[84]:


y.head()


# In[85]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


# In[86]:


X_train.shape


# In[87]:


mx = MinMaxScaler()
X_train = mx.fit_transform(X_train)
X_test = mx.transform(X_test)


# In[88]:


X_train


# In[89]:


sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test=sc.transform(X_test)


# In[90]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score


# In[91]:


models = {
    'LogisticRegression': LogisticRegression(),
    'GaussianNB':GaussianNB(),
    'SVC':SVC(),
    'KNeighborsClassifier':KNeighborsClassifier(),
    'DecisionTreeClassifier':DecisionTreeClassifier(),
    'ExtraTreeClassifier':ExtraTreeClassifier(),
    'RandomForestClassifier':RandomForestClassifier(),
    'BaggingClassifier':BaggingClassifier(),
    'GradientBoostingClassifier':GradientBoostingClassifier(),
    'AdaBoostClassifier':AdaBoostClassifier()
}


# In[92]:


for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"{name} model with accuracy: {score}")


# In[93]:


randclf = RandomForestClassifier()
randclf.fit(X_train, y_train)
y_pred = randclf.predict(X_test)
accuracy_score(y_test, y_pred)


# In[94]:


crop.columns


# In[95]:


def recommendation(N,P,K,temperature,humidity,ph,rainfall):
    features = np.array([[N,P,K,temperature,humidity,ph,rainfall]])
    mx_features = mx.fit_transform(features)
    sc_mx_features = sc.fit_transform(mx_features)
    prediction = randclf.predict(sc_mx_features).reshape(1,-1)
    return prediction[0]


# In[96]:


crop.head()


# In[97]:


N=90
P= 42
K= 43
temperature= 20.879744
humidity=82.002744
ph=6.502985
rainfall=202.935536
predict = recommendation(N,P,K,temperature,humidity,ph,rainfall)


# In[98]:


predict


# In[99]:


import pickle
pickle.dump(randclf, open('model.pkl', 'wb'))
pickle.dump(mx, open('minmaxscaler.pkl', 'wb'))
pickle.dump(sc, open('standscaler.pkl', 'wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




