#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pandas is a library and it contains the functions related to the database
import pandas as pd


# In[2]:


#dataset is structured collection of data assigned to dataset variable
dataset=pd.read_csv("salary_data.csv")


# In[3]:


#dataset is stored variable
dataset


# In[4]:


#independent is input variable
independent=dataset[["YearsExperience"]]


# In[5]:


#here we taking only input variable 
independent


# In[6]:


#dependent is output variable
dependent=dataset[["Salary"]]


# In[7]:


#here we taking only output variable
dependent


# In[8]:


#here we splitting train and test to create model
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(independent, dependent, test_size=0.30,random_state=0)


# In[9]:


#here regressor stores the data.then fit is caklled by access operator
#here output linear regression shows weight and bais
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)


# In[10]:


#a regressor is independent variable.it will be used to call the data
weight=regressor.coef_
weight


# In[11]:


#bais is used for indicating(origin or beginner value)
bais=regressor.intercept_
bais


# In[12]:


#here the regressor predicts the data
Y_pred=regressor.predict(X_test)


# In[13]:


#here the actual and predict is done by r_score
from sklearn.metrics import r2_score
r_score=r2_score(Y_test,Y_pred)


# In[14]:


#r_score nearly to 1 is good model
r_score


# In[20]:


#to save model pickle is used
import pickle
filename="finalized_model_linear.sav"


# In[22]:


#wb is write binary(to read)
pickle.dump(regressor,open(filename,'wb'))


# In[27]:


#to load the saved model
loaded_model=pickle.load(open("finalized_model_linear.sav",'rb'))
result=loaded_model.predict([[13]])


# In[28]:


#it is the final output
result


# In[ ]:




