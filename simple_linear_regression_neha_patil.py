#!/usr/bin/env python
# coding: utf-8

# # Prediction using Supervised ML# 

# # Task 1 : Simple Linear Regression

# #### by Neha Sandeep Patil

# In this task we have to predict the percentage of an student based on the no. of study hours. So this is a simple linear regression task as it involves just 2 variables as hours studied and scores .

# In[1]:


#importing the required libraries
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#read the given dataset and display the first 10 readings 
path="http://bit.ly/w-data"
dataset=pd.read_csv(path)
dataset.head(10)


# In[4]:


#now plot the graph to understand the relationship between the variables.
dataset.plot(x='Hours', y='Scores', style='o') 
plt.title('Relationship between the variables')
#plot variable Hours on X axis
plt.xlabel('Hours of study')
#plot variable Scores on Y axis
plt.ylabel('Scored marks')
#display the graph
plt.show()


# The graph shows that :
# Increase in hours of study there is increase in scored marks . So here , variables have positive linear regression .

# In[5]:


#now we have to divide dataset into attributes(input) and labels(output)
X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 1].values  


# In[6]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0) 


# In[8]:


#Now preparing traning dataset and testing dataset 
from sklearn.linear_model import LinearRegression  
regressor_dataset = LinearRegression() 
regressor_dataset.fit(X_train, y_train) 
print("Done the job !")


# In[10]:


#now we have to plot regression line
regression_line = regressor_dataset.coef_*X+regressor_dataset.intercept_
plt.scatter(X, y)
plt.plot(X, regression_line);
plt.show()


# In[12]:


#making the prediction about our dataset
print(X_test) 
y_prediction = regressor_dataset.predict(X_test)


# In[13]:


#now we get predicted values along the Y axis 
#compare the actual and predicted values
df = pd.DataFrame({'Actual_Value': y_test, 'Predicted_Value': y_prediction})  
print(df)


# In[25]:


#Now final step is to predicted score if a student studies for 9.25 hrs/ day
#put value of X_test as 9.25 in function y_prediction
X_test=9.25
y_prediction = regressor_dataset.predict([[X_test]])
y_prediction


# ### Finally ,Predicted score if a student studies for 9.25 hrs/ day is 93.6917 . 
