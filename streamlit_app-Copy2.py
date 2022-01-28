#!/usr/bin/env python
# coding: utf-8

# In[5]:


#pip install streamlit


# In[ ]:


#pip install pycaret


# In[1]:


# Additional imports are required
import os 

import numpy as np
import pandas as pd

import pycaret
from pycaret.regression import load_model, predict_model
import streamlit as st


# In[38]:


#pip install streamlit


# In[40]:


# Writing App Title and Description

st.title('Oil Production Forecasting Web App')
st.write('This is a web app to forecast the oil production based on        dates that you can see in the sidebar. Please adjust the        value of dates. After that, click on the Prediction button at the bottom to        see the forecast of the model.')


# In[41]:


Dates = st.sidebar.slider(label = 'Months', min_value = 0,
                        max_value = 200 ,
                        value = 200,
                        step = 1)


# In[42]:


features = {
  'Dates':Dates}


# In[43]:


# Converting Features into DataFrame

features_df  = pd.DataFrame([features])

st.table(features_df)


# In[44]:


# Predicting Star Rating

if st.button('Predict'):
    
    prediction = my_model(model, features_df)
    
    st.write(' Based on feature values, the Oil Production is '+ str(int(prediction)))


# In[45]:





# In[46]:


# Defining Prediction Function

def predict_rating(model, df):
    
    predictions_data = predict_model(estimator = model, data = df)
    
    return predictions_data['Label'][0]


# In[47]:


# Loading Model

model = load_model('Oil-Production-Forecasting')


# In[48]:


# Predicting Star Rating

if st.button('Predict'):
    
    prediction = predict_rating(model, features_df)
    
    st.write(' Based on feature values, the car star rating is '+ str(int(prediction)))


# In[ ]:




