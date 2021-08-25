#!/usr/bin/env python
# coding: utf-8

# # Flight Fare Prediction Dataset

# In[1]:


import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 

sns.set()


# In[2]:


# Purpose ? 

# We want to predict flight prices based on several independent variables! 


# ## Import Data

# In[3]:


train_data = pd.read_excel('Data_Train_Flight_Prediction.xlsx')
train_data


# In[4]:


train_data.describe(include='all')


# In[5]:


train_data.info()


# In[6]:


# We'll drop null values 


# In[7]:


train_data.dropna(inplace = True)


# In[8]:


train_data.isnull().sum()


# ## EDA

# In[9]:


train_data.head()


# In[10]:


train_data.columns.values


# In[11]:


# We'lll convert Data of Journey into numerical dates 


# In[12]:


train_data['Date_of_Journey']


# In[13]:


train_data['Journey_day'] = pd.to_datetime(train_data.Date_of_Journey, format = "%d/%m/%Y").dt.day
train_data['Journey_day'].value_counts()


# In[14]:


train_data['Journey_month'] = pd.to_datetime(train_data['Date_of_Journey'], format = "%d/%m/%Y").dt.month


# In[15]:


train_data.head()


# In[16]:


train_data.drop(['Date_of_Journey'], axis = 1, inplace = True)


# In[17]:


train_data.head()


# In[18]:


train_data['Dep_hour'] = pd.to_datetime(train_data['Dep_Time']).dt.hour

train_data['Dep_minute'] = pd.to_datetime(train_data['Dep_Time']).dt.minute

train_data.drop(['Dep_Time'], axis = 1, inplace = True )


# In[19]:


train_data.head()


# In[20]:


train_data['Arrival_hour'] = pd.to_datetime(train_data['Arrival_Time']).dt.hour 

train_data['Arrival_minute'] = pd.to_datetime(train_data['Arrival_Time']).dt.minute

train_data.drop(['Arrival_Time'], axis = 1, inplace = True )


# In[21]:


train_data.head()


# In[22]:


# Time taken by plane to reach destination is called Duration
# It is the differnce betwwen Departure Time and Arrival time


# Assigning and converting Duration column into list
duration = list(train_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration


# In[23]:


train_data['Duration_hours'] = duration_hours 
train_data['Duration_mins'] = duration_mins 


# In[24]:


train_data.head()


# In[25]:


train_data.drop(['Duration'], axis = 1, inplace = True )


# In[26]:


train_data


# ## Handling Categorical Data

# In[28]:


train_data.columns


# In[29]:


train_data['Airline'].value_counts()


# ### Airline Column

# In[30]:



# From graph we can see that Jet Airways Business have the highest Price.
# Apart from the first Airline almost all are having similar median


# In[ ]:





# In[31]:


sns.catplot(y = 'Price', x = 'Airline', data = train_data.sort_values('Price', ascending = False), kind = 'boxen',height = 6,aspect = 3)


# In[32]:


# As Airline is Nominal Categorical data we will perform OneHotEncoding

Airline = train_data[["Airline"]]

Airline = pd.get_dummies(Airline, drop_first= True)

Airline.head()


# In[33]:


train_data['Airline'].value_counts()


# In[34]:


train_data.head()


# ### Source Column

# In[35]:


train_data['Source'].value_counts()


# In[36]:


Source = train_data[["Source"]]

Source = pd.get_dummies(Source, drop_first= True)

Source.head()


# In[37]:


# Destination Column 


# In[38]:


train_data['Destination'].value_counts()


# In[39]:


Destination = train_data[['Destination']]

Destination = pd.get_dummies(Destination, drop_first= True)

Destination.head()


# In[40]:


train_data['Route']


# In[41]:


train_data.head()


# In[42]:


train_data['Additional_Info'].value_counts()


# In[43]:


#Dropping Route and Additional Info 

train_data.drop(['Route', 'Additional_Info'], axis = 1, inplace = True)


# In[44]:


train_data['Total_Stops'].value_counts()


# In[45]:


# total stops is ordnal categorical type so we use label encoding
train_data.replace({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}, inplace = True)


# In[46]:


train_data


# In[47]:


train_data.head()


# In[48]:


# Concatenate dataframe --> train_data + Airline + Source + Destinations

data_train = pd.concat([train_data, Airline, Source, Destination], axis = 1)


# In[49]:


data_train.head()


# In[50]:


data_train.columns


# In[51]:


data_train.drop(['Airline', 'Source', 'Destination'], axis = 1, inplace = True)


# In[52]:


data_train


# ## Test Set 

# In[53]:


test_data = pd.read_excel('Test_set.xlsx')


# In[54]:


test_data.head()


# In[55]:


# Preprocessing

print("Test data Info")
print("-"*75)
print(test_data.info())

print()
print()

print("Null values :")
print("-"*75)
test_data.dropna(inplace = True)
print(test_data.isnull().sum())






# Date_of_Journey
test_data["Journey_day"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day
test_data["Journey_month"] = pd.to_datetime(test_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month
test_data.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Dep_Time
test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour
test_data["Dep_min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute
test_data.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour
test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute
test_data.drop(["Arrival_Time"], axis = 1, inplace = True)



# Duration
duration = list(test_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

# Adding Duration column to test set
test_data["Duration_hours"] = duration_hours
test_data["Duration_mins"] = duration_mins
test_data.drop(["Duration"], axis = 1, inplace = True)


# Categorical data

print("Airline")
print("-"*75)
print(test_data["Airline"].value_counts())
Airline = pd.get_dummies(test_data["Airline"], drop_first= True)

print()

print("Source")
print("-"*75)
print(test_data["Source"].value_counts())
Source = pd.get_dummies(test_data["Source"], drop_first= True)

print()

print("Destination")
print("-"*75)
print(test_data["Destination"].value_counts())
Destination = pd.get_dummies(test_data["Destination"], drop_first = True)

# Additional_Info contains almost 80% no_info
# Route and Total_Stops are related to each other
test_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

# Replacing Total_Stops
test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

# Concatenate dataframe --> test_data + Airline + Source + Destination
data_test = pd.concat([test_data, Airline, Source, Destination], axis = 1)

data_test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)

print()
print()

print("Shape of test data : ", data_test.shape)









# In[56]:


data_test.head()


# # Feature Selection 

# In[57]:


data_train.shape


# In[58]:


data_train.columns


# In[59]:


data_train.head()


# In[60]:


X = data_train.loc[:, ['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_minute', 'Arrival_hour', 'Arrival_minute', 'Duration_hours',
       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]
X.head()


# In[61]:


data_train.head()


# In[62]:


y = data_train.iloc[:, 1]
y.head()


# In[63]:


# Find correlation between Independent and dependent attributes 
plt.figure(figsize = (18,18))
sns.heatmap(train_data.corr(), annot = True, cmap = 'RdYlGn')


# In[64]:


from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X, y)


# In[65]:


print(selection.feature_importances_)


# In[66]:


#plot graph of feature importances for better visualization

plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# In[67]:


plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index = X.columns)
feat_importances.nlargest(20).plot(kind='barh')


# ## Fitting Model using Random Forest 
# 
# 1. Split dataset into train and test set in order to prediction w.r.t X_test
# 2. If needed do scaling of data
# 3. Scaling is not done in Random forest
# 4. Import model
# 5. Fit the data
# 6. Predict w.r.t X_test
# 7. In regression check RSME Score
# 8. Plot graph

# In[68]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[69]:


from sklearn.ensemble import RandomForestRegressor 
reg_rf = RandomForestRegressor() 
reg_rf.fit(X_train, y_train)


# In[70]:


y_pred = reg_rf.predict(X_test)


# In[71]:


reg_rf.score(X_train,y_train)


# In[72]:



reg_rf.score(X_test,y_test)


# In[73]:


sns.distplot(y_test-y_pred)


# In[74]:


plt.scatter(y_test, y_pred, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")


# In[75]:


from sklearn import metrics


print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[76]:


# RMSE/(max(DV)-min(DV))

2090.5509/(max(y)-min(y))


# In[77]:


metrics.r2_score(y_test, y_pred)

