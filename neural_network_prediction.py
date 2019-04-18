
# coding: utf-8

# In[53]:


import pandas as pd
import numpy
import tensorflow as tf
from tensorflow.keras.layers import Dense
import sklearn.model_selection as cross_validation
pd.set_option("display.max_columns", 100)


# In[17]:


data = pd.read_csv('crime.csv',encoding = 'ISO-8859-1')


# In[18]:


data['DAY'] = data['OCCURRED_ON_DATE'].apply(lambda x:x.split('-')[2].split(' ')[0])
data['DAY'] = data['DAY'].astype('int32')


# In[36]:


filter_data = data[['YEAR', 'MONTH', 'DAY', 'HOUR', 'Lat', 'Long']]


# In[33]:


one_hot = pd.get_dummies(data['OFFENSE_CODE_GROUP'])


# In[37]:


new_data = filter_data.merge(one_hot,left_index = True, right_index = True)


# In[39]:


new_data = new_data.dropna()


# In[44]:





# In[184]:



# In[187]:


data


# In[192]:


one_hot_district = pd.get_dummies(data['DISTRICT'])


# In[200]:


new_data_district = data[['YEAR', 'MONTH', 'DAY', 'HOUR']].merge(one_hot_district,left_index = True, right_index = True)
new_data_district = new_data_district.merge(one_hot,left_index = True, right_index = True)
new_data_district = new_data_district.dropna()




# In[202]:


x_data = new_data_district[['YEAR', 'MONTH', 'DAY', 'HOUR', 'A1', 'A15', 'A7', 'B2', 'B3', 'C11',
       'C6', 'D14', 'D4', 'E13', 'E18', 'E5']]
y_data = new_data_district[['Aggravated Assault',
       'Aircraft', 'Arson', 'Assembly or Gathering Violations', 'Auto Theft',
       'Auto Theft Recovery', 'Ballistics', 'Biological Threat', 'Bomb Hoax',
       'Burglary - No Property Taken', 'Commercial Burglary',
       'Confidence Games', 'Counterfeiting', 'Criminal Harassment',
       'Disorderly Conduct', 'Drug Violation', 'Embezzlement', 'Evading Fare',
       'Explosives', 'Fire Related Reports', 'Firearm Discovery',
       'Firearm Violations', 'Fraud', 'Gambling', 'HOME INVASION',
       'HUMAN TRAFFICKING', 'HUMAN TRAFFICKING - INVOLUNTARY SERVITUDE',
       'Harassment', 'Harbor Related Incidents', 'Homicide',
       'INVESTIGATE PERSON', 'Investigate Person', 'Investigate Property',
       'Landlord/Tenant Disputes', 'Larceny', 'Larceny From Motor Vehicle',
       'License Plate Related Incidents', 'License Violation',
       'Liquor Violation', 'Manslaughter', 'Medical Assistance',
       'Missing Person Located', 'Missing Person Reported',
       'Motor Vehicle Accident Response', 'Offenses Against Child / Family',
       'Operating Under the Influence', 'Other', 'Other Burglary',
       'Phone Call Complaints', 'Police Service Incidents',
       'Prisoner Related Incidents', 'Property Found', 'Property Lost',
       'Property Related Damage', 'Prostitution', 'Recovered Stolen Property',
       'Residential Burglary', 'Restraining Order Violations', 'Robbery',
       'Search Warrants', 'Service', 'Simple Assault', 'Towed', 'Vandalism',
       'Verbal Disputes', 'Violations', 'Warrant Arrests']]


# In[239]:


x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_data, y_data, test_size=0.3)


# In[238]:


sgd = tf.keras.optimizers.SGD(lr=0.25, momentum=0.1, decay=0.1, nesterov=False)
model_district = tf.keras.models.Sequential()

model_district.add(Dense(100, input_dim=16,activation = 'sigmoid'))
model_district.add(Dense(1000,activation = 'elu'))
model_district.add(Dense(1000,activation = 'selu'))
model_district.add(Dense(1000,activation = 'softplus'))
model_district.add(Dense(67,activation = 'softmax'))
model_district.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# In[244]:


count = 0
while True:
    count += 1
    result = model_district.fit(x_train, y_train, batch_size = 128)
    print('loop time:',count)
    print('acc:',result.history['acc'][0])
    if result.history['acc'][0]>0.3:
        break


# In[234]:


model_district.save('result_model.model')

