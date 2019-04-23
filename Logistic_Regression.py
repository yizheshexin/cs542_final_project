import scipy.io as scio
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt  
import math
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC


# # Data Cleaning

# In[89]:


boston_crime = pd.read_csv('/Users/zhukaikang/Desktop/BU_courses/2019_spring/CS_542/project/cs542_final_project/crime.csv', encoding = 'gbk')
boston_crime_dataset = boston_crime.values.tolist()


# In[90]:


for i in range(0, len(boston_crime_dataset)):
    val = boston_crime_dataset[i][6]
    if(val != val):
        boston_crime_dataset[i][6] = 'N'
#print(len(boston_crime_dataset))


# In[91]:


boston_crime_df = pd.DataFrame(boston_crime_dataset)
boston_crime_df = boston_crime_df.dropna()
boston_crime_list = np.array(boston_crime_df).tolist()
#print(len(boston_crime_list))
#print(boston_crime_list[0])


# In[93]:


boston_crime_list = [[offenseGroup, reportingArea, district, shooting, occurredDate, hour, day, month, year, street, lat, long] 
                     for (incidentNumber, offenseCode, offenseGroup, description, district, reportingArea, shooting, occurredDate,
                          year, month, day, hour, ucrPart, street, lat, long, latAndLong) in boston_crime_list]

#print(boston_crime_list[0][3])
#preprosessing: get the day in OccurredDate
for i in range(len(boston_crime_list)):
    boston_crime_list[i][4] = int(boston_crime_list[i][4].split()[0].split('-')[2])

print(boston_crime_list[0][4])


# In[55]:


boston_crime_df = pd.DataFrame(boston_crime_list)
boston_crime_df.columns = ['offenseGroup', 'reportingArea', 'district', 'shooting', 'occurredDate', 'hour', 'day', 'month', 'year', 'street', 'lat', 'long']

'''
enc = OneHotEncoder(handle_unknown='ignore')
x = [['495',], ['795',], ['329',]]
x = enc.transform(x).toarray()
print(x)
'''
# # Modeling

# In[67]:


# Y - offenseGroup
# X - reportingArea, shooting, hour, day, month, year, latitude, longitude
boston_crime_df['offenseGroup'].value_counts().head(20)


# In[68]:


offenseGroup_list = ('Motor Vehicle Accident Response', 'Larceny', 'Medical Assistance', 'Investigate Person',
                    'Other', 'Simple Assault', 'Vandalism', 'Drug Violation', 'Verbal Disputes', 'Towed',
                    'Investigate Property', 'Larceny From Motor Vehicle', 'Property Lost', 'Warrant Arrests',
                    'Aggravated Assault', 'Fraud', 'Residential Burglary', 'Violations', 'Missing Person Located',
                    'Auto Theft')
boston_crime_matrix = pd.DataFrame()
for i in range(0, len(offenseGroup_list)):
    boston_crime_matrix = boston_crime_matrix.append(boston_crime_df.loc[boston_crime_df['offenseGroup'] == offenseGroup_list[i]])
# X - reportingArea, shooting, hour, day, month, year, latitude, longitude
# offenseGroup, reportingArea, shooting, occurredDate, hour, day, month, year, street, lat, long
columns = ['reportingArea', 'district', 'shooting', 'occurredDate', 'hour', 'day', 'month', 'year', 'lat', 'long', 'offenseGroup']
boston_crime_matrix = boston_crime_matrix[columns]
boston_crime_matrix.fillna(0, inplace = True)

#print(len(boston_crime_matrix))


# In[69]:


# reportingArea
boston_crime_matrix['reportingArea'] = pd.to_numeric(boston_crime_matrix['reportingArea'], errors='coerce')
boston_crime_matrix['reportingArea'].unique()
#print(boston_crime_matrix['reportingArea'].unique())

# In[70]:

#district
boston_crime_matrix['district'] = boston_crime_matrix['district'].map({
    'B3':1, 
    'E18':2, 
    'B2':3, 
    'E5':4, 
    'C6':5, 
    'D14':6, 
    'E13':7, 
    'C11':8, 
    'D4':9, 
    'A7':10, 
    'A1':11, 
    'A15':12
})
boston_crime_matrix['district'].unique()

# shooting
boston_crime_matrix['shooting'] = boston_crime_matrix['shooting'].map({
    'N' : 0,
    'Y' : 1
})
boston_crime_matrix['shooting'].unique()


# In[71]:


# hour
boston_crime_matrix['hour'].unique()


# In[72]:


# day
boston_crime_matrix['day'] = boston_crime_matrix['day'].map({
    'Monday' : 1,
    'Tuesday' : 2,
    'Wednesday' : 3,
    'Thursday' : 4,
    'Friday' : 5,
    'Saturday' : 6,
    'Sunday' : 7
})
boston_crime_matrix['day'].unique()


# In[73]:


# month
boston_crime_matrix['month'].unique()


# In[74]:


# year
boston_crime_matrix['year'].unique()


# In[75]:


# latitude, longitude
boston_crime_matrix[['lat', 'long']] = boston_crime_matrix[['lat', 'long']].dropna()
boston_crime_matrix[['lat', 'long']] = boston_crime_matrix[['lat', 'long']].loc[(boston_crime_matrix['lat'] > 40) & (boston_crime_matrix['long'] < -70)] 

boston_crime_matrix[['lat', 'long']].head(10)


# In[76]:


X = boston_crime_matrix[['reportingArea', 'district', 'shooting', 'occurredDate', 'hour', 'day', 'month', 'year', 'lat', 'long']]
#print(np.array(X).tolist()[0:5])

Y = boston_crime_matrix['offenseGroup']
Y = Y.map({
    'Motor Vehicle Accident Response' : 1,
    'Larceny' : 2, 
    'Medical Assistance' : 3,
    'Investigate Person' : 4, 
    'Other' : 5,
    'Simple Assault' : 6,
    'Vandalism' : 7,
    'Drug Violation' : 8,
    'Verbal Disputes' : 9, 
    'Towed' : 10,                            
    'Investigate Property' : 11, 
    'Larceny From Motor Vehicle' : 12,
    'Property Lost' : 13,
    'Warrant Arrests' : 14,
    'Aggravated Assault' : 15,
    'Fraud' : 16,
    'Residential Burglary' : 17,
    'Violations' : 18,
    'Missing Person Located' : 19,
    'Auto Theft' : 20                   
})
Y.unique()
#Y_enc = [[var] for var in Y]
#print(Y[0:5])

X = X.fillna(X.mean())
Y = Y.fillna(Y.mean())
#one-hot encoding
X_enc = [[t0,t1,t2,t3,t4,t5,t6,t7] for [t0,t1,t2,t3,t4,t5,t6,t7,t8,t9] in np.array(X).tolist()]
#print(X_trainenc[0])
#[[t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7]] lambda t: for t in np.array(X_train).tolist()]
enc = OneHotEncoder()
clf = enc.fit(np.array(X_enc).tolist())
array = enc.transform(np.array(X_enc).tolist()).toarray()
lat_lon = [[t8,t9] for [t0,t1,t2,t3,t4,t5,t6,t7,t8,t9] in np.array(X).tolist()]
X = np.hstack((array, lat_lon))
#X = array

#X = pd.DataFrame()

# In[78]:


X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    np.array(Y).tolist(), 
    test_size = 0.1
)

#X_train = X_train.fillna(X_train.mean())
#Y_train = Y_train.fillna(Y_train.mean())
#X_test = X_test.fillna(X_test.mean())
#Y_test = Y_test.fillna(Y_test.mean())


# Classify

# In[79]:

'''#BernoulliNB
bernoulli = BernoulliNB()
clf = bernoulli.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
'''
#Logistic Regression classifier
LogisticRegression = LogisticRegression()
clf = LogisticRegression.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

'''#Decision Tree classifier
dec_tree = DecisionTreeClassifier()
clf = dec_tree.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
'''
'''#ExtraTreeClassifier
ext_tree = ExtraTreeClassifier()
clf = ext_tree.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
'''
'''#KNeighborsClassifier
neigh = KNeighborsClassifier()
clf = neigh.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
'''
'''#LGBMClassifier
clf = LGBMClassifier()
clf = clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
'''
'''
#RandomForestClassifier
clf = RandomForestClassifier()
clf = clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
'''
'''
#GaussianNB
clf = GaussianNB()
clf = clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_test)
'''
print(Y_pred[0:5])

#print(Y_pred[0:5])
#print(Y_pred[0:10])
#print(Y_test[0:5])
# Y_test = Y_test.values
Y_pred = np.array(Y_pred)
#Y_pred = np.around(Y_pred)
Y_test = np.array(Y_test)
correct = np.sum(Y_pred==Y_test)
print(correct)
size = float(len(Y_test))
acc = (correct/size)*100
#print(size)
print("the accuracy is ")
print(acc)
#mse = mean_squared_error(Y_test, Y_pred, multioutput = 'uniform_average')
#print(mse)
#print(clf.intercept_)
#print(clf.coef_)

