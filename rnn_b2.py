
# coding: utf-8

# In[147]:


import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Flatten, BatchNormalization, Conv1D, Activation,LSTM
import sklearn.model_selection as cross_validation
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint
from collections import Counter
pd.set_option("display.max_columns", 100)
import os
from tensorflow.keras.callbacks import TensorBoard

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# In[17]:

data = pd.read_csv('crime.csv',encoding = 'ISO-8859-1')

b2 = data[data['DISTRICT'] == 'B2']

b2_motor = b2[b2['OFFENSE_CODE_GROUP']=='Motor Vehicle Accident Response']

b2_motor.sort_values(['OCCURRED_ON_DATE'])

b2_motor['date_new'] = b2_motor['OCCURRED_ON_DATE']


def change_hour(x):
    m = str(x['date_new']).split(':')
    return '{}:00:{}'.format(m[0],m[2])

b2_motor['modified_data'] = b2_motor.apply(change_hour,axis = 1)

b2_motor['modified_data'] = b2_motor.apply(change_hour,axis = 1)

b2_motor

b2_motor.sort_values(['modified_data'])

b2_motor['modified_data'] = pd.to_datetime(b2_motor['modified_data'])

b2_motor['timestamp']=b2_motor['modified_data'].apply(lambda x:(x-np.datetime64('1970-01-01T00:00:00Z'))/np.timedelta64(1, 's'))

timestamp = sorted(b2_motor['timestamp'])

timestamp_int = []
for each in timestamp:
    timestamp_int.append(int(each))
timestamp_int

all_time = list(range(int(timestamp[0]),int(timestamp[-1])+3600,3600))

timestamp_int_dic = Counter(timestamp_int)

for each in all_time:
    if each in timestamp_int_dic:
        continue
    else:
        timestamp_int_dic[each] = 0

sort_dict = sorted(timestamp_int_dic.keys())

time_interval = []
for each in sort_dict:
    time_interval.append(timestamp_int_dic[each])

sequence_length = 73
result = []
for index in range(len(time_interval) - sequence_length):
    result.append(time_interval[index: index + sequence_length])
result = np.array(result)


# In[148]:


x_train, x_test, y_train, y_test = cross_validation.train_test_split(result[:,:-1], result[:,-1], test_size=0.3)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[149]:


model = Sequential()
model.add(LSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
#model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.1))
#model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.2))
#model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
#model.add(Dropout(0.2))

model.add(Dense(1))


# In[150]:


opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='mean_squared_error',
    optimizer=opt,
    metrics=['accuracy']
)


# In[151]:


NAME = 'RNN_b2'
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones


# In[152]:


model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard, checkpoint]
)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Save model
model.save("models/{}".format(NAME))

