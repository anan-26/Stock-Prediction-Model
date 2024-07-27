# -*- coding: utf-8 -*-

# Predict the price of the stock of a company

# before running please install the libraries if not already installed
# pip install numpy
# pip install pandas
# pip install -U scikit-learn
# pip install tensorflow

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tensorflow.keras import layers, models, losses, callbacks
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler


"""Loading the dataset"""

df_train = pd.read_csv('train.csv')['price'].values
df_train = df_train.reshape(-1, 1)
df_test = pd.read_csv('test.csv')['price'].values
df_test = df_test.reshape(-1, 1)

dataset_train = np.array(df_train)
dataset_test = np.array(df_test)

"""Pre process your data"""

## Pre process your data
scaler = StandardScaler()
dataset_train = scaler.fit_transform(dataset_train)
dataset_test = scaler.transform(dataset_test)
##########################################

"""### We create the X_train and Y_train from the dataset train
We take a price on a date as y_train and save the previous 50 closing prices as x_train
"""

trace_back = 50
def create_dataset(df):
    x, y = [], []
    for i in range(trace_back, len(df)):
        x.append(df[i-trace_back:i, 0])
        y.append(df[i, 0])
    return np.array(x),np.array(y)

x_train, y_train = create_dataset(dataset_train)

x_test, y_test = create_dataset(dataset_test)

"""Build your RNN model

1. Design a RNN model that takes in your x_train and do prediction on x_test
2. Model should be able to predict on x_test using model.predict(x_test)
3. Do not use any pretrained model.
"""

## Your RNN model goes here
model = models.Sequential()
model.add(layers.SimpleRNN(100, input_shape=(50, 1), return_sequences=True))
model.add(layers.Dropout(0.2))
model.add(layers.GRU(50))
model.add(layers.Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# model.summary()

history = model.fit(
    x_train, y_train, batch_size=16, epochs=30,
    validation_split=0.3, shuffle=True, 
    callbacks=callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=5))

##########################################

"""Predictions on X_test
"""

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

"""Checking the Root Mean Square Error on X_test"""

rmse_score = mean_squared_error([x[0] for x in y_test_scaled], [x[0] for x in predictions], squared=False)
print("RMSE",STUDENT_ID,":",rmse_score)
