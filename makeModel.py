import math
import pandas as pd
import pandas_datareader as web
import numpy as np
import datetime as dt
import time
import pickle
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
import os.path
from os import path
plt.style.use('fivethirtyeight')


class stockClosePricePrediction:   
    
    ##Initialize the parameters..
    today = dt.date.today()
    preDayDate = today - timedelta(1)
    sDate = '2010-01-01'
    eDate = preDayDate
    retryCount = 3
    datasource = 'yahoo'
    
    def __init__(self, stock):
        
        self.stock = stock

        ## hyperparameters..
        self.batchsize=1
        self.epochs=18
        self.data = None
        self.train_data_len = 0
        self.x_train = []
        self.y_train = []
        self.scaler_data = None
        self.dataset = None
        self.scaler = None
        self.model = None
        
    def _read_data(self):
        df = web.DataReader(self.stock, data_source=stockClosePricePrediction.datasource, start=stockClosePricePrediction.sDate, end=stockClosePricePrediction.eDate, retry_count=stockClosePricePrediction.retryCount)
        self.data = df.filter(['Close'])
    
    def _scale_data(self):
        self.dataset = self.data.values

        self.train_data_len = math.ceil(len(self.dataset) * 0.8)

        ## Scale the data
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.scaler_data = self.scaler.fit_transform(self.dataset)
        train_data = self.scaler_data[0:self.train_data_len, :]
        return train_data
    
    def _split_data(self, train_data):        
        i=0
        for i  in range(60, len(train_data)):
            #print(i)
            self.x_train.append(train_data[i-60:i,0])
            self.y_train.append(train_data[i,0])
            
        self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)
        
        #reshape the array
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))
        
    def _build_and_compile_model(self):
        ## Build the model
        self.model = Sequential()

        self.model.add(LSTM(50, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(25))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))

        ## Compile the model
        self.model.compile(optimizer='adam', loss='mean_squared_error')
    
    def _prepare_test_data(self):
        ## create test dataset
        test_data = self.scaler_data[self.train_data_len - 60: ,:]

        x_test = []
        y_test = self.dataset[self.train_data_len: ,:]
        i=0
        for i  in range(60, len(test_data)):
            #print(i)
            x_test.append(test_data[i - 60:i,0])

        ## convert to array and reshape..
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return x_test, y_test
    
    
    def _fit_model_and_make_prediction(self, x_test, b, e):
        ## fit the model
        t1 = time.time()
        self.model.fit(self.x_train, self.y_train, batch_size=b, epochs=e)
        t2 = time.time()
        total_time_taken = (t2-t1)

        ## Prediction..
        predictions = self.model.predict(x_test)
        predictions = self.scaler.inverse_transform(predictions)
        return predictions
    
    def _save_model(self):
        model_name = self.stock + '-' + str(stockClosePricePrediction.today) + '.sav'
        
        if(path.exists(model_name) == False):
            pickle.dump(self.model, open(model_name, 'wb'))
        
    def predict_price(self, x_test):       
        ## build and compile the model
        self._build_and_compile_model()      
        
        ## fit and predict..
        predictions = self._fit_model_and_make_prediction(x_test, self.batchsize, self.epochs)
                
        return predictions
        
    def prepare_data(self):
        ## read the data from the source..
        self._read_data()
        
        ## scale the data..
        d = self._scale_data()
        
        ## slpit the data into train and test
        self._split_data(d)
        
        ## prepare the test data
        x_test, y_test = self._prepare_test_data()
        return x_test, y_test
    
    def calculate_rmse(self, predictions, y_test):
        ## calculate RMSE
        rmse = np.sqrt(np.mean(predictions - y_test)**2)
        return rmse
    
    def save_model(self):
        self._save_model()
        



