import makeModel
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

class util:
    
    ##Initialize the parameters..
    today = dt.date.today()
    preDayDate = today - timedelta(1)
    sDate = '2010-01-01'
    eDate = preDayDate
    retryCount = 3
    datasource = 'yahoo'
    
    def __init__(self):
        
    def predictions_and_rmse(self, stk):
        obj = makeModel.stockClosePricePrediction(stk, 17, 1)

        x_text, y_test = obj.prepare_data()
        predictions = obj.predict_price(x_text)
        
        rmse = obj.calculate_rmse(predictions, y_test)
        
        return predictions, rmse
    
    def prdicted_Price(self, stk):
        
        ## make predictions on new data..
        quote = web.DataReader(self, stk, data_source=util.datasource, start=util.sDate, end=util.eDate, retry_count=util.retryCount)

        new_df = quote.filter(['Close'])
        
        scaler = MinMaxScaler(feature_range=(0,1))
        
        ## get last 60 days price
        last_60_days = new_df[-60:].values
        last_60_days_scaled = scaler.transform(last_60_days)
        
        X_test = []
        X_test.append(last_60_days_scaled)                                       
        ## create array
        X_test = np.array(X_test)
        ## reshape
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        ## load the model.
        model_name = stk + '-' + str(stockClosePricePrediction.today) + '.sav'
        model = pickle.load(open(model_name, 'rb'))
                                               
        ## make predictions..
        predicted_price = model.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_price)
                                               
        return predicted_price[0][0]
    
    def actual_price(self, stk):
        actual = web.DataReader(stk, data_source='yahoo', start=today, end=today, retry_count=util.retryCount)
        actual_price = actual['Close']
        
        return actual_price[0]
        
        
        
   
