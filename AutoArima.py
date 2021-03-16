# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 11:46:21 2020

@author: DBDA
"""



import pandas as pd
import numpy as np
import math as sqrt
import datetime
from numpy import array
import pandas_datareader as net
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
global y_train
global y_test
y_train=[]
y_test=[]



def getData():
        global y_train,y_test,df,close
        df = net.data.get_data_yahoo('TCS.NS',start='2010-01-17',end=datetime.date.today())
        df.reset_index(inplace=True,drop=False)
        df['Adj Close'].tail
        #data = pd.read_csv("C:\\Users\\Shreyas\\Downloads\\TCS.NS (4).csv")
        #rounded.round(1)
        close = df['Adj Close']
        rounded = close
        if(rounded.isnull().values.any()==False):
                print("Data do not have null values")             
                y_train = close[:2461]
                y_test = close[2461:]
                y_test=y_test.reset_index()
                y_test=y_test.drop(['index'], axis = 1)
                #pd.Series([y_test['Adj Close']])
        else:
                print("Data has null records")
                
                
                
getData()                
                
        
##################################################################################

from statsmodels.tsa.ar_model import AR 
# train autoregression
model = AR(y_train)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)


# make predictions
predictions = model_fit.predict(start=len(y_train), 
                                end=len(y_train)+len(y_test)-1, 
                                dynamic=False)

print("AR Model Predicted and Expected Previous 10 Price:-"  )
print("          Predicted\n",predictions)
#print('Test RMSE: %.3f' % sqrt(error))
print("       Expected\n",y_test)

error = mean_squared_error(y_test, predictions)
print('Test RMSE: %.3f' % sqrt(error))


        
############################################################################################

from pmdarima.arima import auto_arima


model = auto_arima(y_train, trace=True, error_action='ignore', 
                   suppress_warnings=True)
model.fit(y_train)

forecast = model.predict(n_periods=len(y_test))
forecast = pd.DataFrame(forecast,index = y_test.index,
                        columns=['Predictions'])


print("ARIMA Model :-"  )
print("          Predicted\n",predictions)

error = mean_squared_error(y_test, forecast)
print('Test RMSE: %.3f' % sqrt(error))
print("\n",y_test)


                
#print("r2_score:-",r2_score(y_test, predictions))
      
