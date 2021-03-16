# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 21:00:02 2020

@author: Shreyas
"""
###############################################################################


import pandas as pd
import numpy as np
import datetime
from numpy import array
import pandas_datareader as net
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
global y_train
global y_test
y_train=[]
y_test=[]


#Function to collect data and check nulls
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
                y_train = close[:2000]
                y_test = close[2000:]
                y_test=y_test.reset_index()
                y_test=y_test.drop(['index'], axis = 1)
                #pd.Series([y_test['Adj Close']])
        else:
                print("Data has null records")

###############################################################################
###############################################################################
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

getData()

#----------------------------------------  Train Start
n_steps = 5

X, y = split_sequence(y_train, n_steps)

for i in range(len(X)):
	print(X[i], y[i])

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))

model = Sequential()
model.add(Bidirectional(LSTM(162, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=200, verbose=0)

lst=[]
l = len(y_train)
start = l-n_steps
while start!=l:
     #lst = lst.append[close[start]]
     print(y_train[start])
     val = y_train[start]
     lst.append(val)
     start = start+1
     
x_input = array([lst])
y_train.tail(5)

x_input = x_input.reshape((1, n_steps, n_features))

result = model.predict(x_input, verbose=1)
print(result)
#------------------------------------------- Train over
print(result)
print(y_test[1])
day = pd.to_datetime(y_test['Date'].tail(1))
print(day)
day = day +  datetime.timedelta(days=5)

print(day," ",result)
#----------------------------------------------- Test
n_steps = 5

X1, y1 = split_sequence(y_test['Adj Close'], n_steps)

for i in range(len(X1)):
	print(X1[i], y1[i])

n_features = 1

X1 = X1.reshape((X1.shape[0], X1.shape[1], n_features))

x=0
lst=[]
predicted = []

while x<len(X1): 
        lst=X1[x]  
        x_input = array([lst])
        x_input = x_input.reshape((1, n_steps, n_features))
        result = model.predict(x_input, verbose=1)
        predicted.append(result.mean())
        x=x+1

np.sqrt(mean_squared_error(y1,predicted))

r2_score(y1,predicted)
      
#______________________________________________________________________________

def PredictValue():
        lst=[]
        l = len(close)
        start = l-n_steps
        while start!=l:
                #lst = lst.append[close[start]]
                print(close[start])
                val = close[start]
                lst.append(val)
                start = start+1
                
                x_input = array([lst])
                y_test.tail(5)
                
                x_input = x_input.reshape((1, n_steps, n_features))
                
                result = model.predict(x_input, verbose=0)
                
                print(result)
                day = pd.to_datetime(df['Date'].tail(1))
                print(day)
                day = day +  datetime.timedelta(days=1)
                print(day," ",result)

###############################################################################
