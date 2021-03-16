# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 00:03:01 2020

@author: Balaji Dhumale
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score



data = pd.read_csv(r'C:\Users\PRANAV NAIK\Desktop\Bala\Study msterial\Project\Project\TCS.NS.csv')
data.head()


df1 = pd.DataFrame(data,columns=['Date', 'Adj Close'])



df = df1.dropna(axis = 0, how ='any') 
print(df)
print(df.shape)

df.describe()
df.mean()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df, lag=31)

#plot_acf(df['Close'], lags=31)
#plot_pacf(df['Close'], lags=31)
plt.show()

y = df['Adj Close']
y_train = y[:232]
y_test = y[232:]



##############################-------AR--------###########################################


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

error = mean_squared_error(y_test, predictions)
print('Test RMSE: %.3f' % sqrt(error))
print("R2_score:%.3f" % r2_score(y_test, predictions))

# plot results
plt.plot(y_test)
plt.plot(predictions, color='red')
plt.legend()
plt.show()


# plot
y_train.plot(color="blue")
y_test.plot(color="pink")
predictions.plot(color="purple")


#######################---ARIMA---##############################

from pmdarima.arima import auto_arima


model = auto_arima(y_train, trace=True, error_action='ignore', 
                   suppress_warnings=True)
model.fit(y_train)

forecast = model.predict(n_periods=len(y_test))
forecast = pd.DataFrame(forecast,index = y_test.index,
                        columns=['Prediction'])
print("R2_score:%.3f" % r2_score(y_test, forecast))

#plot the predictions for validation set
plt.plot(y_train, label='Train')
plt.plot(y_test, label='Valid')
plt.plot(forecast, label='Prediction')
plt.legend()
plt.show()


#################################################################











