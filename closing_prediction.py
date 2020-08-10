#Dakshay Mehta, dakshay.rio@gmail.com

#Description: This is program uses an artificial intelligence neural network LSTM, to predict the closing stock price of a TICKER


import math
import pandas_datareader as web 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')



#Get the stock quote, and the basic details

df = web.DataReader('aapl', data_source='yahoo', start='2012-01-01', end='2020-08-01')
#Show the data 
print(df)


#Get the number of rows and cells
print(df.shape)

#Visualize the closing price of the stock 
plt.figure(figsize = (16,8))
plt.title('Closing price in the Past')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close price USD', fontsize = 18)
plt.show()

#Create a new dataframe, with the close column in it 

data = df.filter(['Close'])

#Convert the dataframe to numpy dataset 

dataset = data.values

print("ASDFASDFASDFASDFA:",len(dataset))
print("daata>>>>  " , data)
#getting the numbber of rows to train the model 

training_data_len = math.ceil(len(dataset)*.8)


#Scaling the data, before presenting it to the neural network

scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(dataset)

print(scaled_data)

#creating the training dataset now

train_data = scaled_data[0:training_data_len , :]

#spliting the data into x_train and the y_train data sets 

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()


#Covert the x_training and y_training to numpy arrays 

x_train, y_train = np.array(x_train), np.array(y_train)


#reshape the data 
x_train = np.reshape(x_train,(1668, x_train.shape[1], 1) )
print(x_train.shape)

#BUILD THE LSTM MODEL 

model = Sequential()
model.add(LSTM(60, return_sequences=True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(60, return_sequences=False))
model.add(Dense(30))
model.add(Dense(5))

#compiling the model 
model.compile(optimizer='adam', loss = 'mean_squared_error')

#Train the model 

model.fit(x_train, y_train, batch_size=1, epochs=5)


#Creating the testing dataset

#Create a new array, which would have the skilled values from 1668 to 2003
test_data = scaled_data[training_data_len - 60: , :]

#create the dataset x_test and y_test

x_test = []
y_test = dataset[training_data_len:, :]

#Create the x_test set 

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])



#convert the data to a numpy array 

x_test = np.array(x_test)

#reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print(x_test)

#Get the models predicted values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
print(predictions)
#Evaluate the model, therefore we need the RMSE (root mean square of the model), the lower values indicate a better prediction

RMSE = np.sqrt( np.mean(predictions- y_test)**2)
print("Accuracy Number: ",RMSE)


#Plot the data

train = data[:training_data_len]
valid = dataset[training_data_len:]

print("data for valid", valid)







#showing the valid and the predicted prices 
#print(valid)


#Try and predict the closing price of the ticker
#Get the quote 

tsla_quote = web.DataReader('aapl', data_source='yahoo', start = '2012-01-01', end = '2020-08-08')

#Create the new dataframe 

new_df = tsla_quote.filter(['Close'])

#Get the last 60 days of the closing price 

last_60_days = new_df[-60:].values 

#Scale the data to be the values between 0 and 1 

last_60_days_scaled = scaler.transform(last_60_days)


#create an empty list 

X_test = []

#Then append the past 60 days to this list 

X_test.append(last_60_days_scaled)


#Convert the X_test to a numpy array 

X_test = np.array(X_test)

#reshape the data

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


#Get the predicted scales price 
pred_price = model.predict(X_test)

#Then you have to undo the training 

pred_price = scaler.inverse_transform(pred_price)
print("The predicted price for the date is:",pred_price)

