import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import tensorflow as tf

dataFrame = pd.read_csv("ParisHousing.csv")
print(dataFrame.head())

print(dataFrame.describe())

print(dataFrame.isnull().sum())

sbn.distplot(dataFrame["price"])

print(dataFrame.corr()["price"].sort_values())

print(dataFrame.sort_values("price",ascending = False).head(20))


y = dataFrame["price"].values
x = dataFrame.drop("price", axis=1).values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=15)

print(len(x_train))
print(len(x_test))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(18, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(18, activation='relu'))


model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

model.fit(x=x_train, y = y_train,validation_data=(x_test,y_test),batch_size=250,epochs=400)

lossData = pd.DataFrame(model.history.history)

print(lossData.head())

lossData.plot()

from sklearn.metrics import mean_squared_error, mean_absolute_error

guessArray = model.predict(x_test)

print("Mean Absolute Error: " + str(mean_absolute_error(y_test, guessArray)))
print("*******************************************")
print("Mean Squared Error: " + str(mean_squared_error(y_test, guessArray)))

plt.scatter(y_test, guessArray)
plt.plot(y_test, y_test, "r-*")

















