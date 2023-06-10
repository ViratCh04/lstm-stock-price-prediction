import matplotlib.pyplot as plt
from pandas_datareader import data
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def main():
    df = pd.read_csv(os.path.join('Stocks', 'hpq.us.txt'), delimiter=',', usecols=['Date','Open','High','Low','Close'])
    print("Success")
    # Sort df by date
    df = df.sort_values('Date')
    # print first five rows of dataframe
    print(df.head())

    # data visualization
    plt.figure(figsize = (18, 9))
    plt.plot(range(df.shape[0]), (df['Low'] + df['High'])/2.0)
    plt.xticks(range(0, df.shape[0], 500), df['Date'].loc[::500], rotation=45)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mid Price', fontsize=18)
    plt.show()
    
    # Splitting data into training and test sets
    highPrices = df.loc[:, 'High'].to_numpy()
    # AttributeError: 'Series' object has no attribute 'as_matrix'
    lowPrices = df.loc[:, 'Low'].to_numpy()
    midPrices = (highPrices + lowPrices) / 2.0
    trainingData = midPrices[:11000]
    testData = midPrices[11000:]

    # Scaling the data to be between 0 and 1
    # Normalize both test and train data wrt train data as you are not supposed to have access to test data
    scaler = MinMaxScaler()
    trainingData = trainingData.reshape(-1, 1)
    testData = testData.reshape(-1, 1)

    # Train scaler with training data and smooth data by
    # splitting data into 5 windows (2500 data points each)
    smoothingWindowSize = 2500
    for di in range(0, 10000, smoothingWindowSize):
        scaler.fit(trainingData[di:di+smoothingWindowSize, :])
        trainingData[di:di+smoothingWindowSize,:] = scaler.transform(trainingData[di:di+smoothingWindowSize,:])

    # You normalize the last bit of remaining data
    scaler.fit(trainingData[di+smoothingWindowSize:,:])
    trainingData[di+smoothingWindowSize:,:] = scaler.transform(trainingData[di+smoothingWindowSize:,:])

    # Reshaping both training and testing data
    # -1 parameter converts the 2d array into 1d array automatically to assist my laziness
    trainingData = trainingData.reshape(-1)

    # normalize test data
    testData = scaler.transform(testData).reshape(-1)

    # Smoothing the training data by
    # Performing exponential moving average smoothing
    # Now, data will have a smoother curve than the original ragged data
    EMA = 0.0
    gamma = 0.1
    for ti in range(11000):
        EMA = gamma * trainingData[ti] + (1 - gamma) * EMA
        trainingData[ti] = EMA

    # using for visualization and testing purposes
    allMidData = np.concatenate([trainingData, testData], axis=0)

    # Going to use standard averaging and exponential moving average to evaluate results
    # Mean Squared Error(MSE) is used for quantitative evaluation
    # This is done by taking the squared error between the true value one step ahead and the predicted value and averaging it over all predictions

    # Standard Average:
    windowSize = 100
    N = trainingData.size
    stdAvgPredictions = []
    stdAvgX = []
    mseErrors = []
    k = ""

    for predIdx in range(windowSize, N):
        if predIdx >= N:
            date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
        else:
            date = df.loc[predIdx, 'Date']
        stdAvgPredictions.append(np.mean(trainingData[predIdx - windowSize:predIdx]))
        mseErrors.append((stdAvgPredictions[-1] - trainingData[predIdx]) ** 2)
        stdAvgX.append(date)
    print('MSE error for standard averaging: %.5f'%(0.5 * np.mean(mseErrors)))
    # Plotting averaged graph alongside the actual stock graph for qualitative inspection
    plt.figure(figsize = (18, 9))
    plt.plot(range(df.shape[0]), allMidData, color = 'b', label='True')
    plt.plot(range(windowSize, N), stdAvgPredictions, color='orange', label='Prediction')
    #plt.xticks(range(0, df.shape[0], 50), df['Date'].loc[::50], rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Mid Price')
    plt.legend(fontsize=18)
    plt.show()

    # Exponential moving average
    # This is better than standard avg since it responds to recent market trends by focusing on recent data points
    # https://www.investopedia.com/terms/e/ema.asp
    

if __name__ == "__main__":
    main()