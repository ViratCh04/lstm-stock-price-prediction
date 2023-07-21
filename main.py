import matplotlib.pyplot as plt
from pandas_datareader import data
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# TODO: Migrate the entire program to TF 2.x without causing unfixable errors.

tf.compat.v1.disable_eager_execution()
# tf.compat.v1.disable_v2_behavior()

def main():
    df = pd.read_csv(os.path.join('Stocks', 'hpq.us.txt'), delimiter=',', usecols=['Date','Open','High','Low','Close'])
    print("Success")
    # Sort df by date
    df = df.sort_values('Date')
    # print first five rows of dataframe
    print(df.head())

    # data visualization
    """
    plt.figure(figsize = (18, 9), num="Data Visualisation")
    plt.plot(range(df.shape[0]), (df['Low'] + df['High'])/2.0)
    plt.xticks(range(0, df.shape[0], 500), df['Date'].loc[::500], rotation=45)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mid Price', fontsize=18)
    plt.show()
    """
    # Splitting data into training and test sets
    highPrices = df.loc[:, 'High'].to_numpy()
    lowPrices = df.loc[:, 'Low'].to_numpy()
    midPrices = (highPrices + lowPrices) / 2.0
    trainingData = midPrices[:11000]
    testData = midPrices[11000:]

    # Scaling the data to be between 0 and 1
    trainingData = trainingData.reshape(-1, 1)
    testData = testData.reshape(-1, 1)

    # Train scaler with training data and smooth data by splitting data into 5 windows (2500 data points each)
    scaler = MinMaxScaler()
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

    # normalize 1-D test data
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
        stdAvgPredictions.append(np.mean(trainingData[predIdx - windowSize : predIdx]))
        mseErrors.append((stdAvgPredictions[-1] - trainingData[predIdx]) ** 2)
        stdAvgX.append(date)
    print('MSE error for standard averaging: %.5f'%(0.5 * np.mean(mseErrors)))
    """
    # Plotting averaged graph alongside the actual stock graph for qualitative inspection
    plt.figure(figsize = (10, 5), num="Standard Average Predictions")
    #plt.get_current_fig_manager().full_screen_toggle()
    plt.plot(range(df.shape[0]), allMidData, color = 'b', label='True')
    plt.plot(range(windowSize, N), stdAvgPredictions, color='orange', label='Prediction')
    #plt.xticks(range(0, df.shape[0], 50), df['Date'].loc[::50], rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Mid Price')
    plt.legend(fontsize=18)
    plt.show()
    """
    # Exponential moving average
    # This is better than standard avg since it responds to recent market trends by focusing on recent data points
    # https://www.investopedia.com/terms/e/ema.asp
    windowSize = 100
    N = trainingData.size

    runAvgPredictions = []
    runAvgX = []
    mseErrors = []
    emaToday = 0.0
    runAvgPredictions.append(emaToday)
    decay = 0.5

    for predIdx in range(1, N):
        emaToday = emaToday * decay + (1.0 - decay) * trainingData[predIdx - 1]
        runAvgPredictions.append(emaToday)
        mseErrors.append((runAvgPredictions[-1] - trainingData[predIdx]) ** 2)
        runAvgX.append(date)
    print("MSE error for EMA Averaging: %.5f"%(0.5 * np.mean(mseErrors)))
    """
    # Visualising for comparison
    plt.figure(figsize = (10, 5), num="EMA Predictions vs True Values")
    #plt.get_current_fig_manager().full_screen_toggle()
    plt.plot(range(df.shape[0]), allMidData, color='b', label="True")
    plt.plot(range(0, N), runAvgPredictions, color="orange", label="Predictions")
    plt.xlabel("Date")
    plt.ylabel("Mid Price")
    plt.legend(fontsize=18)
    plt.show()
    """
    # Implementing a data generator to train data as it feeds data to model in real-time

    class DataGeneratorSeq(object):

        def __init__(self, prices, batchsize, numUnroll):
            self._prices = prices
            self._pricesLength = len(self._prices) - numUnroll
            self._batchsize = batchsize
            self._numUnroll = numUnroll
            self._segments = self._pricesLength // self._batchsize
            self._cursor = [offset * self._segments for offset in range(self._batchsize)]

        def nextBatch(self):
            batchData = np.zeros((self._batchsize), dtype=np.float32)
            batchLabels = np.zeros((self._batchsize), dtype=np.float32)

            for b in range(self._batchsize):
                if self._cursor[b] + 1 >= self._pricesLength:
                    #self._cursor[b] = b * self._segments
                    self._cursor[b] = np.random.randint(0, (b+1) * self._segments)

                batchData[b] = self._prices[self._cursor[b]]
                batchLabels[b] = self._prices[self._cursor[b] + np.random.randint(0,5)]

                self._cursor[b] = (self._cursor[b] + 1) % self._pricesLength

            return batchData, batchLabels
        
        def unrollBatches(self):
            unrollData, unrollLabels = [], []
            init_data, init_label = None, None
            for ui in range(self._numUnroll):
                data, labels = self.nextBatch()

                unrollData.append(data)
                unrollLabels.append(labels)

            return unrollData, unrollLabels
        
        def resetIndices(self):
            for b in range(self._batchsize):
                self._cursor[b] = np.random.randint(0, min((b+1)*self._segments, self._pricesLength - 1))

    dg = DataGeneratorSeq(trainingData, 5, 5)
    uData, uLabels = dg.unrollBatches()
    
    for ui,(dat, lbl) in enumerate(zip(uData, uLabels)):
        print('\n\nUnrolled Indexx %d'%ui)
        datInd = dat
        lblInd = lbl
        print('\tInputs: ', dat)
        print('\n\tOutput: ', lbl)

    # Defining hyperparameters
    D = 1 # dimensionality of data, data is 1D for now
    numUnrollings = 55 # Number of time steps you look into the future
    batchSize = 500 # number of samples in a batch
    numNodes = [200, 200, 150] # Number of hidden nodes/neurons in each layer of deep LSTM
    nLayers = len(numNodes) # number of layers
    dropout = 0.2

    tf.compat.v1.reset_default_graph() # important for a model to run multiple times

    # Defining placeholders for training inputs and labels
    trainInputs, trainOutputs = [], []

    # You unroll the input over time defining placeholders for each time step
    for ui in range(numUnrollings):
        trainInputs.append(tf.compat.v1.placeholder(tf.float32, shape=[batchSize, D], name='train_inputs_%d'%ui))
        trainOutputs.append(tf.compat.v1.placeholder(tf.float32, shape=[batchSize, 1], name='train_outputs_%d'%ui))

    # Defining parameters for LSTM and regression layer
    # using MultiRNNCell in TF to encapsulate the three LSTMCell objects
    # Creating dropout implemented LSTM cells as they improve performance and reduce overfitting
    lstmCells = [
        tf.keras.layers.LSTMCell(units=numNodes[li])
        for li in range(nLayers)]
    dropLstmCells = [tf.compat.v1.nn.rnn_cell.DropoutWrapper(
        lstm, input_keep_prob = 1.0, output_keep_prob = 1.0 - dropout, state_keep_prob = 1.0 - dropout
    ) for lstm in lstmCells]
    #dropMultiCell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(dropLstmCells)
    dropMultiCell = tf.keras.layers.StackedRNNCells(dropLstmCells)
    #multiCell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(lstmCells)
    multiCell = tf.keras.layers.StackedRNNCells(lstmCells)

    # w and b denote the layers of LSTMs and linear regression layer
    w = tf.compat.v1.get_variable('w', shape = [numNodes[-1], 1], initializer = tf.keras.initializers.GlorotUniform())
    b = tf.compat.v1.get_variable('b', initializer = tf.random.uniform([1], -0.1, 0.1))

    # create cell state and hidden state variables to maintain the state of LSTM
    c , h = [], []
    initialState = []
    for li in range(nLayers):
        c.append(tf.Variable(tf.zeros([batchSize, numNodes[li]]), trainable = False))
        h.append(tf.Variable(tf.zeros([batchSize, numNodes[li]]), trainable = False))
#        initialState.append(tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c[li], h[li]))

    # Do several tensor transformations, because the function dynamic_rnn takes a specific input format
    # Note to self to check out https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN
    allInputs = tf.concat([tf.expand_dims(t, 0) for t in trainInputs], axis = 0)

    # allOutputs is [seqLength, batchSize, numNodes]
    allLstmOutputs = tf.keras.layers.RNN(
        dropMultiCell, return_sequences = True,
        return_state = True, time_major = True)
    
    allLstmOutputs = tf.reshape(allLstmOutputs, [batchSize * numUnrollings, numNodes[-1]])

    allOutputs = tf.compat.v1.nn.xw_plus_b(allLstmOutputs, w, b)

    splitOutputs = tf.split(allOutputs, numUnrollings, axis = 0)



if __name__ == "__main__":
    main()