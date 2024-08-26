from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
#import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler


#df = pd.read_csv('data/stock_data.csv')


df = pd.read_csv('sp500_stocks.csv', delimiter=',', usecols=['Date', 'Symbol','Open', 'High', 'Low', 'Close'])
df = df[df['Symbol'] == 'AAPL']
#print('Loaded data for HPQ from the folder')
#print(df)

high_prices = df.loc[:,'High'].to_numpy()
low_prices = df.loc[:,'Low'].to_numpy()
mid_prices = (high_prices+low_prices)/2.0
split = (len(mid_prices) // 2)
train_data = mid_prices[:split]
test_data = mid_prices[split:]
scaler = MinMaxScaler()
train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)
smoothing_window_size = 250

for di in range(0,1000,smoothing_window_size):
    scaler.fit(train_data[di:di+smoothing_window_size,:])
    train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])



train_data = train_data.reshape(-1)
test_data = scaler.transform(test_data).reshape(-1)
EMA = 0.0
gamma = 0.1
for ti in range(split):
  EMA = gamma*train_data[ti] + (1-gamma)*EMA
  train_data[ti] = EMA

all_mid_data = np.concatenate([train_data,test_data],axis=0)

window_size = 100  # Adjusted to a smaller window size
N = train_data.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []

class DataGeneratorSeq(object):
    def __init__(self,prices,batch_size,num_unroll):
        self._prices = prices
        self._prices_length = len(self._prices) - num_unroll
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._segments = self._prices_length //self._batch_size
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def next_batch(self):
        batch_data = np.zeros((self._batch_size),dtype=np.float32)
        batch_labels = np.zeros((self._batch_size),dtype=np.float32)

        for b in range(self._batch_size):
            if self._cursor[b]+1>=self._prices_length:
                #self._cursor[b] = b * self._segments
                self._cursor[b] = np.random.randint(0,(b+1)*self._segments)

            batch_data[b] = self._prices[self._cursor[b]]
            batch_labels[b]= self._prices[self._cursor[b]+np.random.randint(0,5)]

            self._cursor[b] = (self._cursor[b]+1)%self._prices_length

        return batch_data,batch_labels
    
    def unroll_batches(self):
        unroll_data,unroll_labels = [],[]
        init_data, init_label = None,None
        for ui in range(self._num_unroll):
            data, labels = self.next_batch()    

            unroll_data.append(data)
            unroll_labels.append(labels)

        return unroll_data, unroll_labels

dg = DataGeneratorSeq(train_data,500,5)
u_data, u_labels = dg.unroll_batches()

D = 1
num_unrollings = 5
batch_size = 500
num_nodes = [256,128,64]
n_layers = len(num_nodes)   
dropout_rate = 0.3

model = tf.keras.models.Sequential()

# Adding the first LSTM layer and Dropout regularization
model.add(tf.keras.layers.LSTM(num_nodes[0], return_sequences=True, input_shape=(num_unrollings, D)))
model.add(tf.keras.layers.Dropout(dropout_rate))

# Adding the second LSTM layer and Dropout regularization
model.add(tf.keras.layers.LSTM(num_nodes[1], return_sequences=True))
model.add(tf.keras.layers.Dropout(dropout_rate))

# Adding the third LSTM layer and Dropout regularization
model.add(tf.keras.layers.LSTM(num_nodes[2]))
model.add(tf.keras.layers.Dropout(dropout_rate))

# Adding the output layer
model.add(tf.keras.layers.Dense(1))  # Since it's a regression problem

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Summary of the model
model.summary()

# Convert the unrolled data into the appropriate format for training
X_train = np.stack(u_data, axis=1)
y_train = np.stack(u_labels, axis=1)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=batch_size)

# Unrolling the test data similar to how the training data was unrolled
test_data_gen = DataGeneratorSeq(test_data, batch_size, num_unrollings)
u_test_data, u_test_labels = test_data_gen.unroll_batches()

# Prepare the test dataf
X_test = np.stack(u_test_data, axis=1)
y_test = np.stack(u_test_labels, axis=1)

# Make predictions
predictions = model.predict(X_test)

# Plot the results
plt.figure(figsize = (18,9))
plt.plot(range(len(y_test)), y_test, color='blue', label='True')
plt.plot(range(len(predictions)), predictions, color='red', label='Predicted')
plt.title('Prediction vs True')
plt.legend()
plt.show()
