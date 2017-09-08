from utils import *
import argparse
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM
from keras import backend as K

parser = argparse.ArgumentParser(description='KERAS RNN')
parser.add_argument('--num-runs', metavar='N', type=int, default=100)
parser.add_argument('--file-name', metavar='S', default='performance.txt')

args = parser.parse_args()

keras_backend(K, args.file_name)


def create_RNN_model_KERAS():

    model = Sequential()
    model.add(LSTM(32, batch_input_shape=(1, 10, 64), return_sequences=True))
    model.add(LSTM(16, return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(16, return_sequences=True))
    model.add(LSTM(1, return_sequences=True))

    model.compile(optimizer='rmsprop', loss='mse')
    return model
    
@profile
def execute_model_KERAS(model, input_data):

    model.predict(input_data, batch_size=1)
    
# Execute a test

model = create_RNN_model_KERAS()
data_creator = generate_data_KERAS([1, 10, 64])

for i in range(args.num_runs):
    if i % 100 == 0:
        print("Currently at: %d", i)
        
    data = data_creator()
        
    execute_model_KERAS(model, data)
    
print_data()
write_file("KERAS RNN test", args.file_name)
clear_data()