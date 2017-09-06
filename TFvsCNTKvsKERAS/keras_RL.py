from utils import *
import argparse
import numpy as np

from keras.models import Sequential as Sequential_KERAS
from keras.layers import Conv2D, MaxPooling2D, LSTM

parser = argparse.ArgumentParser(description='KERAS RL')
parser.add_argument('--num-runs', metavar='N', type=int, default=100)
parser.add_argument('--file-name', metavar='S', default='performance.txt')

args = parser.parse_args()


def create_RL_model_KERAS():

    model = Sequential_KERAS()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='valid', activation='relu', input_shape=(84, 84, 4)))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Conv2D(512, (7, 7), padding='valid', activation='relu'))
    model.add(Conv2D(6, (1, 1), padding='valid', activation='relu'))

    model.compile(optimizer='rmsprop', loss='mse')
    return model
    
@profile
def execute_model_KERAS(model, input_data):

    model.predict(input_data, batch_size=1)
    
# Execute a test

model = create_RL_model_KERAS()
data_creator = generate_data_KERAS([1, 84, 84, 4])

for i in range(args.num_runs):
    if i % 100 == 0:
        print("Currently at: %d", i)
        
    data = data_creator()
        
    execute_model_KERAS(model, data)
    
print_data()
write_file("KERAS RL test", args.file_name)
clear_data()