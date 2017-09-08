from utils import *
import argparse
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

parser = argparse.ArgumentParser(description='KERAS VGG')
parser.add_argument('--num-runs', metavar='N', type=int, default=10)
parser.add_argument('--file-name', metavar='S', default='performance.txt')

args = parser.parse_args()

keras_backend(K, args.file_name)


def create_VGG_model_KERAS():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(224, 224, 3)))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Conv2D(4096, (7, 7), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Conv2D(4096, (1, 1), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Conv2D(1024, (1, 1), strides=(1, 1), padding='valid', activation='softmax'))

    model.compile(optimizer='rmsprop', loss='mse')
    return model
    
@profile
def execute_model_KERAS(model, input_data):

    model.predict(input_data, batch_size=1)
    
# Execute a test

model = create_VGG_model_KERAS()
data_creator = generate_data_KERAS([1, 224, 224, 3])

for i in range(args.num_runs):
    if i % 100 == 0:
        print("Currently at: %d", i)
        
    data = data_creator()
        
    execute_model_KERAS(model, data)
    
print_data()
write_file("KERAS VGG test", args.file_name)
clear_data()
