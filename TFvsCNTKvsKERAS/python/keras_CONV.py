from utils import *
import argparse
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D
from keras import backend as K

parser = argparse.ArgumentParser(description='KERAS CONV')
parser.add_argument('--num-runs', metavar='N', type=int, default=500)
parser.add_argument('--file-name', metavar='S', default='performance.txt')

args = parser.parse_args()

keras_backend(K, args.file_name)


def create_CONV_model_KERAS():
    model84x84 = Sequential()
    model84x84.add(Conv2D(128, (8, 8), strides=(1, 1), padding='valid', activation='relu', input_shape=(84, 84, 16)))
    model84x84.compile(optimizer='rmsprop', loss='mse')

    model168x168 = Sequential()
    model168x168.add(Conv2D(128, (8, 8), strides=(1, 1), padding='valid', activation='relu', input_shape=(168, 168, 16)))
    model168x168.compile(optimizer='rmsprop', loss='mse')

    model336x336 = Sequential()
    model336x336.add(Conv2D(128, (8, 8), strides=(1, 1), padding='valid', activation='relu', input_shape=(336, 336, 16)))
    model336x336.compile(optimizer='rmsprop', loss='mse')

    model772x772 = Sequential()
    model772x772.add(Conv2D(128, (8, 8), strides=(1, 1), padding='valid', activation='relu', input_shape=(772, 772, 16)))
    model772x772.compile(optimizer='rmsprop', loss='mse')

    model1544x1544 = Sequential()
    model1544x1544.add(Conv2D(128, (8, 8), strides=(1, 1), padding='valid', activation='relu', input_shape=(1544, 1544, 16)))
    model1544x1544.compile(optimizer='rmsprop', loss='mse')

    return [model84x84, model168x168, model336x336, model772x772, model1544x1544]


@profile
def execute_model_KERAS(model, input_data):
    model.predict(input_data, batch_size=1)


# Execute a test

model = create_CONV_model_KERAS()
data_creator_84x84 = generate_data_KERAS([1, 84, 84, 16])
data_creator_168x168 = generate_data_KERAS([1, 168, 168, 16])
data_creator_336x336 = generate_data_KERAS([1, 336, 336, 16])
data_creator_772x772 = generate_data_KERAS([1, 772, 772, 16])
data_creator_1544x1544 = generate_data_KERAS([1, 1544, 1544, 16])

for i in range(args.num_runs):
    if i % 100 == 0:
        print("Currently at: %d", i)

    data = data_creator_84x84()

    execute_model_KERAS(model[0], data)

print_data()

with open(args.file_name, "a") as f:
    f.write("---- KERAS CONV test - 84x84 ----\n")

write_file("KERAS CONV test", args.file_name)
clear_data()


for i in range(args.num_runs):
    if i % 100 == 0:
        print("Currently at: %d", i)

    data = data_creator_168x168()

    execute_model_KERAS(model[1], data)

print_data()

with open(args.file_name, "a") as f:
    f.write("---- KERAS CONV test - 168x168 ----- \n")

write_file("KERAS CONV test", args.file_name)
clear_data()


for i in range(args.num_runs):
    if i % 100 == 0:
        print("Currently at: %d", i)

    data = data_creator_336x336()

    execute_model_KERAS(model[2], data)

print_data()

with open(args.file_name, "a") as f:
    f.write("---- KERAS CONV test - 336x336 ----- \n")

write_file("KERAS CONV test", args.file_name)
clear_data()


for i in range(args.num_runs):
    if i % 100 == 0:
        print("Currently at: %d", i)

    data = data_creator_772x772()

    execute_model_KERAS(model[3], data)

print_data()

with open(args.file_name, "a") as f:
    f.write("---- KERAS CONV test - 772x772 ----- \n")

write_file("KERAS CONV test", args.file_name)
clear_data()


for i in range(args.num_runs):
    if i % 100 == 0:
        print("Currently at: %d", i)

    data = data_creator_1544x1544()

    execute_model_KERAS(model[4], data)

print_data()

with open(args.file_name, "a") as f:
    f.write("---- KERAS CONV test - 1544x1544 ----- \n")

write_file("KERAS CONV test", args.file_name)
clear_data()