from utils import *
import argparse

import numpy as np
import cntk as c

parser = argparse.ArgumentParser(description='CNTK CONV')
parser.add_argument('--num-runs', metavar='N', type=int, default=500)
parser.add_argument('--file-name', metavar='S', default='performance.txt')

args = parser.parse_args()

input_CONV_84x84 = c.input_variable(dtype = np.float32, shape=(16, 84, 84))
input_CONV_168x168 = c.input_variable(dtype = np.float32, shape=(16, 168, 168))
input_CONV_336x336 = c.input_variable(dtype = np.float32, shape=(16, 336, 336))
input_CONV_772x772 = c.input_variable(dtype = np.float32, shape=(16, 772, 772))
input_CONV_1544x1544 = c.input_variable(dtype = np.float32, shape=(16, 1544, 1544))


def create_CONV_model_CNTK():
    filter = np.float32(np.random.random((128, 16, 8, 8)))
    kernel1 = c.constant(value=filter)
    model84x84 = (
    c.relu(c.layers.convolution(kernel1, input_CONV_84x84, strides=(1, 1), auto_padding=[False, False, False])))

    filter = np.float32(np.random.random((128, 16, 8, 8)))
    kernel2 = c.constant(value=filter)
    model168x168 = (
        c.relu(c.layers.convolution(kernel2, input_CONV_168x168, strides=(1, 1), auto_padding=[False, False, False])))

    filter = np.float32(np.random.random((128, 16, 8, 8)))
    kernel3 = c.constant(value=filter)
    model336x336 = (
        c.relu(c.layers.convolution(kernel3, input_CONV_336x336, strides=(1, 1), auto_padding=[False, False, False])))

    filter = np.float32(np.random.random((128, 16, 8, 8)))
    kernel4 = c.constant(value=filter)
    model772x772 = (
        c.relu(c.layers.convolution(kernel4, input_CONV_772x772, strides=(1, 1), auto_padding=[False, False, False])))

    filter = np.float32(np.random.random((128, 16, 8, 8)))
    kernel5 = c.constant(value=filter)
    model1544x1544 = (
        c.relu(c.layers.convolution(kernel5, input_CONV_1544x1544, strides=(1, 1), auto_padding=[False, False, False])))

    return [model84x84, model168x168, model336x336, model772x772, model1544x1544]


@profile
def execute_model_CNTK(model, input_data):
    model.eval(input_data)


# Execute a test

model = create_CONV_model_CNTK()
data_creator_84x84 = generate_data_CNTK(input_CONV_84x84, [16, 84, 84])
data_creator_168x168 = generate_data_CNTK(input_CONV_168x168, [16, 168, 168])
data_creator_336x336 = generate_data_CNTK(input_CONV_336x336, [16, 336, 336])
data_creator_772x772 = generate_data_CNTK(input_CONV_772x772, [16, 772, 772])
data_creator_1544x1544 = generate_data_CNTK(input_CONV_1544x1544, [16, 1544, 1544])

for i in range(args.num_runs):
    if i % 100 == 0:
        print("Currently at: %d", i)

    data = data_creator_84x84()

    execute_model_CNTK(model[0], data)

print_data()

with open(args.file_name, "a") as f:
    f.write("---- CNTK CONV test - 84x84 ----\n")

write_file("CNTK CONV test", args.file_name)
clear_data()

for i in range(args.num_runs):
    if i % 100 == 0:
        print("Currently at: %d", i)

    data = data_creator_168x168()

    execute_model_CNTK(model[1], data)

print_data()

with open(args.file_name, "a") as f:
    f.write("---- CNTK CONV test - 168x168 ----- \n")

write_file("CNTK CONV test", args.file_name)
clear_data()

for i in range(args.num_runs):
    if i % 100 == 0:
        print("Currently at: %d", i)

    data = data_creator_336x336()

    execute_model_CNTK(model[2], data)

print_data()

with open(args.file_name, "a") as f:
    f.write("---- CNTK CONV test - 336x336 ----- \n")

write_file("CNTK CONV test", args.file_name)
clear_data()

for i in range(args.num_runs):
    if i % 100 == 0:
        print("Currently at: %d", i)

    data = data_creator_772x772()

    execute_model_CNTK(model[3], data)

print_data()

with open(args.file_name, "a") as f:
    f.write("---- CNTK CONV test - 772x772 ----- \n")

write_file("CNTK CONV test", args.file_name)
clear_data()

for i in range(args.num_runs):
    if i % 100 == 0:
        print("Currently at: %d", i)

    data = data_creator_1544x1544()

    execute_model_CNTK(model[4], data)

print_data()

with open(args.file_name, "a") as f:
    f.write("---- CNTK CONV test - 1544x1544 ----- \n")

write_file("CNTK CONV test", args.file_name)
clear_data()