from utils import *
import argparse
import numpy as np
import cntk as c

parser = argparse.ArgumentParser(description='CNTK RL')
parser.add_argument('--num-runs', metavar='N', type=int, default=100)
parser.add_argument('--file-name', metavar='S', default='performance.txt')

args = parser.parse_args()

input_RL_data_cntk = c.input_variable(shape=(4, 84, 84), dtype=np.float32)


def create_RL_model_CNTK():

    filter = np.float32(np.random.random((32, 4, 8, 8)))
    kernel1 = c.constant(value=filter)
    conv1=(c.relu(c.layers.convolution(kernel1, input_RL_data_cntk, strides=(4, 4), auto_padding=[False, False, False])))

    filter = np.float32(np.random.random((64, 32, 4, 4)))
    kernel2 = c.constant(value=filter)
    conv2 = (c.relu(c.layers.convolution(kernel2, conv1, strides=(2, 2), auto_padding=[False, False, False])))

    filter = np.float32(np.random.random((64, 64, 3, 3)))
    kernel3 = c.constant(value=filter)
    conv3 = (c.relu(c.layers.convolution(kernel3, conv2, strides=(1, 1), auto_padding=[False, False, False])))

    filter = np.float32(np.random.random((512, 64, 7, 7)))
    kernel4 = c.constant(value=filter)
    fc1 = (c.relu(c.layers.convolution(kernel4, conv3, strides=(1, 1), auto_padding=[False, False, False])))

    filter = np.float32(np.random.random((6, 512, 1, 1)))
    kernel5 = c.constant(value=filter)
    fc2 = c.layers.convolution(kernel5, fc1, strides=(1, 1), auto_padding=[False, False, False])

    return fc2
    
    
@profile
def execute_model_CNTK(model, input_data):
    model.eval(input_data)
    
# Execute a test

model = create_RL_model_CNTK()
data_creator = generate_data_CNTK(input_RL_data_cntk, [4, 84, 84])

for i in range(args.num_runs):
    if i % 100 == 0:
        print("Currently at: %d", i)
        
    data = data_creator()
        
    execute_model_CNTK(model, data)
    
print_data()
write_file("CNTK RL test", args.file_name)
clear_data()
    
