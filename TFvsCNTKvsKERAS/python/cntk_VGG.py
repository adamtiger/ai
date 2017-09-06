from utils import *
import argparse
import numpy as np
import cntk as c

parser = argparse.ArgumentParser(description='CNTK VGG')
parser.add_argument('--num-runs', metavar='N', type=int, default=10)
parser.add_argument('--file-name', metavar='S', default='performance.txt')

args = parser.parse_args()

input_VGG_data_cntk = c.input_variable(shape=(3, 224, 224), dtype=np.float32)  # here the channel is the first


def create_VGG_model_CNTK():

    kernel1_1 = c.constant(value=np.float32(np.random.random((64, 3, 3, 3))))
    conv1_1 = c.relu(c.layers.convolution(kernel1_1, input_VGG_data_cntk, strides=(1, 1), auto_padding=[False, True, True]))
    kernel1_2 = c.constant(value=np.float32(np.random.random((64, 64, 3, 3))))
    conv1_2 = c.relu(c.layers.convolution(kernel1_2, conv1_1, strides=(1, 1), auto_padding=[False, True, True]))
    pool1 = c.layers.pooling(conv1_2, c.MAX_POOLING, (2, 2), (2, 2), auto_padding=[False, False, False])

    kernel2_1 = c.constant(value=np.float32(np.random.random((128, 64, 3, 3))))
    conv2_1 = c.relu(c.layers.convolution(kernel2_1, pool1, strides=(1, 1), auto_padding=[False, True, True]))
    kernel2_2 = c.constant(value=np.float32(np.random.random((128, 128, 3, 3))))
    conv2_2 = c.relu(c.layers.convolution(kernel2_2, conv2_1, strides=(1, 1), auto_padding=[False, True, True]))
    pool2 = c.layers.pooling(conv2_2, c.MAX_POOLING, (2, 2), (2, 2), auto_padding=[False, False, False])

    kernel3_1 = c.constant(value=np.float32(np.random.random((256, 128, 3, 3))))
    conv3_1 = c.relu(c.layers.convolution(kernel3_1, pool2, strides=(1, 1), auto_padding=[False, True, True]))
    kernel3_2 = c.constant(value=np.float32(np.random.random((256, 256, 3, 3))))
    conv3_2 = c.relu(c.layers.convolution(kernel3_2, conv3_1, strides=(1, 1), auto_padding=[False, True, True]))
    kernel3_3 = c.constant(value=np.float32(np.random.random((256, 256, 3, 3))))
    conv3_3 = c.relu(c.layers.convolution(kernel3_3, conv3_2, strides=(1, 1), auto_padding=[False, True, True]))
    kernel3_4 = c.constant(value=np.float32(np.random.random((256, 256, 3, 3))))
    conv3_4 = c.relu(c.layers.convolution(kernel3_4, conv3_3, strides=(1, 1), auto_padding=[False, True, True]))
    pool3 = c.layers.pooling(conv3_4, c.MAX_POOLING, (2, 2), (2, 2), auto_padding=[False, False, False])

    kernel4_1 = c.constant(value=np.float32(np.random.random((512, 256, 3, 3))))
    conv4_1 = c.relu(c.layers.convolution(kernel4_1, pool3, strides=(1, 1), auto_padding=[False, True, True]))
    kernel4_2 = c.constant(value=np.float32(np.random.random((512, 512, 3, 3))))
    conv4_2 = c.relu(c.layers.convolution(kernel4_2, conv4_1, strides=(1, 1), auto_padding=[False, True, True]))
    kernel4_3 = c.constant(value=np.float32(np.random.random((512, 512, 3, 3))))
    conv4_3 = c.relu(c.layers.convolution(kernel4_3, conv4_2, strides=(1, 1), auto_padding=[False, True, True]))
    kernel4_4 = c.constant(value=np.float32(np.random.random((512, 512, 3, 3))))
    conv4_4 = c.relu(c.layers.convolution(kernel4_4, conv4_3, strides=(1, 1), auto_padding=[False, True, True]))
    pool4 = c.layers.pooling(conv4_4, c.MAX_POOLING, (2, 2), (2, 2), auto_padding=[False, False, False])

    kernel5 = c.constant(value=np.float32(np.random.random((4096, 512, 7, 7))))
    fc1 = c.relu(c.layers.convolution(kernel5, pool4, strides=(1, 1), auto_padding=[False, False, False]))

    kernel6 = c.constant(value=np.float32(np.random.random((4096, 4096, 1, 1))))
    fc2 = c.relu(c.layers.convolution(kernel6, fc1, strides=(1, 1), auto_padding=[False, False, False]))

    kernel7 = c.constant(value=np.float32(np.random.random((1024, 4096, 1, 1))))
    fc3 = c.softmax(c.layers.convolution(kernel7, fc2, strides=(1, 1), auto_padding=[False, False, False]))

    return fc3
    
    
@profile
def execute_model_CNTK(model, input_data):
    model.eval(input_data)
    
# Execute a test

model = create_VGG_model_CNTK()
data_creator = generate_data_CNTK(input_VGG_data_cntk, [3, 224, 224])

for i in range(args.num_runs):
    if i % 100 == 0:
        print("Currently at: %d", i)
        
    data = data_creator()
        
    execute_model_CNTK(model, data)
    
print_data()
write_file("CNTK VGG test", args.file_name)
clear_data()
