from utils import *
import argparse
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(description='TF RL')
parser.add_argument('--num-runs', metavar='N', type=int, default=100)
parser.add_argument('--file-name', metavar='S', default='performance.txt')

args = parser.parse_args()

input_RL_data_tf = tf.placeholder(tf.float32, shape=(None, 84, 84, 4))


def create_RL_model_TF():

    def create_variable(kernel_shape):
        init = tf.truncated_normal(kernel_shape, stddev=0.1)
        return tf.Variable(init)

    def create_bias(filter_num):
        init = tf.constant(0.1, shape=[filter_num])
        return tf.Variable(init)

    w1 = create_variable([8, 8, 4, 32])
    b1 = create_bias(32)
    conv1 = tf.nn.relu(tf.nn.conv2d(input_RL_data_tf, w1, padding='VALID', strides=[1, 4, 4, 1]) + b1)
    w2 = create_variable([4, 4, 32, 64])
    b2 = create_bias(64)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, w2, padding='VALID', strides=[1, 2, 2, 1]) + b2)
    w3 = create_variable([3, 3, 64, 64])
    b3 = create_bias(64)
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, w3, padding='VALID', strides=[1, 1, 1, 1]) + b3)

    w4 = create_variable([7, 7, 64, 512])
    b4 = create_bias(512)
    fc1 = tf.nn.relu(tf.nn.conv2d(conv3, w4, padding='VALID', strides=[1, 1, 1, 1]) + b4)
    w5 = create_variable([1, 1, 512, 6])
    b5 = create_bias(6)
    fc2 = tf.nn.conv2d(fc1, w5, padding='VALID', strides=[1, 1, 1, 1]) + b5

    return fc2
    
sess = tf.Session()
  
@profile
def execute_model_TF(model, input_data):
    sess.run(model, input_data)
    
# Execute a test

model = create_RL_model_TF()
data_creator = generate_data_TF(input_RL_data_tf, [1, 84, 84, 4])
sess.run(tf.global_variables_initializer())

for i in range(args.num_runs):
    if i % 100 == 0:
        print("Currently at: %d", i)
        
    data = data_creator()
        
    execute_model_TF(model, data)
    
print_data()
write_file("TF RL test", args.file_name)
clear_data()