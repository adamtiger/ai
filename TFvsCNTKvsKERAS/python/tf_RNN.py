from utils import *
import argparse
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(description='TF RNN')
parser.add_argument('--num-runs', metavar='N', type=int, default=100)
parser.add_argument('--file-name', metavar='S', default='performance.txt')

args = parser.parse_args()

input_RNN_data_tf = tf.placeholder(tf.float32, shape=(1, 10, 64))


def create_RNN_model_TF():

    def lstm(units):
        return tf.nn.rnn_cell.LSTMCell(units)  # default activation: tanh

    lstm_sizes = [32, 16, 256, 128, 64, 32, 16, 1]

    network = tf.nn.rnn_cell.MultiRNNCell([lstm(lstm_sizes[i]) for i in range(len(lstm_sizes))])

    state = network.zero_state(1, tf.float32)
    for t in range(10):
        _, state = network(input_RNN_data_tf[:, t, :], state)

    return state
    
sess = tf.Session()
  
@profile
def execute_model_TF(model, input_data):
    sess.run(model, input_data)
    
# Execute a test

model = create_RNN_model_TF()
data_creator = generate_data_TF(input_RNN_data_tf, [1, 10, 64])
sess.run(tf.global_variables_initializer())

for i in range(args.num_runs):
    if i % 100 == 0:
        print("Currently at: %d", i)
        
    data = data_creator()
        
    execute_model_TF(model, data)
    
print_data()
write_file("TF RNN test", args.file_name)
clear_data()
