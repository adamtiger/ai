from utils import *
import argparse
import numpy as np
import cntk as c

parser = argparse.ArgumentParser(description='CNTK RNN')
parser.add_argument('--num-runs', metavar='N', type=int, default=100)
parser.add_argument('--file-name', metavar='S', default='performance.txt')

args = parser.parse_args()

input_RNN_data_cntk = c.sequence.input_variable(64)  # default is float32, the length of the series is dynamic


def create_RNN_model_CNTK():

    m = c.layers.Recurrence(c.layers.LSTM(32))(input_RNN_data_cntk)
    m = c.layers.Recurrence(c.layers.LSTM(16))(m)
    m = c.layers.Recurrence(c.layers.LSTM(256))(m)
    m = c.layers.Recurrence(c.layers.LSTM(128))(m)
    m = c.layers.Recurrence(c.layers.LSTM(64))(m)
    m = c.layers.Recurrence(c.layers.LSTM(32))(m)
    m = c.layers.Recurrence(c.layers.LSTM(16))(m)
    m = c.layers.Recurrence(c.layers.LSTM(1))(m)

    return m
    
    
@profile
def execute_model_CNTK(model, input_data):
    model.eval(input_data)
    
# Execute a test

model = create_RNN_model_CNTK()
data_creator = generate_data_CNTK(input_RNN_data_cntk, [10, 64])

for i in range(args.num_runs):
    if i % 100 == 0:
        print("Currently at: %d", i)
        
    data = data_creator()
        
    execute_model_CNTK(model, data)
    
print_data()
write_file("CNTK RNN test", args.file_name)
clear_data()
    
