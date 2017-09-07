from utils import *
import argparse

import tensorflow as tf

parser = argparse.ArgumentParser(description='TF CONV')
parser.add_argument('--num-runs', metavar='N', type=int, default=500)
parser.add_argument('--file-name', metavar='S', default='performance.txt')

args = parser.parse_args()


input_CONV_84x84 = tf.placeholder(tf.float32, shape=(None, 84, 84, 16))
input_CONV_168x168 = tf.placeholder(tf.float32, shape=(None, 168, 168, 16))
input_CONV_336x336 = tf.placeholder(tf.float32, shape=(None, 336, 336, 16))
input_CONV_772x772 = tf.placeholder(tf.float32, shape=(None, 772, 772, 16))
input_CONV_1544x1544 = tf.placeholder(tf.float32, shape=(None, 1544, 1544, 16))


def create_CONV_model_TF():
    
    def w():
        init = tf.truncated_normal([8, 8, 16, 128], stddev=0.1)
        return tf.Variable(init)

    def b():
        init = tf.constant(0.1, shape=[128])
        return tf.Variable(init)

    model84x84 = tf.nn.relu(tf.nn.conv2d(input_CONV_84x84, w(), padding='SAME', strides=[1, 1, 1, 1]) + b())

    model168x168 = tf.nn.relu(tf.nn.conv2d(input_CONV_168x168, w(), padding='SAME', strides=[1, 1, 1, 1]) + b())

    model336x336 = tf.nn.relu(tf.nn.conv2d(input_CONV_336x336, w(), padding='SAME', strides=[1, 1, 1, 1]) + b())

    model772x772 = tf.nn.relu(tf.nn.conv2d(input_CONV_772x772, w(), padding='SAME', strides=[1, 1, 1, 1]) + b())

    model1544x1544 = tf.nn.relu(tf.nn.conv2d(input_CONV_1544x1544, w(), padding='SAME', strides=[1, 1, 1, 1]) + b())
    
    return [model84x84, model168x168, model336x336, model772x772, model1544x1544]


@profile
def execute_model_TF(model, input_data):
    model.predict(input_data, batch_size=1)


# Execute a test

model = create_CONV_model_TF()
data_creator_84x84 = generate_data_TF(input_CONV_84x84, [1, 84, 84, 16])
data_creator_168x168 = generate_data_TF(input_CONV_168x168, [1, 168, 168, 16])
data_creator_336x336 = generate_data_TF(input_CONV_336x336, [1, 336, 336, 16])
data_creator_772x772 = generate_data_TF(input_CONV_772x772, [1, 772, 772, 16])
data_creator_1544x1544 = generate_data_TF(input_CONV_1544x1544, [1, 1544, 1544, 16])

for i in range(args.num_runs):
    if i % 100 == 0:
        print("Currently at: %d", i)

    data = data_creator_84x84()

    execute_model_TF(model[0], data)

print_data()

with open(args.file_name, "a") as f:
    f.write("---- TF CONV test - 84x84 ----\n")

write_file("TF CONV test", args.file_name)
clear_data()


for i in range(args.num_runs):
    if i % 100 == 0:
        print("Currently at: %d", i)

    data = data_creator_168x168()

    execute_model_TF(model[1], data)

print_data()

with open(args.file_name, "a") as f:
    f.write("---- TF CONV test - 168x168 ----- \n")

write_file("TF CONV test", args.file_name)
clear_data()


for i in range(args.num_runs):
    if i % 100 == 0:
        print("Currently at: %d", i)

    data = data_creator_336x336()

    execute_model_TF(model[2], data)

print_data()

with open(args.file_name, "a") as f:
    f.write("---- TF CONV test - 336x336 ----- \n")

write_file("TF CONV test", args.file_name)
clear_data()


for i in range(args.num_runs):
    if i % 100 == 0:
        print("Currently at: %d", i)

    data = data_creator_772x772()

    execute_model_TF(model[3], data)

print_data()

with open(args.file_name, "a") as f:
    f.write("---- TF CONV test - 772x772 ----- \n")

write_file("TF CONV test", args.file_name)
clear_data()


for i in range(args.num_runs):
    if i % 100 == 0:
        print("Currently at: %d", i)

    data = data_creator_1544x1544()

    execute_model_TF(model[4], data)

print_data()

with open(args.file_name, "a") as f:
    f.write("---- TF CONV test - 1544x1544 ----- \n")

write_file("TF CONV test", args.file_name)
clear_data()