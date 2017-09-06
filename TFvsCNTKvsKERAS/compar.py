import time
import numpy as np

from keras.models import Sequential as Sequential_KERAS
from keras.layers import Conv2D, MaxPooling2D, LSTM

import tensorflow as tf

import cntk as c


# Profiler decorator

PROF_DATA = {}


def profile(fn):
    def with_profiler(*args, **kwargs):

        start_time = time.time()

        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time

        if fn.__name__ not in PROF_DATA:
            PROF_DATA[fn.__name__] = [1, [elapsed_time]]
        else:
            PROF_DATA[fn.__name__][0] += 1
            PROF_DATA[fn.__name__][1].append(elapsed_time)

        return ret
    return with_profiler


def print_data():
    for fn_name, data in PROF_DATA.items():
        max_time = max(data[1])
        max_place = data[1].index(max_time)
        avg_time = (sum(data[1]) - data[1][0])/(data[0] - 1)
        print("Max execution time for function %s was %0.3f s at %d." % (fn_name, max_time, max_place))
        print("Average execution time: %0.5f s." % avg_time)


def write_file(file_name):
    with open(file_name, 'w') as f:
        for fn_name, data in PROF_DATA.items():
            max_time = max(data[1])
            max_place = data[1].index(max_time)
            avg_time = (sum(data[1]) - data[1][0]) / (data[0] - 1)
            f.write("Max execution time for function %s was %0.3f s at %d.\n" % (fn_name, max_time, max_place))
            f.write("Average execution time: %0.5f s.\n" % avg_time)


def print_data2():
    for fn_name, data in PROF_DATA.items():
        print(fn_name)
        for d in data[1]:
            print(d)
        print("------")


def clear_data():
    global PROF_DATA
    PROF_DATA = {}


def generate_random_data(input_size):  # input_size = [1, h, w, ch]
    return np.float32(np.random.random(input_size) * 255.0)

# Creating the models
# 1) Keras version


def create_VGG_model_KERAS():
    model = Sequential_KERAS()
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


def create_RL_model_KERAS():

    model = Sequential_KERAS()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='valid', activation='relu', input_shape=(84, 84, 4)))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Conv2D(512, (7, 7), padding='valid', activation='relu'))
    model.add(Conv2D(6, (1, 1), padding='valid', activation='relu'))

    model.compile(optimizer='rmsprop', loss='mse')
    return model


def create_RNN_model_KERAS():

    model = Sequential_KERAS()
    model.add(LSTM(32, batch_input_shape=(1, 10, 64), return_sequences=True))
    model.add(LSTM(16, return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(16, return_sequences=True))
    model.add(LSTM(1, return_sequences=True))

    model.compile(optimizer='rmsprop', loss='mse')
    return model


@profile
def execute_model_KERAS(model, input_data):

    model.predict(input_data, batch_size=1)

# 2) TF version

input_VGG_data_tf = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
input_RL_data_tf = tf.placeholder(tf.float32, shape=(None, 84, 84, 4))
input_RNN_data_tf = tf.placeholder(tf.float32, shape=(1, 10, 64))


def create_VGG_model_TF():

    def w(channels, kernels):
        init = tf.truncated_normal([3, 3, channels, kernels], stddev=0.1)
        return tf.Variable(init)

    def b(filter_num):
        init = tf.constant(0.1, shape=[filter_num])
        return tf.Variable(init)

    conv1_1 = tf.nn.relu(tf.nn.conv2d(input_VGG_data_tf, w(3, 64), padding='SAME', strides=[1, 1, 1, 1]) + b(64))
    conv1_2 = tf.nn.relu(tf.nn.conv2d(conv1_1, w(64, 64), padding='SAME', strides=[1, 1, 1, 1]) + b(64))
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], padding='VALID', strides=[1, 2, 2, 1])

    conv2_1 = tf.nn.relu(tf.nn.conv2d(pool1, w(64, 128), padding='SAME', strides=[1, 1, 1, 1]) + b(128))
    conv2_2 = tf.nn.relu(tf.nn.conv2d(conv2_1, w(128, 128), padding='SAME', strides=[1, 1, 1, 1]) + b(128))
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], padding='VALID', strides=[1, 2, 2, 1])

    conv3_1 = tf.nn.relu(tf.nn.conv2d(pool2, w(128, 256), padding='SAME', strides=[1, 1, 1, 1]) + b(256))
    conv3_2 = tf.nn.relu(tf.nn.conv2d(conv3_1, w(256, 256), padding='SAME', strides=[1, 1, 1, 1]) + b(256))
    conv3_3 = tf.nn.relu(tf.nn.conv2d(conv3_2, w(256, 256), padding='SAME', strides=[1, 1, 1, 1]) + b(256))
    conv3_4 = tf.nn.relu(tf.nn.conv2d(conv3_3, w(256, 256), padding='SAME', strides=[1, 1, 1, 1]) + b(256))
    pool3 = tf.nn.max_pool(conv3_4, ksize=[1, 2, 2, 1], padding='VALID', strides=[1, 2, 2, 1])

    conv4_1 = tf.nn.relu(tf.nn.conv2d(pool3, w(256, 512), padding='SAME', strides=[1, 1, 1, 1]) + b(512))
    conv4_2 = tf.nn.relu(tf.nn.conv2d(conv4_1, w(512, 512), padding='SAME', strides=[1, 1, 1, 1]) + b(512))
    conv4_3 = tf.nn.relu(tf.nn.conv2d(conv4_2, w(512, 512), padding='SAME', strides=[1, 1, 1, 1]) + b(512))
    conv4_4 = tf.nn.relu(tf.nn.conv2d(conv4_3, w(512, 512), padding='SAME', strides=[1, 1, 1, 1]) + b(512))
    pool4 = tf.nn.max_pool(conv4_4, ksize=[1, 2, 2, 1], padding='VALID', strides=[1, 2, 2, 1])

    var_fc1 = tf.Variable(tf.truncated_normal([7, 7, 512, 4096], stddev=0.1))
    fc1 = tf.nn.relu(tf.nn.conv2d(pool4, var_fc1, padding='VALID', strides=[1, 1, 1, 1]) + b(4096))
    var_fc2 = tf.Variable(tf.truncated_normal([1, 1, 4096, 4096], stddev=0.1))
    fc2 = tf.nn.relu(tf.nn.conv2d(fc1, var_fc2, padding='VALID', strides=[1, 1, 1, 1]) + b(4096))
    var_fc3 = tf.Variable(tf.truncated_normal([1, 1, 4096, 1024], stddev=0.1))
    fc3 = tf.nn.softmax(tf.nn.conv2d(fc2, var_fc3, padding='VALID', strides=[1, 1, 1, 1]) + b(1024), dim=-1)

    return fc3


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

# 3) CNTK version

input_VGG_data_cntk = c.input_variable(shape=(3, 224, 224), dtype=np.float32)  # here the channel is the first
input_RL_data_cntk = c.input_variable(shape=(4, 84, 84), dtype=np.float32)
input_RNN_data_cntk = c.sequence.input_variable(64)  # default is float32, the length of the series is dynamic


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


def generate_data_KERAS(shape):
    def generator():
        return generate_random_data(shape)

    return generator


def generate_data_TF(var, shape):
    def generator():
        return {var: generate_random_data(shape)}

    return generator


def generate_data_CNTK(var, shape):
    def generator():
        return {var: generate_random_data(shape)}

    return generator


def measure_a_model(num_of_runs, model_data):
    sess.run(tf.global_variables_initializer())
    for i in range(num_of_runs):
        if i % 100 == 0:
            print("Currently at: %d", i)

        execute_model_KERAS(model_data[0], model_data[1]())

    for i in range(num_of_runs):

        execute_model_TF(model_data[2], model_data[3]())

    for i in range(num_of_runs):

        execute_model_CNTK(model_data[4], model_data[5]())

    print("Finished.")

    print_data()
    write_file("performance.txt")
    # print_data2()
    clear_data()

# Measure the elapsed times

# VGG
VGG_model_KERAS = create_VGG_model_KERAS()
VGG_model_TF = create_VGG_model_TF()
VGG_model_CNTK = create_VGG_model_CNTK()

input_keras = generate_data_KERAS([1, 224, 224, 3])
input_tf = generate_data_TF(input_VGG_data_tf, [1, 224, 224, 3])
input_cntk = generate_data_CNTK(input_VGG_data_cntk, [3, 224, 224])

print("----- Measuring VGG model -----")
measure_a_model(10, [VGG_model_KERAS, input_keras, VGG_model_TF, input_tf, VGG_model_CNTK, input_cntk])

# RL model
RL_model_KERAS = create_RL_model_KERAS()
RL_model_TF = create_RL_model_TF()
RL_model_CNTK = create_RL_model_CNTK()

input_keras = generate_data_KERAS([1, 84, 84, 4])
input_tf = generate_data_TF(input_RL_data_tf, [1, 84, 84, 4])
input_cntk = generate_data_CNTK(input_RL_data_cntk, [4, 84, 84])

print("----- Measuring RL model -----")
measure_a_model(100, [RL_model_KERAS, input_keras, RL_model_TF, input_tf, RL_model_CNTK, input_cntk])

# RNN model
RNN_model_KERAS = create_RNN_model_KERAS()
RNN_model_TF = create_RNN_model_TF()
RNN_model_CNTK = create_RNN_model_CNTK()

input_keras = generate_data_KERAS([1, 10, 64])
input_tf = generate_data_TF(input_RNN_data_tf, [1, 10, 64])
input_cntk = generate_data_CNTK(input_RNN_data_cntk, [10, 64])

print("----- Measuring RNN model -----")
measure_a_model(100, [RNN_model_KERAS, input_keras, RNN_model_TF, input_tf, RNN_model_CNTK, input_cntk])

