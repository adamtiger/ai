from utils import *
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description='TF VGG')
parser.add_argument('--num-runs', metavar='N', type=int, default=10)
parser.add_argument('--file-name', metavar='S', default='performance.txt')

args = parser.parse_args()

input_VGG_data_tf = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))


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
    
sess = tf.Session()
  
@profile
def execute_model_TF(model, input_data):
    sess.run(model, input_data)
    
# Execute a test

model = create_VGG_model_TF()
data_creator = generate_data_TF(input_VGG_data_tf, [1, 224, 224, 3])
sess.run(tf.global_variables_initializer())

for i in range(args.num_runs):
    if i % 100 == 0:
        print("Currently at: %d", i)
        
    data = data_creator()
        
    execute_model_TF(model, data)
    
print_data()
write_file("TF VGG test", args.file_name)
clear_data()