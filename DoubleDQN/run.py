import environment
import tf
import dqn
import BaseAgent as ba

# RUN THE ALGORITHM

actions = 6
batch_size = 32
alpha = 0.00025
C = 1000
max_iter =200000#50000000
mem_size = 2000#1000000
exp_start = 1.0
exp_end = 0.1
last_fm = 200000#1000000
gamma = 0.99

tf_f = tf.Dnn(actions, batch_size, alpha)
dqn_f = dqn.DQN()
dqn_f.set_params(tf_f, C, max_iter, mem_size, exp_start, exp_end, last_fm, gamma)
os = ba.BrAgent(dqn_f)
fname = "agent"

environment.train(os, fname)
