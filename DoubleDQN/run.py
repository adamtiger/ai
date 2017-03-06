import environment
import tf
import dqn
import agent

# RUN THE ALGORITHM

actions = 6
batch_size = 32
alpha = 0.00025

C = 5000
max_iter =5000000#10000000
mem_size = 50000#1000000
exp_start = 1.0
exp_end = 0.1
last_fm = 1000000
gamma = 0.99

evaluation_freq =500000
evaluation_number = 10
init_number_in_replay_mem = 5000

tf_f = tf.Dnn(actions, batch_size, alpha)
dqn_f = dqn.DQN()
dqn_f.set_params(tf_f, C, max_iter, mem_size, exp_start, exp_end, last_fm, gamma)
ag = agent.Agent(dqn_f)

env = environment.Environment(ag, evaluation_freq, evaluation_number, 
                              init_number_in_replay_mem)

fname = "files/agent.hdf5"
env.train(fname)
