import gym
import tf
import random as r

# This is the concrete implementation of Double DQN.

  # Class for storing and handling experiences.
  class ExpReplay:

    def __init__(self, capacity, max_batchSize):
      self.cap = capacity
      self.len = 0
      self.mem = [()]*capacity
      self.cntr = 0
      self.mxbtch = max_batchSize

    def capacity(self):
      return self.capacity

    def add(self, exp):
      self.mem.insert[self.cntr] = exp
      self.cntr = (self.cntr + 1) % capacity
      if self.len != self.capacity:
        self.len = self.len + 1

    # Returns a list of tuples. (experiences)
    def sample(self):
      batch = [0]*self.mxbtch
      for i in range(0, self.mxbtch):
        k = r.randint(0, self.len-1)
        batch[i] = self.mem[k]
      return batch

  # A class implements the epsilon greedy policy.
  class EpsGreedy:

    def __init__(self, init_exp, final_exp, fn_frame, actions):
      self.start = init_exp # The exploration at the beginning
      self.end = final_exp
      self.frame = fn_frame # The number of frames during the exploration is linearly annealed
      self.exp = self.start
      self.actions = actions

    def action(self, act):
      explore = (r.random() < self.exp)
      c_act = act
      if explore:
        k = r.randint(0, self.actions-2)
        if k >= act:
          c_act = k + 1
        else:
          c_act = k
      return c_act

    def anneal(self):
      self.exp = self.exp - (self.start - self.end)/self.frame
    
  # Class for DQN.

  class DQN:
    
    cntr_stepsSofar = 0
  
    gamma = 0.99
    alpha = 0.00025

    def __init__(self, actions):
      self.actions = actions
      self.erply = ExpReplay(1000000, 32)
      self.grdy = EpsGreedy(1.0, 0.1, 1000000, actions)
    
    # Functions to interact with the environment.

    def init(obs, action, rw, obs_nx):
      tp_exp = (obs, action, rw, obs, obs_nx)
      self.erply.add(tp_exp)

    def train(obs, action, rw, obs_nx):
      DQN.cntr_stepsSofar += 1
      self.erply.add((obs, action, rw, obs_nx))
      if DQN.cntr_stepsSofar % 4 == 0:
        batch = self.erply.sample()
        tr_batch = [()]*32
        for i in range(0, 32):
          tr_batch[i][0] = batch[i][0]
          tr_batch[i][1] = batch[i][1]
          a = tf.argmaxQ(batch[i][3])
          tr_batch[i][2] = batch[i][2] + gamma * tf.Q_frozen(batch[i][3], a)
        tf.train(tr_batch, alpha)
        if DQN.cntr_stepsSofar % 1000 == 0:
          tf.UpdateNetwork()

    def action(obs):
      a = tf.argmaxQ(obs)
      a = self.grdy.action(a)
      if cntr_stepsSofar < 1000000:
        self.grdy.anneal()
      return a

    def end():
      return cntr_stepsSofar > 50000000



