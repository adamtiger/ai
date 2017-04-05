# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 23:22:19 2017

@author: Adam Budai
"""

import random as r
import numpy as np
from BaseAlg import BaseAlgorithm

from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten
from keras.optimizers import RMSprop

r.seed(133)

# This is the concrete implementation of Double DQN.

# Class for storing and handling experiences.
class ExpReplay:

    def __init__(self, capacity, max_batchSize):
        self.cap = capacity
        self.length = 0
        self.mem = [()]*capacity
        self.cntr = 0
        self.mxbtch = max_batchSize

    def capacity(self):
        return self.cap

    def add(self, exp):
        self.mem[self.cntr] = exp
        self.cntr = (self.cntr + 1) % self.cap
        if self.length != self.cap:
            self.length = self.length + 1

    # Returns a list of tuples. (experiences)
    def sample(self):
        batch = [()]*self.mxbtch
        for i in range(0, self.mxbtch):
            k = r.randint(0, self.length-1)
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
        self.no_op = 0

    def action(self, act):
        explore = (r.random() < self.exp)
        c_act = act
        if act == 0:
            self.no_op += 1
        if explore and self.no_op < 40:
            k = r.randint(0, self.actions-2)
            if k >= act:
              c_act = k + 1
            else:
              c_act = k
        elif explore:
            c_act = r.randint(1, self.actions-2) # do not generate no_op (act = 0)
            self.no_op = 0
        return c_act

    def anneal(self):
        self.exp = self.exp - (self.start - self.end)/self.frame


class Dnn:

    def __create_model(self, actions, alpha):
      model = Sequential()
      model.add(Convolution2D(32, 8, 8, border_mode='valid', input_shape=(84, 84, 4), subsample=(4, 4)))
      model.add(Activation('relu'))
      model.add(Convolution2D(64, 4, 4, border_mode='valid', subsample=(2, 2)))
      model.add(Activation('relu'))
      model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1)))
      model.add(Activation('relu'))
      model.add(Flatten())
      model.add(Dense(512))
      model.add(Activation('relu'))
      model.add(Dense(actions))
      
      rmsprop = RMSprop(lr=alpha)
      model.compile(optimizer=rmsprop, loss='mse')
      
      return model

    def __init__(self, actions, batch_size, alpha):
        self.actions = actions
        self.batch_size = batch_size
        self.alpha = alpha
        self.Q = self.__create_model(actions, alpha)
        self.Q_ = self.__create_model(actions, alpha)

    def get_action_number(self):
        return self.actions

    def get_batch_size(self):
        return self.batch_size

    def argmaxQ(self, state):
        return self.Q.predict(state, batch_size=1).argmax()
        
    def Q_frozen(self, state, action):
        return self.Q_.predict(state, batch_size=1)[0, action]
    
    def update_network(self):
        self.Q_.set_weights(self.Q.get_weights())
        
    def train(self, mini_batch):
        target = self.Q.predict(mini_batch[0], batch_size=self.batch_size)
        target[:, mini_batch[1]] = mini_batch[2]
        self.Q.fit(mini_batch[0], target, nb_epoch=1, batch_size=self.batch_size, verbose=0)
        
    def save(self, fname):
        self.Q.save_weights(fname)
        
    def load(self, fname):
        self.Q.load_weights(fname, by_name = False)

        
# Class for implementing Double deep Q-network.
class DQN:

    def __init__(self):
        self._cntr = 0

    def set_params(self, actions, alpha, C, max_iter, mem_size, exp_start, exp_end, last_fm, gamma):
        self._actions = actions
        self._tf = Dnn(actions, 32, alpha)
        self.C = C
        self.max_iter = max_iter
        self.batch_size = self._tf.get_batch_size()
        self.last_fm = last_fm
        self.erply = ExpReplay(mem_size, self.batch_size)
        self.grdy = EpsGreedy(exp_start, exp_end, last_fm, self._actions)
        self._gamma = gamma
    
    # Functions to interact with the environment.

    def init(self, obs, action, rw, obs_nx):
        tp_exp = (obs, action, rw, obs_nx)
        self.erply.add(tp_exp)
        self.no_op = 0

    def train(self, obs, action, rw, obs_nx):
        self._cntr += 1
        self.erply.add((obs, action, rw, obs_nx))
        if self._cntr % 4 == 0:
            batch = self.erply.sample()
            y = np.zeros((self.batch_size))
            actions = np.zeros((self.batch_size))
            states = np.zeros((self.batch_size, 84,84,4))
            for i in range(0, self.batch_size):
                a = self._tf.argmaxQ(batch[i][3])
                y[i] = batch[i][2] + self._gamma * self._tf.Q_frozen(batch[i][3], a)
                actions[i] = batch[i][1]
                states[i] = batch[i][0]
            actions_int = actions.astype(int)
            tr_batch = [states[:], actions_int[:], y[:]]
            self._tf.train(tr_batch)
            if self._cntr % self.C == 0:
                self._tf.update_network()
        if self._cntr <= self.last_fm:
            self.grdy.anneal()

    def action(self, obs):
        a = self._tf.argmaxQ(obs)
        a = self.grdy.action(a)
        return a
        
    def action_nogreedy(self, obs):
        a = self._tf.argmaxQ(obs)
        if a == 0:
            self.no_op += 1
        if self.no_op % 30 == 0:
            a = r.randint(1, self._actions-1)
        return a

    def end(self):
        return self._cntr > self.max_iter
        
    def save(self, fname):
        self._tf.save(fname)
        
class DqnAgent(BaseAlgorithm):

  def __init__(self, dqn):
    self.DQN = dqn
    self.action = 0
    self.rw = 0
    self.obs_old = []
  

  def init(self, obs1, action1, rw1, obs2):
    self.DQN.init(obs1, action1, rw1, obs2)
    
  def next_action_train(self, obs, rw):
    if len(self.obs_old) != 0: 
      self.DQN.train(self.obs_old, self.action, self.rw, obs)
    self.obs_old = obs
    self.rw = rw
    self.action = self.DQN.action(obs)
    return self.action
    
  def next_action(self, obs):
    return self.DQN.action_nogreedy(obs)
    
  def is_training_finished(self):
    return self.DQN.end()
    
  def save_agent(self, fname):
    self.DQN.save(fname)
