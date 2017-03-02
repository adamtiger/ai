# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 22:00:02 2017

@author: Adam Budai
"""
import threading

from preproc import PreProcessing 
import nn
import logger

import gym


class A3C(threading.Thread):

    def __init__(self, atari_env_name, params):

        threading.Thread.__init__(self)
        self.environment = gym.make(atari_env_name)
        self.params = params
        self.T = IterationNum(self.params.max_it(), threading.Lock())
        self.preprocessor = PreProcessing(self.environment)
        self.t = 1
        self.neural = nn.create_nn()


    def run(self):
        gradients = self.neural.create_gradient_vec()
        t_start = 0
        state = self.preprocessor(self.environment.step(0)) 
        while self.T.should_continue:

            self.neural.to_zero(gradients)
            self.neural.synchronize(self.params.shared())
            t_start = self.t
            store_st_rw = [()]*self.params.t_max()
            idx = 0
            while not done and self.t - t_start < self.params.t_max():
               state,reward, done =  self._choose_action(state)
               store_st_rw[idx] = (state, reward)
               self.t += 1
               self.T.increment()
               idx += 1
            R = 0
            if not done:
                R = self.neural.V(state)
            idx = self.t - 1
            while self.t >= t_start:
                R = store_st_rw[idx-t_start][1] + self.params.gamma() * R
                self.neural.accumulate_gradients(R,store_st_rw[idx-t_start][0])
                idx -= 1

            self.neural.async_update(gradients)

    def _choose_action(self, state):
        pass


class IterationNum:

    def __init__(self, max_it, lock):


        self.max_it
        self.it = 0
        self.lock = lock

    def should_continue(self):
        return self.it < self.max_it

    def increment(self):
        self.lock.acquire()
        self.it += 1
        df


        self.lock.release()


