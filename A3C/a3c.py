# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 22:00:02 2017

@author: Adam Budai
"""
import threading

import preproc
import nn
import logger

import gym


class A3C(threading.Thread):
    
    def __init__(self, atari_env_name, params):
        threading.Thread.__init__(self)
        self.environment = gym.make(atari_env_name)
        self.params = params
        self.T = IterationNum(self.params.max_it(), threading.Lock())
        self.t = 1
        self.neural = nn.create_nn()
        
        
    def run(self):
        gradients = self.neural.create_gradientVec()
        t_start = 0
        while self.T.should_continue:
            self.neural.to_zero(gradients)
            self.neural.synchronize(self.params.shared())
            t_start = self.t
            

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
        self.lock.release()
    
class Parameters:
    
    def __init__(self, max_it, shared):
        self.max_it = max_it
        self.shared = shared
        
        
    def max_it(self):
        return self.max_it
        
    def shared(self):
        return self.shared
    
    