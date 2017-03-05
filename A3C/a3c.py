# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 22:00:02 2017

@author: Adam Budai
"""
from preproc import PreProcessing 
import nn
import logger
 
import gym

def a3c_agent(args, shared_nn, T):
    local_nn = nn.create_nn(args)
    environment = gym.make(args.atari_env)
    preprocessor = PreProcessing(environment)
    t = 1
    t_start = 0
    done = False
    state = preprocessor(environment.step(0)) 
    while T.should_continue:
        local_nn.gradients_to_zero()
        local_nn.synchronize(shared_nn)
        t_start = t
        store_st_rw = [()]*args.t_max
        idx = 0
        while not done and t - t_start < args.t_max:
            state,reward, done =  _choose_action(state)
            store_st_rw[idx] = (state, reward)
            t += 1
            T.increment()
            idx += 1
        R = 0
        if not done:
            R = local_nn.V(state)
        idx = t - 1
        while t >= t_start:
            R = store_st_rw[idx-t_start][1] + args.gamma * R
            local_nn.accumulate_gradients(R,store_st_rw[idx-t_start][0])
            idx -= 1

        local_nn.async_update(shared_nn)

def _choose_action(state):
    pass

def create_shared_nn(args):
    return nn.create_nn(args)

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


