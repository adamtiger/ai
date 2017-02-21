# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 23:00:00 2017

@author: Adam Budai
"""

class BaseAlgorithm:
    
    def next_action_train(self, observation, reward):
        raise NotImplementedError()
        
    def next_action(self, observation):
        raise NotImplementedError()
        
    def is_training_finished(self):
        raise NotImplementedError()
    
    def save_agent(self, fname):
        raise NotImplementedError()
        
