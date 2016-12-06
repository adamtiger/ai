import tensorflow as tf
import random as r
r.seed(113)

# Base class for using the network from outside.
class IDnn:

    def get_action_number():
        raise NotImplementedError()

    def argmaxQ(self, state):
        raise NotImplementedError()
        
    def Q_frozen(self, state, action):
        raise NotImplementedError()
    
    def update_network(self):
        raise NotIMplementedError()
        
    def train(self, mini_batch):
        raise NotImplementedError()

# The deep neural network to approximate the action-values.        
class Dnn(IDnn):

    def __init__(self, actions, batch_size, alpha):
        self.actions = actions
        self.batch_size = batch_size
        self.alpha = alpha

    def get_action_number():
        return self.actions

    def argmaxQ(self, state):
        raise NotImplementedError()
        
    def Q_frozen(self, state, action):
        raise NotImplementedError()
    
    def update_network(self):
        raise NotIMplementedError()
        
    def train(self, mini_batch):
        raise NotImplementedError()