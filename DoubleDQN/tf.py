import random as r
r.seed(113)

# Base class for using the network from outside.
class IDnn:

    def get_action_number(self):
        raise NotImplementedError()

    def get_batch_size(self):
        raise NotImplementedError()

    def argmaxQ(self, state):
        raise NotImplementedError()
        
    def Q_frozen(self, state, action):
        raise NotImplementedError()
    
    def update_network(self):
        raise NotImplementedError()
        
    def train(self, mini_batch):
        raise NotImplementedError()

# The deep neural network to approximate the action-values.    

from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten
from keras.optimizers import RMSprop
    
class Dnn(IDnn):

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
      
      rmsprop = RMSprop(lr=alpha, epsilon=0.01, clipvalue=1.0, decay=0.01)
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

