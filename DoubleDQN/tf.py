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
from keras.layers import Dense, Activation, Conv2D, Flatten
from keras.optimizers import RMSprop
    
class Dnn(IDnn):

    def __create_model(self, actions, alpha):
      model = Sequential()
      model.add(Conv2D(32, (8, 8), padding='valid', input_shape=(84, 84, 4), strides=(4, 4)))
      model.add(Activation('relu'))
      model.add(Conv2D(64, (4, 4), padding='valid', strides=(2, 2)))
      model.add(Activation('relu'))
      model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1)))
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
        return self.Q_.predict(state, batch_size=1).argmax() # This should be Q but try this out.
        
    def Q_frozen(self, state, action):
        return self.Q_.predict(state, batch_size=1)[0, action]
    
    def update_network(self):
        self.Q_.set_weights(self.Q.get_weights())
        
    def train(self, mini_batch):
        target = self.Q.predict(mini_batch[0], batch_size=self.batch_size)
        for i in range(0, self.batch_size):
            target[i, mini_batch[1][i]] = mini_batch[2][i]
        self.Q.train_on_batch(mini_batch[0], target)
        
    def save(self, fname):
        self.Q.save_weights(fname)

