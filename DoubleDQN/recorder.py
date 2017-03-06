import gym
from environment import Preprocessing as p
import agent
import tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten
from keras.optimizers import RMSprop
from gym import wrappers

class Dnn_(tf.IDnn):
    
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

    def __init__(self, actions):
        self.Q = self.__create_model(actions, 0.1)
        self.Q_ = self.__create_model(actions, 0.1)

    def argmaxQ(self, state):
        return self.Q.predict(state, batch_size=1).argmax()
        
    def load(self, fname):
        self.Q.load_weights(fname, by_name = False)

class DQN_:
    
    def __init__(self, network):
        self._tf = network
    
    def action(self, obs):
        a = self._tf.argmaxQ(obs)
        return a
    
    def load(self, fname):
        self._tf.load(fname)
    
    

def recording(atari_name, fname, target_dir):
    
    nn = Dnn_(6)
    dqn = DQN_(nn)
    ag = agent.Agent(dqn)
    ag.loadAgent(fname)
    
    env = gym.make(atari_name)
    p_obj = p.Preprocessing(env)
    state = env.reset()
    
    env = wrappers.Monitor(env, target_dir)
    
    end = False
    while not end:
        act = ag.nextAction(p_obj.preprocessing(state))
        state, rw, done, inf = env.step(act)
        end = done
    



