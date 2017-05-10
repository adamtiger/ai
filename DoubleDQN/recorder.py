import gym
from environment import Preprocessing as p
import agent
import tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten
from keras.optimizers import RMSprop
from gym import wrappers
import random as r

class Dnn_(tf.IDnn):
    
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
      
      rmsprop = RMSprop(lr=alpha)
      model.compile(optimizer=rmsprop, loss='mse')
      
      return model

    def __init__(self, actions):
        self.Q = self.__create_model(actions, 0.1)

    def argmaxQ(self, state):
        return self.Q.predict(state, batch_size=1).argmax()
        
    def load(self, fname):
        self.Q.load_weights(fname, by_name = False)

class DQN_:
    
    def __init__(self, network):
        self._tf = network
    
    def action_nogreedy(self, obs):
        state = self.__normalize_img(obs)
        a = self._tf.argmaxQ(state)
        rd_nm = r.randint(0,100)
        if (rd_nm < 2):
            a = r.randint(0, 5)        

        return a
    
    def load(self, fname):
        self._tf.load(fname)
        
    def __normalize_img(self, img):
        return img/255.0 - 0.5
    
    

def recording(atari_name, fname, target_dir):
    
    nn = Dnn_(6)
    dqn = DQN_(nn)
    ag = agent.Agent(dqn, atari_name)
    ag.loadAgent(fname)
    
    env = ag.makeEnvironment()
    p_obj = p(env)
    
    env = wrappers.Monitor(env, target_dir)
    state = env.reset()
    
    end = False
    while not end:
        act = ag.nextAction(p_obj.preprocessing(state))
        state, rw, done, inf = env.step(act)
        end = done
    

for i in range(0, 15):
    file_name = 'files/video' + str(i)
    recording('Breakout-v0', 'agent.hdf5', file_name)

