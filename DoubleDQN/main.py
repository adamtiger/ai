import gym
import numpy as np
import scipy as sc
import math as m
import logger

# This file is for running the different Atari
# environments. 3 of them are examined:
#  -> Freeway (a chicken goes through the road)
#  -> Breakout (like tetris)
#  -> Riverraid (an airplane flies above a river)
# This implementation can run all of them.

import BaseAgent

state = []*4 # list to store the most recent frames
log = logger.Logger()

def map2Y(img):
    """ Calculates the luminance from the input RGB picture.

    Args:
      img: a numpy array with shape (height, width, 3)
    Output:
      a numpy array with shape (height, width, 1)
    """
    np_img = np.array(img)
    shape = np_img.shape
    shape[2] = 1
    ou_img = np.zeros(shape)

    ou_img[:,:,1] = (2*np_img[:,:,0] + 5*np_img[:,:,1] + np_img[:,:2])/8.0
    return ou_img
    
def rescale(img):
    """ Rescale the image for size 84x84.

    Args:
      img: a numpy array with shape (height, width, 1)
    Output:
      a numpy array with shape (84, 84, 1)
    """

    return sc.misc.imresize(img, (84,84,1), interp='bilinear')
    
def init_state(env):
    """ At the very beginning the there was not
    appeared four consequtive frames yet.

    Gathers the first four frames during no-op.
    Creates the first state.

    Args:
      env: the environment of the game
    Output:
      nothing
    """
    
    for i in range(0, 4):
        img = env.step(0) # no-op
        imgY = map2Y(img)
        obs = rescale(imgY)
        state[i] = obs

def preprocessing(img):

    imgY = map2Y(img)
    obs = rescale(imgY)
    del state[0]
    state.append(obs)
    ou_state = np.zeros((84,84,3))
    ou_state[:,:,0] = state[0][:,:,0]
    ou_state[:,:,1] = state[1][:,:,0]
    ou_state[:,:,2] = state[2][:,:,0]
    return ou_state

def evaluate(os):
  
    env = os.makeEnvironment()

    episend = False
    obs = env.reset()
    fi = preprocessing(obs)
    action = os.nextAction(fi)

    while(not episend):
        obs, rw, done, inf = env.step(action)
        log.write(obs, rw, done)
        fi = preprocessing(obs)
        action = os.nextAction(fi)
        episend = done
  
    return 0

def train(os, fname):

    env = os.makeEnvironment()
    exit = False
    cntr = 0
    k_evaluate = 0
    
    init_state(env)
    
    # Fill up the experience replay memory with
    # experiences.
    action1 = env.action_states.sample()
    obs1, rw1, done, inf = env.step(action)
    fi1 = preprocessing(obs1)
    for i in range(1,50000):
        action2 = env.action_states.sample()
        obs2, rw2, done, inf = env.step(action)
        fi2 = preprocessing(obs2)
        os.init(fi1, action1, rw1, fi2)
        action1 = action2
        rw1 = rw2
        fi1 = fi2
        if done:
            env.reset()

    # Start learning.
    while(not exit):
        episend = False	
        obs = env.reset()
        fi = preprocessing(obs)
        action = os.nextActionAndTrain(fi, 0.0)
        while(not episend):
            obs, rw, done, inf = env.step(action)
            cntr += 1
            fi = preprocessing(obs)
            action = os.nextActionAndTrain(fi, rw)
            episend = done
        exit = os.isTrainingFinished()
        if (m.floor(cntr / 1000000)-k_evaluate) > 0.0001: # evaluate the performance of the agent
            k_evaluate += 1
            evaluate(os)
        
    os.saveAgent(fname)  
    
    return 0
