import gym
import numpy as np
import scipy.misc as sc
import math as m
from gym import wrappers

# This file is for running the different Atari
# environments. 3 of them are examined:
#  -> Freeway (a chicken goes through the road)
#  -> Breakout (like tetris)
#  -> Riverraid (an airplane flies above a river)
# This implementation can run all of them.

import BaseAgent as ba
import logger
import dqn
import tf
#import test

# Global variables and constants:

state = [] # list to store the most recent frames
evaluation_freq =10000#1000000
evaluation_number = 10
log = logger.Logger(evaluation_number)
init_number_in_replay_mem = 1000#50000
evaluation_counter = 0

def map2Y(img):
    """ Calculates the luminance from the input RGB picture.

    Args:
      img: a numpy array with shape (height, width, 3)
    Output:
      a numpy array with shape (height, width, 1)
    """
    np_img = np.array(img)
    shape = np_img.shape
    shape_new = (shape[0], shape[1], 1)
    ou_img = np.zeros(shape_new)

    ou_img[:,:,0] = (2*np_img[:,:,0] + 5*np_img[:,:,1] + np_img[:,:,2])/8.0
    return ou_img
    
def rescale(img):
    """ Rescale the image for size 84x84.

    Args:
      img: a numpy array with shape (height, width, 1)
    Output:
      a numpy array with shape (84, 84, 1)
    """
    shape = img.shape
    img_ = np.zeros((shape[0], shape[1]))
    img_[:,:] = img[:,:,0]
    
    img_resized = sc.imresize(img_, (84,84), interp='bilinear', mode=None)
    
    img_ou = np.zeros((84,84,1))
    img_ou[:,:,0] = img_resized[:,:]
    
    return img_ou
    
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
        img, rw, done, inf = env.step(0) # no-op
        imgY = map2Y(img)
        obs = rescale(imgY)
        state.append(obs)

def preprocessing(img):
    imgY = map2Y(img)
    obs = rescale(imgY)
    del state[0]
    state.append(obs)
    ou_state = np.zeros((1,84,84,4))
    ou_state[0,:,:,0] = state[0][:,:,0]
    ou_state[0,:,:,1] = state[1][:,:,0]
    ou_state[0,:,:,2] = state[2][:,:,0]
    ou_state[0,:,:,3] = state[3][:,:,0]
    return ou_state

def evaluate(os):
    
    global evaluation_counter
    evaluation_counter += 1
  
    env = os.makeEnvironment()
    file_name = 'files/videos-' + str(evaluation_counter)
    env = wrappers.Monitor(env, file_name)
    
    for i in range(0,evaluation_number):
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
    
    print "Start training."
    
    env = os.makeEnvironment()
    exit = False
    cntr = 0
    k_evaluate = 0
    
    init_state(env)
    
    print "State was initialized."
    
    # Fill up the experience replay memory with
    # experiences.
    action1 = env.action_space.sample()
    obs1, rw1, done, inf = env.step(action1)
    fi1 = preprocessing(obs1)
    for i in range(1,init_number_in_replay_mem):
        action2 = env.action_space.sample()
        obs2, rw2, done, inf = env.step(action2)
        fi2 = preprocessing(obs2)
        os.init(fi1, action1, rw1, fi2)
        action1 = action2
        rw1 = rw2
        fi1 = fi2
        if done:
            env.reset()
    
    print "Experience replay was filled up."
    
    # Start learning.
    while(not exit):
        episend = False	
        obs = env.reset()
        fi = preprocessing(obs)
        action = os.nextActionAndTrain(fi, 0.0)
        while(not episend):
            obs, rw, done, inf = env.step(action)
            cntr += 1
            if cntr % 100 == 0:
                print "Current iteration: %r" % cntr
            fi = preprocessing(obs)
            action = os.nextActionAndTrain(fi, rw)
            episend = done
        exit = os.isTrainingFinished()
        if (m.floor(cntr / evaluation_freq)-k_evaluate) > 0.0001: # evaluate the performance of the agent
            k_evaluate += 1
            evaluate(os)
            print "An evaluation occured."
        
    os.saveAgent(fname)  
    
    return 0
