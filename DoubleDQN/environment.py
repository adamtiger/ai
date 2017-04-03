import numpy as np
import scipy.misc as sc
import math as m

# This file is for running the different Atari
# environments. 3 of them are examined:
#  -> Freeway (a chicken goes through the road)
#  -> Breakout (like tetris)
#  -> Riverraid (an airplane flies above a river)
# This implementation can run all of them.

import logger
import agent
import tf
import dqn


class Preprocessing:
    
    def __init__(self, env):
        
        """ 
        At the very beginning there was not
        appeared four consequtive frames yet.

        Gathers the first four frames during no-op.
        Creates the first state.
        """
        self.state = [] # list to store the most recent frames
        
        for i in range(0, 4):
            img, rw, done, inf = env.step(0) # no-op
            imgY = self.map2Y(img)
            img_crop = self.crop(imgY)
            obs = self.rescale(img_crop)
            self.state.append(obs)

    def map2Y(self, img):
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
        
    def crop(self, img):
      img_cropped = np.zeros((185, 160, 1))
      img_cropped[:,:,0] = img[16:201,:,0]
      return img_cropped 
    
    def rescale(self, img):
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

    def preprocessing(self, img):
        
        imgY = self.map2Y(img)
        img_crop = self.crop(imgY)
        obs = self.rescale(img_crop)
        del self.state[0]
        self.state.append(obs)
        ou_state = np.zeros((1,84,84,4))
        ou_state[0,:,:,0] = self.state[0][:,:,0]
        ou_state[0,:,:,1] = self.state[1][:,:,0]
        ou_state[0,:,:,2] = self.state[2][:,:,0]
        ou_state[0,:,:,3] = self.state[3][:,:,0]
        return np.uint8(ou_state)

class Environment:
    
    def __init__(self, parser):
        dqn_f = dqn.DQN()
        self.agent = agent.Agent(dqn_f, parser.atari_env)
        self.env = self.agent.makeEnvironment()
        
        tf_f = tf.Dnn(self.env.action_space.n, 32, parser.lr)
        dqn_f.set_params(tf_f, parser.C, parser.max_iter, parser.mem_size, parser.exp_start, parser.exp_end, parser.last_fm, parser.gamma)
        self.evaluation_freq = parser.eval_freq
        self.evaluation_number = parser.eval_num
        self.log = logger.Logger(parser.eval_num)
        self.init_number_in_replay_mem = parser.init_replay_size
        
        
    def evaluate(self):
    
        self.env.reset()
        pre = Preprocessing(self.env)
        print("Evaluation started.")

        for i in range(0, self.evaluation_number):
            episend = False
            obs = self.env.reset()
            fi = pre.preprocessing(obs)
            action = 0
            while action == 0:
                action = self.env.action_space.sample()

            while(not episend):
                obs, rw, done, inf = self.env.step(action)
                self.log.write(obs, rw, done)
                fi = pre.preprocessing(obs)
                action = self.agent.nextAction(fi)
                episend = done


    def train(self, fname):
    
        print ("Start training.")
    
        self.env.reset()
        exit_ = False
        cntr = 0
        k_evaluate = 0
    
        pre = Preprocessing(self.env)
    
        print ("State was initialized.")
    
        # Fill up the experience replay memory with
        # experiences.
        action1 = self.env.action_space.sample()
        obs1, rw1, done, inf = self.env.step(action1)
        fi1 = pre.preprocessing(obs1)
        for i in range(1, self.init_number_in_replay_mem):
            action2 = self.env.action_space.sample()
            obs2, rw2, done, inf = self.env.step(action2)
            fi2 = pre.preprocessing(obs2)
            self.agent.init(fi1, action1, rw1, fi2)
            action1 = action2
            rw1 = rw2
            fi1 = fi2
            if done:
                self.env.reset()
    
        print ("Experience replay was filled up.")
    
        # Start learning.
        while(not exit_):
            episend = False	
            obs = self.env.reset()
            fi = pre.preprocessing(obs)
            action = self.agent.nextActionAndTrain(fi, 0.0)
            while(not episend):
                obs, rw, done, inf = self.env.step(action)
                cntr += 1
                if cntr % 1000 == 0:
                    print ("Current iteration: %r" % cntr)
                fi = pre.preprocessing(obs)
                action = self.agent.nextActionAndTrain(fi, rw)
                episend = done
            exit_ = self.agent.isTrainingFinished()
            if (m.floor(cntr / self.evaluation_freq)-k_evaluate) > 0.0001: # evaluate the performance of the agent
               k_evaluate += 1
               self.evaluate()
               print ("An evaluation occured.")
        
        self.agent.saveAgent(fname)  
