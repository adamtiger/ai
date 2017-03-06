# -*- coding: utf-8 -*-

import numpy as np
import scipy.misc as sc

class PreProcessing:
    
    '''
    This class is responsible for preprocessing the raw frames.
    The method follows the procedure described in (Mnih, 2015).
    Shortly:
        1. Transforms the image into gray-scale.
        2. Rescales it to 84x84.
        3. Stacks the last four most recent frames together.
    
    This class has only one public method, the preprocessing. It 
    has one input observation which is a concrete raw frame provided by 
    the OpenAi gym environment.
        
    '''
    
    def __init__(self, env):
        self.state = []
        for i in range(0, 4):
            img, rw, done, inf = env.step(0) # no-op
            imgY = self.__map2Y(img)
            obs = self.__rescale(imgY)
            self.state.append(obs)
        
    def __map2Y(self, img):
    
        np_img = np.array(img)
        shape = np_img.shape
        shape_new = (shape[0], shape[1], 1)
        ou_img = np.zeros(shape_new)

        ou_img[:,:,0] = (2*np_img[:,:,0] + 5*np_img[:,:,1] + np_img[:,:,2])/8.0
        return ou_img
    
    def __rescale(self, img):
    
        shape = img.shape
        img_ = np.zeros((shape[0], shape[1]))
        img_[:,:] = img[:,:,0]
    
        img_resized = sc.imresize(img_, (84,84), interp='bilinear', mode=None)
    
        img_ou = np.zeros((84,84,1))
        img_ou[:,:,0] = img_resized[:,:]
    
        return img_ou
        
    def preprocessing(self, observation):
        
        imgY = self.__map2Y(observation)
        obs = self.__rescale(imgY)
        del self.state[0]
        self.state.append(obs)
        ou_state = np.zeros((1,84,84,4))
        ou_state[0,:,:,0] = self.state[0][:,:,0]
        ou_state[0,:,:,1] = self.state[1][:,:,0]
        ou_state[0,:,:,2] = self.state[2][:,:,0]
        ou_state[0,:,:,3] = self.state[3][:,:,0]
        return np.uint8(ou_state)
