import numpy as np
import logging
from cv4rl.cv4pool import pool
from cv4rl.cv4alg import KerasModeltoJSON as js

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten
from keras.optimizers import RMSprop

# configure logger
logging.basicConfig(format='%(asctime)s %(message)s', level = logging.DEBUG,
                    filename = 'output.log')

class Environment:
    
    def __init__(self, args):
        self.pool = pool.ImagePool(args.pool_size, args.refresh_freq, args.folder)
        self.image = self.pool.next_image()
        self.framed_image = self.__create_framed_img()
        self.batch_size = args.batch_size
        self.num_iter = args.num_iter
        self.C = args.C
        self.dnn = Dnn(args.batch_size, args.alpha)
        
        
    def __map2Y(self, img):
        np_img = np.array(img)
        shape = np_img.shape
        shape_new = (shape[0], shape[1], 1)
        ou_img = np.zeros(shape_new)

        ou_img[:,:,0] = (2*np_img[:,:,0] + 5*np_img[:,:,1] + np_img[:,:,2])/8.0
        return ou_img
    
    def __crop(self, height, width, x, y):
        cropped = np.ndarray((height, width, 3))
        for i in range(0, height):
            for j in range(0, width):
                cropped[i,j, :] = self.framed_image[x - int(height/2) + i, y - int(width/2) + j]
        return cropped
    
    def __create_framed_img(self):
        shp = (self.image.shape()[0]+86, self.image.shape()[1]+86, 3)
        fr_img = np.zeros(shp) + 255
        fr_img[43:-43,43:-43,:] = getattr(self.image, 'base_img')[:,:,:]
        return fr_img
    
    def __update_base_image(self):
        self.image = self.pool.next_image()
        self.framed_image = self.__create_framed_img()
        
    def train(self):
        for cntr in range(0, self.num_iter):
            
            y = np.zeros((self.batch_size))
            w_l = self.pool.get_white_vec_length()
            b_l = self.pool.get_black_vec_length()
            states = np.zeros((self.batch_size, 84,84,1))
            for i in range(0, self.batch_size):
                current = (0,0) # defining current
                if (i % 2) == 0:    
                    y[i] = 0
                    idx = np.random.randint(0, high=w_l)
                    current = self.pool.get_coord_from_white_vec(idx)
                else:
                    y[i] = 255
                    idx = np.random.randint(0, high=b_l)
                    current = self.pool.get_coord_from_black_vec(idx) 
                 
                states[i] = self.__get_new_state(current)

            tr_batch = [states, y]
            self.dnn.train(tr_batch)
            
            if (cntr % self.C) == 0:
                self.evaluate()
                
            if (cntr % self.refresh_freq_image) == 0:
                self.__update_base_image()
        
        self.dnn.model_to_json('saved.json')
        
    # create one color image 
    def __get_new_state(self, current_coord):
        img = self.__crop(84, 84, current_coord[0], current_coord[1])
        obs = self.__map2Y(img)

        ou_state = np.zeros((1,84,84,1))
        ou_state[0,:,:,0] = obs[:,:,0]
        return np.uint8(ou_state)


    def evaluate(self):
        y = np.zeros((1))
        w_l = self.pool.get_white_vec_length()
        b_l = self.pool.get_black_vec_length()
        state = np.zeros((1, 84,84,1))
        success = 0
        all_tests = 0
        for i in range(0, 20):
            current = (0,0) # defining current
            if (i % 2) == 0:    
                y[0] = 0
                idx = np.random.randint(0, high=w_l)
                current = self.pool.get_coord_from_white_vec(idx)
            else:
                y[i] = 255
                idx = np.random.randint(0, high=b_l)
                current = self.pool.get_coord_from_black_vec(idx) 
                 
            state[0] = self.__get_new_state(current)
            z = self.dnn.predict(state)
            if (z[0,0] == y[0]):
                success += 1
            all_tests += 1
        
        ratio = float(success)/float(all_tests)
        logging.info("Evaluation performance: " + str(ratio))
                
# The neural network
class Dnn:

    def __create_model(self, alpha):
      model = Sequential()
      model.add(Conv2D(32, (8, 8), border_mode='valid', input_shape=(84, 84, 1), strides=(4, 4)))
      model.add(Activation('relu'))
      model.add(Conv2D(64, (4, 4), border_mode='valid', strides=(2, 2)))
      model.add(Activation('relu'))
      model.add(Conv2D(64, (3, 3), border_mode='valid', sstrides=(1, 1)))
      model.add(Activation('relu'))
      model.add(Flatten())
      model.add(Dense(512))
      model.add(Activation('relu'))
      model.add(Dense(1))
      
      rmsprop = RMSprop(lr=alpha)
      model.compile(optimizer=rmsprop, loss='mse')
      
      return model

    def __init__(self, batch_size, alpha):
        self.batch_size = batch_size
        self.alpha = alpha
        self.Weight = self.__create_model(alpha)

    def get_batch_size(self):
        return self.batch_size
        
    def train(self, mini_batch):
        self.Weight.fit(mini_batch[0], mini_batch[1], nb_epoch=1, batch_size=self.batch_size, verbose=0)
    
    def predict(self, state):
        return self.Weight.predict(state, batch_size=1)
        
    def save(self, fname):
        self.Weight.save_weights(fname)
        
    def load(self, fname):
        self.Weight.load_weights(fname, by_name = False)
        
    def model_to_json(self, fname):
        wrt =js.JSONwriter(self.Weight, fname)
        wrt.save()
