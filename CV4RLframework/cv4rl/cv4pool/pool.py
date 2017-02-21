import numpy as np
from scipy import misc
import os

class Image:

    def __init__(self, base_img, segmented_img):
        self.base_img = base_img
        self.sgm_img = segmented_img
        self.white_vec = self.__gen_segm_vec()
        
    def get_pixel_base(self, x, y):
        return self.base_img[x, y]
    
    def get_pixel_sgm(self, x, y):
        return self.sgm_img[x, y]
        
    def get_coord_from_white_vec(self, idx):
        return self.white_vec[idx]
        
    def get_white_vec_length(self):
        return len(self.white_vec)
    
    def shape(self):
        return self.base_img.shape
    
    def __gen_segm_vec(self):
        x_size = self.sgm_img.shape[0] # May be reverse order!
        y_size = self.sgm_img.shape[1]
        vec = []
        
        for x in range(0, x_size):
            for y in range(0, y_size):
                if (self.sgm_img[x, y] > 200.0):
                    vec.append((x, y))
                    self.sgm_img[x, y] = 255.0
                elif (self.sgm_img[x, y] < 200.0):
                    self.sgm_img[x, y] = 0.0
        
        return vec
                    
                    
class ImagePool:
    
    def __init__(self, size_pool, refresh_freq, folder):
        self.size = size_pool
        self.refresh_freq = refresh_freq
        self.folder = folder
        self.images = [0]*size_pool
        self.idx = 0
        self.cntr = 0
        self.file_list = os.listdir(folder)

        assert len(self.file_list) % 2 == 0, "A solution or a base image is missing!"
        
        self._refresh()

    def next_image(self):
        return self.__get_random_img()
    
    def __get_random_img(self):
        if self.cntr == self.refresh_freq:
            self._refresh()
            self.cntr = 0
        self.cntr += 1
        
        rand_idx = np.random.randint(0, self.size)
        return self.images[rand_idx]
    
    def _refresh(self):
        rand_idxs = np.random.randint(0, len(self.file_list)/2, (self.size))

        for idx in rand_idxs:
            base_img = misc.imread(self.folder + '/' + self.file_list[idx], False, 'RGB')
            sgm_img = misc.imread(self.folder + '/' +self.file_list[idx + 1], False, 'L')
            img = Image(base_img, sgm_img)
            self._add(img)
    
    def _add(self, image):
        self.images[self.idx] = image
        self.idx = (self.idx + 1) % self.size 