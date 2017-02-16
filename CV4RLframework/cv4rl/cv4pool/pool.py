import numpy as np
from scipy import misc
import os

class Image:

    def __init__(self, base_img, segmented_img):
        self.base_img = base_img
        self.sgm_img = segmented_img
        
    def get_pixel_base(self, x, y):
        return self.base_img[x, y]
    
    def get_pixel_sgm(self, x, y):
        return self.sgm_img[x, y]
        

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
            base_img = misc.imread(self.folder + '/' + self.file_list[idx])
            sgm_img = misc.imread(self.folder + '/' +self.file_list[idx + 1])
            img = Image(base_img, sgm_img)
            self._add(img)
    
    def _add(self, image):
        self.images[self.idx] = image
        self.idx = (self.idx + 1) % self.size 