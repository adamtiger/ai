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

    def __init__(self, size_pool, refresh_freq):
        self.refresh_freq = refresh_freq
        self.size = size_pool
        self.images = [0]*size_pool
        self.idx = 0
        self.cntr = 0
        self.file_list = os.listdir("images")

        assert len(self.file_list) % 2 == 0, "A solution or a base image is missing!"
        
    def get_random_img(self):
        if self.cntr == refresh_freq:
            __refresh()
            self.cntr = 0
        self.cntr += 1
        
        rand_idx = np.random.randint(0, self.size-1)
        return self.images[rand_idx]

    def __refresh(self):
        rand_idxs = np.random.randint(0, self.file_list.size/2-1, (self.size))

        for idx in rand_idxs:
            base_img = scipy.misc.imread("images/" + self.file_list[idx])
            sgm_img = scipy.misc.imread("images/" + self.file_list[idx + 1])
            img = Image(base_img, sgm_img)
            __add(img)
        
    def __add(self, image):
        self.images[self.idx] = image
        self.idx = (self.idx + 1) % self.size    
