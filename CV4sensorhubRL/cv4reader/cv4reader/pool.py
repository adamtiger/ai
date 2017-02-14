import numpy as np
from scipy import misc

class Image:

    def __init__(self, base_img, segmented_img):
        self.base_img = base_img
        self.sgm_img = segmented_img
        
    def get_pixel_base(self, x, y):
        pass
    
    def get_pixel_sgm(self, x, y):
        pass
        

class ImagePool:

    def __init__(self, size_pool):
        self.size = size_pool
        self.images = [0]*size_pool
        self.idx = 0
        
    def add(self, image):
        self.images[self.idx] = image
        self.idx = (self.idx + 1) % size_pool
        
    def get_random_img(self):
        self