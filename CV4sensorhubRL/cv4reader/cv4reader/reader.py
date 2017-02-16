import pool

class Cv4Reader:

    def __init__(self, size_of_pool, refresh_freq):
        self.image_pool = pool.ImagePool(size_of_pool, refresh_freq)
        
    def next_image(self):
        return self.image_pool.get_random_img()    
