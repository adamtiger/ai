import cv4pool as cv4p

class BaseEnvironment

    def __init__(self, size_of_pool, refresh_freq):
        self.pool = cv4p.ImagePool(size_of_pool, refresh_freq)

    def generate_new_situation(self):
        pass

    def receive_state(self):
        pass

    def receive_reward(self):
        pass

    def step(self):
        pass

    def update_base_image():
        self.pool.next_image()

    def _get_correct(self):
        pass