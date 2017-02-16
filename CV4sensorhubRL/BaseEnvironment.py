import cv4reader as cv4

class BaseEnvironment

    def __init__(self, size_of_pool, refresh_freq):
        self.reader = cv4.Cv4Reader(size_of_pool, refresh_freq)

    def generate_new_situation(self):
        pass

    def receive_state(self):
        pass

    def receive_reward(self):
        pass

    def step(self):
        pass

    def update_base_image():
        self.reader.next_image()

    def _get_correct(self):
        pass

    
