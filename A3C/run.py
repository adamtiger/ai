# -*- coding: utf-8 -*-

from  a3c import A3C, Parameters
import nn

max_it = 10000



class Launch:
    
    def __init__(self, num_threads, atari_name):
        self.shared_vectors = nn.CreateShared()
        self.parameters = Parameters(max_it, self.shared_vectors)
        self.threads = []

        for idx in range(0, num_threads):
            self.threads.append(A3C(atari_name, self.parameters))
            

    def start(self):
        for idx in range(0, len(self.threads)):
            self.threads[idx].start()
        


# RUN the algorithm

launching = Launch(2, "Breakout-v0")

launching.start()