import json

class Logger:
    """ Class for saving the performance indicators.

    Saves the rewards and the frames durig evaluation.

    Can create statistics from the results.

    """

    def __init__(self):
        self.cntr = 0
        self.ret = 0
        self.frames = []

    def write(self, obs, rw, done):
        self.cntr += 1
        self.ret += rw
        self.frames.append(obs)
        if done:
            print self.cntr
            self.cntr = 0
            self.ret = 0
            with open("files/file1.json", 'w') as f:
                json.dump(self.frames, f)
                json.dump(self.ret,f)
    
    
