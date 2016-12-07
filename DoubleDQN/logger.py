
class Logger:
    """ Class for saving the performance indicators.

    Saves the rewards and the frames durig evaluation.

    Can create statistics from the results.

    """

    def __init__(self):
        self.cntr = 0

    def write(self, obs, rw, done):
        self.cntr += 1
        if done:
            print self.cntr
            self.cntr = 0
    
    
