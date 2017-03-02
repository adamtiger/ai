import json

class Logger:
    """ Class for saving the performance indicators.

    Saves the rewards and the frames durig evaluation.

    Can create statistics from the results.

    """

    def __init__(self, runs_evaluation):
        self.cntr = 0
        self.ret = []
        self.runs_evaluation = runs_evaluation
        self.runs_in_eval = 0
        self.num_evals = 1

    def write(self, obs, rw, done):
        self.cntr += 1
        self.ret.append(rw)
        if done:
            print (self.cntr)
            self.runs_in_eval +=1
            rw_file_name = "files/rewards/reward" + str(self.num_evals) + "_" + str(self.runs_in_eval) + ".json"
            if self.runs_in_eval == self.runs_evaluation:
                self.runs_in_eval = 0
                self.num_evals += 1
            with open(rw_file_name, 'w') as f:
                json.dump(self.ret, f)
            self.cntr = 0
            self.ret = []

