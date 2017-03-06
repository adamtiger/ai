import json

class Logger:
    """ Class for saving the performance indicators.

    Saves the rewards and the frames durig evaluation.

    Can create statistics from the results.

    """

    def __init__(self, runs_evaluation):
        self.cntr = 0
        self.ret = []
        self.sum_ret = 0.0
        self.runs_evaluation = runs_evaluation
        self.runs_in_eval = 0
        self.num_evals = 1

    def write(self, obs, rw, done):
        self.cntr += 1
        self.ret.append(rw)
        self.sum_ret += rw 
        if done:
            print (self.sum_ret)
            self.runs_in_eval +=1
            self.ret.append(self.sum_ret)
            self.sum_ret = 0.0
            rw_file_name = "files/rewards/reward" + str(self.num_evals) + "_" + str(self.runs_in_eval) + ".json"
            if self.runs_in_eval == self.runs_evaluation:
                self.runs_in_eval = 0
                self.num_evals += 1
            with open(rw_file_name, 'w') as f:
                json.dump(self.ret, f)
            self.cntr = 0
            self.ret = []
    
        
    
    
