import json

class Logger:
    """ Class for saving the performance indicators.

    Saves the rewards and the frames durig evaluation.

    Can create statistics from the results.

    """

    def __init__(self, runs_evaluation):
        self.cntr = 0
        self.ret = []
        #self.frames = []
        self.runs_evaluation = runs_evaluation
        self.runs_in_eval = 0
        self.num_evals = 1

    def write(self, obs, rw, done):
        self.cntr += 1
        self.ret.append(rw)
        #self.frames.append(obs.tolist())
        if done:
            print self.cntr
            self.runs_in_eval +=1
            #frame_file_name = "files/frames" + str(self.num_evals) + "_" + str(self.runs_in_eval) + ".json"
            rw_file_name = "files/reward" + str(self.num_evals) + "_" + str(self.runs_in_eval) + ".json"
            if self.runs_in_eval == self.runs_evaluation:
                self.runs_in_eval = 0
                self.num_evals += 1
            #with open(frame_file_name, 'w') as f:
                #json.dump(self.frames, f)
            with open(rw_file_name, 'w') as f:
                json.dump(self.ret, f)
            self.cntr = 0
            self.ret = []
            #self.frames = []
    
    
