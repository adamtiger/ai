import random as r
import numpy as np


r.seed(133)

def normalize_img(img):
    return img/255.0 - 0.5


# This is the concrete implementation of Double DQN.

# Class for storing and handling experiences.
class ExpReplay:

    def __init__(self, capacity, max_batchSize):
        self.cap = capacity
        self.length = 0
        self.mem = [()]*capacity
        self.cntr = 0
        self.mxbtch = max_batchSize

    def capacity(self):
        return self.cap

    def add(self, exp):
        self.mem[self.cntr] = exp
        self.cntr = (self.cntr + 1) % self.cap
        if self.length != self.cap:
            self.length = self.length + 1

    # Returns a list of tuples. (experiences)
    def sample(self):
        batch = [()]*self.mxbtch
        for i in range(0, self.mxbtch):
            k = r.randint(0, self.length-1)
            element = (normalize_img(self.mem[k][0]), self.mem[k][1], self.mem[k][2], normalize_img(self.mem[k][3]))
            batch[i] = element
        return batch

# A class implements the epsilon greedy policy.
class EpsGreedy:

    def __init__(self, init_exp, final_exp, fn_frame, actions):
        self.start = init_exp # The exploration at the beginning
        self.end = final_exp
        self.frame = fn_frame # The number of frames during the exploration is linearly annealed
        self.exp = self.start
        self.actions = actions
        self.no_op = 0
        self.cntr = 0

    def action(self, act):
        explore = (r.random() < self.exp)
        c_act = act
        if act == 0:
            self.no_op += 1
        if explore and self.no_op < 20:
            k = r.randint(0, self.actions-2)
            if k >= act:
              c_act = k + 1
            else:
              c_act = k
        elif explore:
            c_act = r.randint(1, self.actions-2) # do not generate no_op (act = 0)
            self.no_op = 0
        return c_act

    def anneal(self):
        self.exp = self.exp - (self.start - self.end)/self.frame
        #self.exp = (self.start - self.end)*math.exp(-self.cntr/self.frame) + self.end
        #self.cntr += 1


# Class for implementing Double deep Q-network.
class DQN:

    def __init__(self):
        self._cntr = 0

    def set_params(self, tf, C, max_iter, mem_size, exp_start, exp_end, last_fm, gamma):
        self._actions = tf.get_action_number()
        self._tf = tf
        self.C = C
        self.max_iter = max_iter
        self.batch_size = tf.get_batch_size()
        self.last_fm = last_fm
        self.erply = ExpReplay(mem_size, self.batch_size)
        self.grdy = EpsGreedy(exp_start, exp_end, last_fm, self._actions)
        self._gamma = gamma
    
    # Functions to interact with the environment.

    def init(self, obs, action, rw, obs_nx):
        tp_exp = (obs, action, rw, obs_nx)
        self.erply.add(tp_exp)

    def train(self, obs, action, rw, obs_nx):
        self._cntr += 1
        self.erply.add((obs, action, rw, obs_nx))
        if self._cntr % 4 == 0:
            batch = self.erply.sample()
            y = np.zeros((self.batch_size))
            actions = np.zeros((self.batch_size))
            states = np.zeros((self.batch_size, 84,84,4))
            for i in range(0, self.batch_size):
                a = self._tf.argmaxQ(batch[i][3])
                y[i] = batch[i][2] + self._gamma * self._tf.Q_frozen(batch[i][3], a)
                actions[i] = batch[i][1]
                states[i] = batch[i][0]
            actions_int = actions.astype(int)
            tr_batch = [states[:], actions_int[:], y[:]]
            self._tf.train(tr_batch)
            if self._cntr % self.C == 0:
                self._tf.update_network()
        if self._cntr <= self.last_fm:
            self.grdy.anneal()

    def action(self, obs):
        state = normalize_img(obs)
        a = self._tf.argmaxQ(state)
        a = self.grdy.action(a)
        return a
        
    def action_nogreedy(self, obs):
        state = normalize_img(obs)
        a = self._tf.argmaxQ(state)
        rd_nm = r.randint(0,100)
        if (rd_nm < 5):
            a = r.randint(0, self._actions-1)        

        return a

    def end(self):
        return self._cntr > self.max_iter
        
    def save(self, fname):
        self._tf.save(fname)



