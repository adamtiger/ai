import environment
import gym
import dqn

# Test Experience replay in dqn.py
def setup_rpy():
    return dqn.ExpReplay(5, 3)

def test_ExperienceReplay_init():
    eReply = setup_rpy()
    assert eReply.capacity() == 5, "the capacity is not set correctly: %r" % eReply.capacity()

    print "successful: test_ExperienceReplay_init"
     
def test_ExperienceReplay_add():
    eReply = setup_rpy()
    e1 = (10, 6, 8)
    e2 = (12, 9, 11)
    e3 = (32, 7, 9)
    e4 = (25, 1, 76)
    e5 = (20, 0, 65)
    e6 = (21, 2, 34)
    eReply.add(e1)
    eReply.add(e2)
    assert getattr(eReply, 'length') == 2, "number of experiences should be 2: %r" % getattr(eReply, 'length')
    eReply.add(e3)
    eReply.add(e4)
    eReply.add(e5)
    eReply.add(e6)
    assert getattr(eReply, 'mem')[0] == e6, "at the first place e6 should be present: %r" % getattr(eReply, 'mem')[0]
    assert getattr(eReply, 'mem')[1] == e2, "at the second place e2 should be present: %r" % getattr(eReply, 'mem')[1]
    assert getattr(eReply, 'mem')[3] == e4, "at the second place e4 should be present: %r" % getattr(eReply, 'mem')[3]
    
    print "successful: test_ExperienceReplay_add"
    
def test_ExperienceReplay_sample():
    eReply = setup_rpy()
    e1 = (10, 6, 8)
    e2 = (12, 9, 11)
    e3 = (32, 7, 9)
    e4 = (25, 1, 76)
    e5 = (20, 0, 65)
    eReply.add(e1)
    eReply.add(e2)
    eReply.add(e3)
    eReply.add(e4)
    eReply.add(e5)
    batch = eReply.sample()
    assert len(batch) == 3, "batch should contain 3 elements!"
    assert len(batch[2]) == 3, "tuple should have 3 elements!"
    
    print "successful: test_ExperienceReplay_sample"

def test_ExperienceReplay():
    test_ExperienceReplay_init()
    test_ExperienceReplay_add()
    test_ExperienceReplay_sample()

# --------------------------------------

# Test epsilon greedy policy in dqn.py
def setup_greedy():
    return dqn.EpsGreedy(0.6, 0.1, 10, 5)

def test_EpsGreedy_init():
    grdy = setup_greedy()
    assert getattr(grdy, 'exp') == 0.6, "the value of exp should be 1.0 instead of: %r" % getattr(grdy, 'exp')
    assert getattr(grdy, 'frame') == 10, "the value of frame should be 10 instead of: %r" % getattr(grdy, 'frame')
    
    print "successful: test_EpsGreedy_init"

def test_EpsGreedy_action():
    grdy = setup_greedy()
    ls = []
    for i in range(0, 10):
        act = grdy.action(3)
        ls.append(act)
        assert act >= 0 and act < 5, "returned action should be between 0 and 4: %r" % act
    
    print "successful: test_EpsGreedy_action"

def test_EpsGreedy_anneal():
    grdy = setup_greedy()
    for i in range(0, 5):
        grdy.anneal()
    assert getattr(grdy, 'exp') == 0.35, "the value exp should be 0.35 instead of: %r" % getattr(grdy, 'exp')
    
    print "successful: test_EpsGreedy_anneal"

def test_EpsGreedy():
    test_EpsGreedy_init()
    test_EpsGreedy_action()
    test_EpsGreedy_anneal()

# ------------------------------------

# Test for DQN.

import tf
import random as r
r.seed(113)

# Mocking class for tests:

class MockDnn(tf.IDnn):

    def __init__(self, actions, batch_size):
        self.actions = actions
        self.batch_size = batch_size
        self.seq = []

    def get_action_number(self):
        return self.actions

    def get_batch_size(self):
        return self.batch_size
        
    def get_seq(self):
        return self.seq

    def argmaxQ(self, state): # Method Id = 1
        self.seq.append(1)
        return r.randint(0, self.actions-1)
        
    def Q_frozen(self, state, action): # Method Id = 2
        self.seq.append(2)
        return r.uniform(0.0, 5.0)
    
    def update_network(self): # Method Id = 3
        self.seq.append(3)
        
    def train(self, mini_batch): # Method Id = 4
        self.seq.append(4)
        assert self.batch_size == len(mini_batch), "mini_batch should have %r length" % self.batch_size
        assert len(mini_batch[0]) == 3, "a training sampe should be a tuple with 3 elements instead of: %r" % len(mini_batch[0])

def setup_DQN():
    mock = MockDnn(10, 3)
    class_dqn = dqn.DQN()
    class_dqn.set_params(mock, 8, 25, 5, 1.0, 0.1, 2, 0.99)
    return class_dqn

def test_DQN_init():
    dq = setup_DQN()
    
    e1 = (10, 6, 8, 12)
    e2 = (12, 9, 11, 32)
    e3 = (32, 7, 9, 25)
    e4 = (25, 1, 76, 20)
    e5 = (20, 0, 65, 22)
    
    dq.init(e1[0], e1[1], e1[2], e1[3])
    dq.init(e2[0], e2[1], e2[2], e2[3])
    dq.init(e3[0], e3[1], e3[2], e3[3])
    dq.init(e4[0], e4[1], e4[2], e4[3])
    dq.init(e5[0], e5[1], e5[2], e5[3])
    
    eReply = getattr(dq, 'erply')
    assert getattr(eReply, 'length') == 5, "number of experiences should be 5: %r" % getattr(eReply, 'length')
    assert getattr(eReply, 'mem')[0] == e1, "at the first place e1 should be present: %r" % getattr(eReply, 'mem')[0]
    assert getattr(eReply, 'mem')[1] == e2, "at the second place e2 should be present: %r" % getattr(eReply, 'mem')[1]
    assert getattr(eReply, 'mem')[3] == e4, "at the second place e4 should be present: %r" % getattr(eReply, 'mem')[3]
    
    print "successful: test_DQN_init"

def test_DQN_train():
    dq = setup_DQN()
    cntr = 0
    _end = False
    ls = []
    while not _end and cntr < 200:
        cntr += 1
        obs = r.randint(0,6)
        act = r.randint(0,9)
        rw = r.uniform(-1.0, 1.0)
        obs_next = r.randint(0,6)
        dq.train(obs, act, rw, obs_next)
        _end = dq.end()
        if cntr % 4 == 0:
            for i in range(0, 3):
                ls.append(1)
                ls.append(2)
            ls.append(4)
            if cntr % 8 == 0:
                ls.append(3)

    assert ls == getattr(dq, '_tf').get_seq(), "different calling sequences!"
    
    grdy = getattr(dq, 'grdy')
    exp = getattr(grdy, 'exp')
    assert abs(exp - 0.1) < 0.0001, "exp should be 0.1" 
    
    print "successful: test_DQN_train"

def test_DQN_action():
    dq = setup_DQN()
    for i in range(0, 5):
        dq.action(0)
        
    print "successful: test_DQN_action"

def test_DQN():
    test_DQN_init()
    test_DQN_train()
    test_DQN_action()

# --------------------------------------------
import numpy as np

# Test for the neural network.

def setup_network():
    nn = tf.Dnn(5, 32, 0.0001)
    return nn

def test_network_init():
    nn = setup_network()
    
    assert getattr(nn, 'actions') == 5, "The number of actions should be 5 instead of: %r" % getattr(nn, 'actions')
    assert getattr(nn, 'batch_size') == 32, "The number of actions should be 32 instead of: %r" % getattr(nn, 'batch_size')
    getattr(nn, 'Q').summary()
    
    print "successful: test_network_init"
    
def test_network_get_action():
    nn = setup_network()
    
    assert nn.get_action_number() == 5, "The number of actions should be 5 instead of: %r" % nn.get_action_number()
    
    print "successful: test_network_get_action"

# Only checks whether the sizes are correct and the learning happens.
# No assertion.
def test_network_train():
    nn = setup_network()
    
    
    # Generate examples 
    n = 320
    x = np.random.rand(n, 84, 84, 4)
    a = np.random.randint(0, 5, (n))
    y = np.random.rand(n)
    
    for i in range(1, n/32):
        l = (i-1)*32
        h = i*32
        batch = [x[l:h],a[l:h], y[l:h]]
        nn.train(batch)
    
    nn.update_network()
    
    nn.argmaxQ(x[0:1])
    nn.Q_frozen(x[1:2], 1)
    
    print "successful: test_network_train"
    
def test_network():
    test_network_init()
    test_network_get_action()
    test_network_train()

# -----------------------------------------

# Test preprocessing

def test_preprocessing():
    env = gym.make('Breakout-v0')
    img = env.reset()
    environment.init_state(env)
    ou = environment.preprocessing(img)
    
    assert (1,84,84,4) == ou.shape, "The output shape is wrong."
    
    print "successful: test_preprocessing"

# -----------------------------------------

import logger

# Test logger

def test_logger():
    obs = np.random.rand(12,84,84,4)
    rw = np.random.rand(12)
    done = [False, False, True, False, True, False, True, False, False, True, False, True]

    l = logger.Logger(3)

    for i in range(0, 12):
        l.write(obs[i], rw[i], done[i])

# Function to run all tests
def run_AllTests():
    
    test_ExperienceReplay()
    test_EpsGreedy()
    test_DQN()
    test_network()
    test_preprocessing()
    test_logger()
    
    print "All tests succeeded."
    return True
  
# RUN THE TESTS  
run_AllTests()
