import gym

# This file is for running the different Atari
# environments. 3 of them are examined:
#  -> Freeway (a chicken goes through the road)
#  -> Breakout (like tetris)
#  -> Riverraid (an airplane flies above a river)
# This implementation can run all of them.

import BaseAgent

def testGame(os):
  
  env = os.makeEnvironment()
  exit = False

  episend = False	
  action = os.nextAction(obs)

  while(!episend):
    obs, rw, done, inf = env.step(action)
    action = os.nextAction(obs)
    episend = done
  
  return 0

# TODO: evaluation is needed.
def train(os, fname):

  env = os.makeEnvironment()
  exit = False

  # Fill up the experience replay memory with
  # experiences.
  action1 = env.action_states.sample()
  obs1, rw1, done, inf = env.step(action)
  for i in range(1,50000):
    action2 = env.action_states.sample()
    obs2, rw2, done, inf = env.step(action)
    os.init(obs1, action1, rw1, obs2)
    action1 = action2
    rw1 = rw2
    obs1 = obs2
    if done:
      env.reset()

  # Start learning.
  while(!exit):
    episend = False	
    obs = env.reset()
    action = os.nextActionAndTrain(obs, 0.0)
    while(!episend):
      obs, rw, done, inf = env.step(action)
      action = os.nextActionAndTrain(obs, rw)
      episend = done
    exit = os.isTrainingFinished()
    
  os.saveAgent(fname)  
    
  return 0
