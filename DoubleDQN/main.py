import gym

# This file is for running the different Atari
# environments. 3 of them are examined:
#  -> Freeway (a chicken goes through the road)
#  -> Breakout (like tetris)
#  -> Riverraid (an airplane flies above a river)
# This implementation can run all of them.

import BaseAgent

def train(os, fname):

  env = os.makeEnvironment()
  exit = False

  while(!exit):
    episend = False	
    obs = env.reset()
    action = os.nextActionAndTrain(obs, 0.0, False, {})
    while(!episend):
      obs, rw, done, inf = env.step(action)
      action = os.nextActionAndTrain(obs, rw, done, inf)
      episend = done
    exit = os.isTrainingFinished()
    
  os.saveAgent(fname)  
    
  return 0

def testGame(os):
  
  env = os.makeEnvironment()
  exit = False

  episend = False	
  obs = env.reset()
  action = os.nextAction(obs, 0.0, False, {})

  while(!episend):
    obs, rw, done, inf = env.step(action)
    action = os.nextAction(obs, rw, done, inf)
    episend = done
  
  return 0