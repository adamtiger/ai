import BaseAgent
import gym
import dqn

class RiverAgent(BaseAgent):

  

  def makeEnvironment(self):
    env = gym.make('Riverrider-v0')
    env.reset()
    return env
    
  def nextActionAndTrain(self, obs, rw, done, inf):
    raise NotImplementedError()
    
  def nextAction(self, obs, rw, done, inf):
    raise NotImplementedError()
    
  def isTrainingFinished(self):
    raise NotImplementedError()
    
  def saveAgent(self, fname):
    raise NotImplementedError()
