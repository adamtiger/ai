import gym

# This is the abstract class for al of the agents.


class BaseAgent:

  def __init__(self, dqn):
    self.DQN = dqn
    self.action = 0
    self.rw = 0
    self.obs_old = []
  
  def makeEnvironment(self):
    raise NotImplementedError()

  def init(self, obs1, action1, rw1, obs2):
    self.DQN.init(obs1, action1, rw1, obs2)
    
  def nextActionAndTrain(self, obs, rw):
    if len(self.obs_old) != 0:
      self.DQN.train(self.obs_old, self.action, self.rw, obs)
    self.obs_old = obs
    self.rw = rw
    self.action = self.DQN.action(obs)
    return self.action
    
  def nextAction(self, obs):
    return self.DQN.action(obs)
    
  def isTrainingFinished(self):
    return self.DQN.end()
    
  def saveAgent(self, fname):
    raise NotImplementedError()


# Special environments for testing Double DQN
   
class RiverAgent(BaseAgent):

  def makeEnvironment(self):
    env = gym.make('Riverrider-v0')
    env.reset()
    return env

class FrAgent(BaseAgent):

  def makeEnvironment(self):
    env = gym.make('Freeway-v0')
    env.reset()
    return env

class BrAgent(BaseAgent):

  def makeEnvironment(self):
    env = gym.make('Breakout-v0')
    env.reset()
    return env
