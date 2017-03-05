import gym

class Agent:

  def __init__(self, dqn):
      self.DQN = dqn
      self.action = 0
      self.rw = 0
      self.atari_name = ""
      self.obs_old = []
  
  def makeEnvironment(self, atari_game_name):
      self.atari_name = atari_game_name
      env = gym.make(atari_game_name)
      env.reset()
      return env

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
      return self.DQN.action_nogreedy(obs)
    
  def isTrainingFinished(self):
      return self.DQN.end()
    
  def saveAgent(self, fname):
      self.DQN.save(fname)

  def loadAgent(self, fname):
      self.DQN.load(fname)