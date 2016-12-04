# This is the abstract class for al of the agents.

class BaseAgent:
  
  def makeEnvironment(self):
    raise NotImplementedError()
    
  def nextActionAndTrain(self, obs, rw, done, inf):
    raise NotImplementedError()
    
  def nextAction(self, obs, rw, done, inf):
    raise NotImplementedError()
    
  def isTrainingFinished(self):
    raise NotImplementedError()
    
  def saveAgent(self, fname):
    raise NotImplementedError()
    
  