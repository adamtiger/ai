from cv4rl.cv4env import BaseEnvironment as benv
from cv4rl.cv4alg import dqn
import logging

# configure logger
logging.basicConfig(format='%(asctime)s %(message)s', level = logging.DEBUG,
                    filename = 'output.log')

# Provide an environment for Double DQN.
#   -> trains it
#   -> evaluates it
# Actions: (numbers show the next pixel)
#   0 1 2
#   3 * 4
#   5 6 7
# x: vertical, y: horizontal

class Environment:
    
    def __init__(self, args):
        d = dqn.DQN()
        d.set_params(args.actions, args.alpha, args.C, args.max_iter,
                     args.mem_size, args.exp_start, args.exp_end, 
                     args.last_fm, args.gamma)
        self.dqn = dqn.DqnAgent(d)
    
        self.base_env = benv.BaseEnvironment(args.pool_size, args.refresh_freq,
                                             args.min_rect, args.max_rect, 
                                             args.folder)
        self.rf_freq = args.refresh_freq_image
        self.eval_freq = args.eval_freq
        
    def train(self):
        
        cntr = 0
        iteration = 0
        
        while not self.dqn.is_training_finished():
            
            logging.info('Current: ' + str(iteration))
            
            if iteration % self.eval_freq == 0:
                self.evaluate()
            
            # generate new situation
            self.base_env.generate_new_situation()
            start = self.base_env.get_start_coord()
            current_coord = start
            reward = 0.0
            correct = self.base_env.get_correct()
            
            
            for idx in range(0, len(correct)-1):
                
                # crop picture around the current position (84x84)
                obs = self.base_env.crop(84, 84, current_coord[0], current_coord[1])
                
                # feed it into the dqn agent
                action = self.dqn.next_action_train(obs, reward)
                
                # move the current position according to the action
                self.__execute_action(current_coord, action)
                
                # give reward
                reward = self.__calculate_reward(correct, idx, current_coord)
                
                # save the current position  
                current_coord = correct[idx+1]

            
            # update the image pool when necessary
            cntr += 1
            if cntr > self.rf_freq:
                self.base_env.update_base_image()
                cntr = 0
        
        self.dqn.save_agent("saved_agent.hdf5")
    
    def __execute_action(self, current_coord, action):
        
        if action == 0:
            current_coord[0] = current_coord[0] - 1
            current_coord[1] = current_coord[1] - 1
        elif action == 1:
            current_coord[0] = current_coord[0] - 1
            current_coord[1] = current_coord[1] - 0
        elif action == 2:
            current_coord[0] = current_coord[0] - 1
            current_coord[1] = current_coord[1] + 1
        elif action == 3:
            current_coord[0] = current_coord[0] - 0
            current_coord[1] = current_coord[1] - 1
        elif action == 4:
            current_coord[0] = current_coord[0] - 0
            current_coord[1] = current_coord[1] + 1
        elif action == 5:
            current_coord[0] = current_coord[0] + 1
            current_coord[1] = current_coord[1] - 1
        elif action == 6:
            current_coord[0] = current_coord[0] + 1
            current_coord[1] = current_coord[1] + 0
        elif action == 7:
            current_coord[0] = current_coord[0] + 1
            current_coord[1] = current_coord[1] + 1
        else:
            raise ValueError('Unknown action!')
    
    def __calculate_reward(self, correct, idx, recom):
        
        if idx+1 == len(correct):
            return 0.0
        
        rw = 0.0
        
        x_dist1 = abs(correct[idx+1][0] - recom[0])
        y_dist1 = abs(correct[idx+1][1] - recom[1])
        
        x_dist2 = abs(correct[idx+1][0] - correct[idx][0])
        y_dist2 = abs(correct[idx+1][1] - correct[idx][1])
        
        summa = x_dist1 + y_dist1
        
        if x_dist2 + y_dist2 == 1:
            if summa == 0:
                rw = 2.0
            elif summa == 1:
                rw = 1.0
            elif x_dist1 == 1 and y_dist1 == 1:
                rw = -1.0
            elif summa == 3:
                rw = -2.0
            else:
                rw = -3.0
        else:
            if summa == 0:
                rw = 2.0
            elif summa == 1:
                rw = 1.0
            elif summa == 2:
                rw = -1.0
            elif summa == 3:
                rw = -2.0
            else:
                rw = -3.0
        
        return rw
    
    def evaluate(self):
        
        cntr = 0
        
        for episode in range(0, 20):
            
            # generate new situation
            self.base_env.generate_new_situation()
            start = self.base_env.get_start_coord()
            current_coord = start
            reward = 0.0
            correct = self.base_env.get_correct()
            
            
            for idx in range(0, len(correct)-1):
                
                # crop picture around the current position (84x84)
                obs = self.base_env.crop(84, 84, current_coord[0], current_coord[1])
                
                # feed it into the dqn agent
                action = self.dqn.next_action(obs)
                
                # move the current position according to the action
                self.__execute_action(current_coord, action)
                
                # give reward
                reward += self.__calculate_reward(correct, idx, current_coord)
                
                # save the current position  
                current_coord = correct[idx+1]

            
            # update the image pool when necessary
            cntr += 1
            if cntr > self.rf_freq:
                self.base_env.update_base_image()
                cntr = 0
                
            logging.info('Result in ' + str(episode) + ' is: ' + str(reward))