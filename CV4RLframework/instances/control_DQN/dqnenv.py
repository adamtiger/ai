from cv4rl.cv4env import BaseEnvironment as benv
from cv4rl.cv4alg import dqn


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
        
    def train(self):
        
        cntr = 0
        
        while not self.dqn.is_training_finished():
            
            # generate new situation
            self.base_env.generate_new_situation()
            start = self.base_env.get_start_coord()
            current_coord = start
            reward = 0.0
            correct = self.base_env.get_correct()
            
            
            for point in correct:
                
                # crop picture around the current position (84x84)
                obs = self.base_env.crop(84, 84, current_coord[0], current_coord[1])
                
            
                # feed it into the dqn agent
                action = self.dqn.next_action_train(obs, reward)
                
                # move the current position according to the action
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
                
                # give reward
                
                reward = 0.0 # calculate this
                
                # save the current position  
                current_coord = point

            
            # update the image pool when necessary
            cntr += 1
            if cntr > self.rf_freq:
                self.base_env.update_base_image()
            