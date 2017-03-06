import environment
import argparse 


# RUN THE ALGORITHM

<<<<<<< HEAD
actions = 6
batch_size = 32
alpha = 0.0005
C = 5000
max_iter =200000#10000000
mem_size = 10000#1000000
exp_start = 1.0
exp_end = 0.05
last_fm = 10000
gamma = 0.9999
=======
# Parameter settings
parser = argparse.ArgumentParser(description='DQN algorithm')
>>>>>>> 8b91c4f277b11da7fcde04fa3c13456b3ae3d211

parser.add_argument('--atari-env', default='Breakout-v0', metavar='S',
        help='the name of the Atari environment (default:Breakout-v0)')
parser.add_argument('--lr', type=float, default=0.00025, metavar='F',
        help='the learning rate (default:0.00025)')
parser.add_argument('--C', type=int, default=5000, metavar='N',
        help='the update frequency of the neural network (default:5000)')
parser.add_argument('--max-iter', type=int, default=500000, metavar='N',
        help='the maximum number of steps (default:500000)')
parser.add_argument('--mem-size', type=int, default=50000, metavar='N',
        help='the capacity of the experience replay (default:50000)')
parser.add_argument('--exp-start', type=float, default=1.0, metavar='F',
        help='the number of the agents (default:4)')
parser.add_argument('--exp-end', type=float, default=0.1, metavar='F',
        help='the number of individual steps (default:10000)')
parser.add_argument('--last-fm', type=int, default=100000, metavar='N',
        help='the exploration gradually decreasing, it has exp_end value after last_fm steps (default:100000)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='F',
        help='the discounting factor (default:0.99)')
parser.add_argument('--eval-freq', type=int, default=50000, metavar='N',
        help='the number of steps between two evaluation of the agent (default:50000)')
parser.add_argument('--eval-num', type=int, default=10, metavar='N',
        help='the number of evaluations at once (default:10)')
parser.add_argument('--init-replay-size', type=int, default=10000, metavar='N',
        help='the initial number of frames in the experience replay memory (default:10000)')

args = parser.parse_args()

env = environment.Environment(args)

fname = "files/agent.hdf5"
env.train(fname)
