import argparse 
import dqnenv

# RUN THE ALGORITHM

# Parameter settings
parser = argparse.ArgumentParser(description='cv4rl framework DQN algorithm')

parser.add_argument('--pool-size', type=int, default=2, metavar='N',
        help='the number of pictures in the pool at a time')
parser.add_argument('--refresh-freq-image', type=int, default=10, metavar='N',
        help='the number of times the same image is used')
parser.add_argument('--refresh-freq', type=int, default=50, metavar='N',
        help='the number of times new images are choosen from the pool before refresh the pool')
parser.add_argument('--min-rect', type=int, default=50, metavar='N',
        help='the minimum size of the generated rectangular')
parser.add_argument('--max-rect', type=int, default=150, metavar='N',
        help='the minimum size of the generated rectangular')
parser.add_argument('--folder', default="IMAGES", metavar='S',
        help='the name of the folder which containes the images')

parser.add_argument('--actions', type=int, default=8, metavar='N',
        help='the number of possible actions for the agent')
parser.add_argument('--lr', type=float, default=0.00025, metavar='F',
        help='the learning rate (default:0.00025)')
parser.add_argument('--C', type=int, default=5000, metavar='N',
        help='the update frequency of the neural network (default:5000)')
parser.add_argument('--max-iter', type=int, default=10000, metavar='N',
        help='the number of iterations during training')
parser.add_argument('--mem-size', type=int, default=10000, metavar='N',
        help='the number of experiences in experience replay')

parser.add_argument('--exp-start', type=float, default=1.0, metavar='F',
        help='the number of the agents (default:4)')
parser.add_argument('--exp-end', type=float, default=0.1, metavar='F',
        help='the number of individual steps (default:10000)')
parser.add_argument('--last-fm', type=int, default=100000, metavar='N',
        help='the exploration gradually decreasing, it has exp_end value after last_fm steps (default:100000)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='F',
        help='the discounting factor (default:0.99)')



args = parser.parse_args()

env = dqnenv.Environment(args)
env.train()