import argparse 
import predenv as env

# RUN THE ALGORITHM

# Parameter settings
parser = argparse.ArgumentParser(description='cv4rl framework prediction based approach')

parser.add_argument('--pool-size', type=int, default=2, metavar='N',
        help='the number of pictures in the pool at a time')
parser.add_argument('--refresh-freq-image', type=int, default=10, metavar='N',
        help='the number of times the same image is used')
parser.add_argument('--refresh-freq', type=int, default=50, metavar='N',
        help='the number of times new images are choosen from the pool before refresh the pool')
parser.add_argument('--folder', default="IMAGES", metavar='S',
        help='the name of the folder which containes the images')
parser.add_argument('--C', type=int, default=100, metavar='N',
        help='evaluation frequency')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
        help='batch size during training')
parser.add_argument('--num-iter', type=int, default=1000, metavar='N',
        help='the number of iteration until the end of training')
parser.add_argument('--alpha', type=float, default=0.001, metavar='F',
        help='learning rate')


args = parser.parse_args()

env = env.Environment(args)
env.train()