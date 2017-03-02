# -*- coding: utf-8 -*-

import a3c
import nn

import argparse 
import torch
import torch.multiprocessing as mp


# Parameter settings
parser = argparse.ArgumentParser(descritpion='A3C algorithm')

parser.add_argument('--num-actions', type=int, default=6, metavar='N',
        help='the number of possible actions available (default:6, Breakout)')
parser.add_argument('--lr', type=float, default=0.00025, metavar='F',
        help='the learning rate (default:0.00025)')
parser.add_argument('--num-agents', type=int, default=4, metavar='N',
        help='the number of the agents (default:4)')
parser.add_argument('--max-it', type=int, default=10000, metavar='N',
        help='the number of individual steps (default:10000)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='F',
        help='the discounting factor (default:0.99)')
parser.add_argument('--atari-env', default='Breakout-v0', metavar='S',
        help='the Atari environment (default:Breakout-v0)')

# Instantiation:

if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(1)

    shared_nn = a3c.shared_nn()
    shared_nn.shared_memory()

    processes = []
    for agent in range(args.num_agents):
        a = mp.Process(target=a3c.a3c, args=(args, shared_nn))
        a.start()
        processes.append(a)
    for p in processes:
        p.join()




