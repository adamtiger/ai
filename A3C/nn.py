
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):


    def __init__(self, action_num):
    
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fully_conn = nn.Linear(1024, 512)

        self.actor = nn.Linear(512, action_num)
        self.critic = nn.Linear(512, 1)

    def forward(self, state):

        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fully_conn(x))

        return self.actor(x), self.critic(x)


