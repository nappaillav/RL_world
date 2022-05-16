import numpy as np
import scipy.signal
import torch
from torch import nn
import torch.nn.functional as F
import itertools
from torchvision.models.resnet import conv3x3
from torch.optim import Adam



def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.bn1(conv1)
        conv1 = self.relu(conv1)

        conv2 = self.conv2(x)
        conv2 = self.bn2(conv2)

        return conv1 + conv2            

class CNNActor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        h, w, c = obs_dim
        self.model = nn.Sequential(
            nn.Conv2d(c, 32, 5, 2),
            nn.ELU(),
            nn.Conv2d(32, 64, 5, 2),
            nn.ELU(),
            nn.Conv2d(64, 64, 5, 2),
            nn.ELU(),
            nn.Conv2d(64, 32, 5, 2),
            nn.ELU(),
            #nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(256, 100),
            nn.ELU(),
            nn.Linear(100, act_dim),
            nn.Tanh() # squash to [-1,1]
        )
        self.act_limit = act_limit


    def forward(self, obs):
        obs = obs.permute(0, 3, 1, 2)
        return self.act_limit * self.model(obs)

class CNNQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, fc_act_dim=50):
        super().__init__()
        h, w, c = obs_dim
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 5, 2),
            nn.ELU(),
            nn.Conv2d(32, 64, 5, 2),
            nn.ELU(),
            nn.Conv2d(64, 64, 5, 2),
            nn.ELU(),
            nn.Conv2d(64, 32, 5, 2),
            nn.ELU(),
            #nn.Dropout(0.5),
            nn.Flatten(),
        )
        self.fc_act_dim = fc_act_dim

        self.act_linear = nn.Sequential(
            nn.Linear(act_dim, 25),
            nn.ELU(),
            nn.Linear(25, fc_act_dim),
            nn.ELU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(256 + self.fc_act_dim, 100),
            nn.ELU(),
            nn.Linear(100, 1),
            nn.Sigmoid() # squash to [0,1]
        )

    def forward(self, obs, act):
        obs = obs.permute(0, 3, 1, 2)
        obs = self.conv(obs)
        act = self.act_linear(act)
        # print(obs.shape)
        q = self.linear(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)

class CNNActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()

        # obs_dim = observation_space.shape[0]
        # act_dim = action_space.shape[0]
        # act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = CNNActor(obs_dim, act_dim, act_limit)
        self.q = CNNQFunction(obs_dim, act_dim)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).squeeze(dim=0).numpy()

if __name__ == '__main__':

    ac = CNNActorCritic((80, 120, 3), 2, 1)
    # model = CNNQFunction((120, 80, 3), 2)
    model = ac.q
    inp1 = torch.randn([10, 80, 120, 3])
    inp2 = torch.randn([10, 2])
    out = model(inp1, inp2)
    print(out.shape)

    # model2 = CNNActor((120, 80, 3), 2, 1)
    model2 = ac.pi 
    out = model2(inp1)
    print(out)


