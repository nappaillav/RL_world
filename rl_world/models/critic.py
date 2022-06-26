import torch
import torch.nn as nn

class CNNValueFunction(nn.Module):
    def __init__(self, obs_dim):
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

        self.linear = nn.Sequential(
            nn.Linear(256, 100),
            nn.ELU(),
            nn.Linear(100, 1),
            nn.Sigmoid() # squash to [0,1]
        )

    def forward(self, obs):
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(dim=0)
        obs = obs.permute(0, 3, 1, 2)
        obs = self.conv(obs)
        value = self.linear(obs)
        return torch.squeeze(value, -1)

