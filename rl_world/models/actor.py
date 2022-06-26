import torch.nn as nn

class CNNGaussianActor(nn.Module):

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # log_std = -0.5 * np.ones(act_dim, dtype=np.float32) 
        # self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        h, w, c = obs_dim
        # print(act_dim)
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
            nn.Linear(100, act_dim[0]),
            nn.Tanh() # squash to [-1,1]
        )

        # self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs):
        # print(obs.shape)
        obs = obs.permute(0, 3, 1, 2)
        return self.model(obs)

class CNNGaussianStudent(nn.Module):

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        h, w, c = obs_dim
        # print(act_dim)
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
            nn.Linear(100, act_dim[0]),
            nn.Tanh() # squash to [-1,1]
        )
        
    def forward(self, obs):
        # print(obs.shape)
        obs = obs.permute(0, 3, 1, 2)
        return self.model(obs)