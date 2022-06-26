import torch
import numpy as np

from rl_world.utils.pytorch_utils import to_numpy

def save_buffer(obs, actions, logprobs, rewards, dones, values, path="./buffer_save", torch_saving=False):

    # Handle the buffer from the GPU
    if not torch_saving:
        obs = to_numpy(obs)
        actions = to_numpy(actions)
        logprobs = to_numpy(logprobs)
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)
        values = to_numpy(values)
    else:
        obs = obs.cpu().detach()
        actions = actions.cpu().detach()
        logprobs = logprobs.cpu().detach()
        rewards = rewards.cpu().detach()
        dones = dones.cpu().detach()
        values = values.cpu().detach()    
    
    buffer = {
        'obs': obs,
        'actions': actions,
        'logprobs': logprobs,
        'rewards': rewards,
        'dones': dones,
        'values': values
    }

    if torch_saving:
        torch.save(buffer, path)
    else:
        np.save(path, buffer)

    print("Buffer saved at location: {} (torch saving: {})".format(path, torch_saving))
    return

def load_buffer(path="./buffer_save", torch_loading=False, device=torch.device('cpu')):
    buffer = torch.load(path) if torch_loading else np.load(path)

    obs = buffer['obs']
    actions = buffer['actions']
    logprobs = buffer['logprobs']
    rewards = buffer['rewards']  
    dones = buffer['dones']
    values = buffer['values']

    if not torch_loading:
        obs = torch.tensor(obs).float()
        actions = torch.tensor(actions).float()
        logprobs = torch.tensor(logprobs).float()
        rewards = torch.tensor(rewards).float()
        dones = torch.tensor(dones).float()
        values = torch.tensor(values).float()

    print("Buffer loaded from location: {} (torch loading: {})".format(path, torch_loading))
    return obs.to(device), actions.to(device), logprobs.to(device), rewards.to(device), dones.to(device), values.to(device)


class ReplayBuffer:

    def __init__(self, capacity=2048, device=torch.device('cpu')):

        self.capacity = capacity
        self.device = device
        self.obs = np.zeros(capacity)
        self.actions = np.zeros(capacity)
        self.logprobs = np.zeros(capacity)
        self.rewards = np.zeros(capacity)
        self.dones = np.zeros(capacity)
        self.values = np.zeros(capacity)
        self.pointer = 0
        self.size = 0
        
        return

    def load(self, path, load_as_torch=False):
        self.obs, self.actions, self.logprobs, self.rewards, self.dones, self.values = load_buffer(path, torch_loading=load_as_torch, device=self.device)
        return

    def save(self, path, save_as_torch):
        save_buffer(self.obs, self.actions, self.logprobs, self.rewards, self.dones, self.values, path=path, torch_saving=save_as_torch)
        return

    def update(self, idx, obs, ac, logprob, reward, done, value):
        self.obs[idx] = obs
        self.actions[idx] = ac
        self.logprobs[idx] = logprob
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.values[idx] = value
        return

    def add(self, obs, ac, logprob, reward, done, value):

        self.obs[self.pointer] = obs
        self.actions[self.pointer] = ac
        self.logprobs[self.pointer] = logprob
        self.rewards[self.pointer] = reward
        self.dones[self.pointer] = done
        self.values[self.pointer] = value
        self.pointer = (self.pointer+1) % self.capacity
        self.size = min(self.size+1, self.capacity)
        return

    def sample(self, batch_size):

        if self.size <= batch_size:
            return (self.obs[:self.size],
                   self.actions[:self.size],
                   self.logprobs[:self.size],
                   self.rewards[:self.size],
                   self.dones[:self.size],
                   self.values[:self.size])
        else:
            idx = list(range(self.size))
            idx = np.random.permutation(idx)
            return (self.obs[:batch_size],
                   self.actions[:batch_size],
                   self.logprobs[:batch_size],
                   self.rewards[:batch_size],
                   self.dones[:batch_size],
                   self.values[:batch_size])