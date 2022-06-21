import torch
import numpy as np

from rl_world.utils.pytorch_util import from_numpy, to_numpy

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

    path += '.pt' if torch_saving else '.npy'
    if torch_saving:
        torch.save(buffer, path)
    else:
        np.save(path, buffer)

    print("Buffer saved at location: {} (torch saving: {})".format(path, torch_saving))
    return

def load_buffer(path="./buffer_save", torch_loading=False, device=torch.device('cpu')):
    path += '.pt' if torch_loading else '.npy'
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