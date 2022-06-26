from gym_duckietown import logger
from rl_world.utils.replaybuffer import ReplayBuffer
from rl_world.utils.model_util import CNNActorCritic
from rl_world.utils.pytorch_utils import from_numpy, to_numpy
from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import time
import cv2
import wandb


MAX_EP_LEN = 2000

class ddpg:
    def __init__(self,env, test_env, actor_critic, obs_shape, ac_kwargs=dict(), seed=0, 
         steps_per_epoch=2000, epochs=100, replay_size=int(10000), gamma=0.99, 
         polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=2000, 
         update_after=2000, update_every=100, act_noise=0.1, num_test_episodes=10, 
         max_ep_len=500, logger_kwargs=dict(), save_freq=1, device='cpu', wandb_logging=False):

        self.wandb_logging = wandb_logging

        self.env = env
        self.test_env = test_env
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.gamma = gamma

        self.polyak=polyak
        self.pi_lr=pi_lr
        self.q_lr=q_lr 
        self.start_steps=start_steps

        self.update_after=update_after 
        self.update_every=update_every
        self.act_noise=act_noise 
        self.num_test_episodes=num_test_episodes

        self.max_ep_len=max_ep_len
        self.save_freq=1 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        torch.manual_seed(seed)
        np.random.seed(seed)  

        
        self.obs_dim = inp_shape
        self.batch_size = batch_size
        self.act_dim = self.env.action_space.shape[0]
        self.act_limit = self.env.action_space.high[0]
        logger.debug("#### {} -- {} -- {} ####".format(self.obs_dim, self.act_dim, self.act_limit))
        self.device = device
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size)
        self.actor_critic = actor_critic
        self.actor_critic_target = deepcopy(self.actor_critic)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.actor_critic_target.parameters():
            p.requires_grad = False

        self.pi_optimizer = Adam(self.actor_critic.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.actor_critic.q.parameters(), lr=q_lr)

        if self.wandb_logging:
            # init wandb project/run
            # add more tags
            wandb.init(project="duckietown", tags=['ddpg', 'test_experiment'], entity="valudem", config = vars(self))
        
    def critic_loss(self, data):

        # transfer to device 

        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = self.actor_critic.q(o,a) # qnet

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.actor_critic_target.q(o2, self.actor_critic_target.pi(o2))
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # Useful info for logging
        loss_info = loss_q.detach().numpy().mean()

        return loss_q, loss_info

    def actor_loss(self, data):
        o = data['obs']
        q_pi = self.actor_critic.q(o, self.actor_critic.pi(o))
        loss_info = q_pi.detach().numpy().mean()
        return -q_pi.mean(), loss_info


    def update(self, data):
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.critic_loss(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in self.actor_critic.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.actor_loss(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.actor_critic.q.parameters():
            p.requires_grad = True

        # Record things
        # logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)
        if self.wandb_logging:
            print('---------------- Check -----------------')
            wandb.log({"q_loss": q_info, "actor_loss": pi_info})

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.actor_critic.parameters(), self.actor_critic_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
    
    def get_action(self, obs, noise):
        obs = from_numpy(obs)
        obs = obs.unsqueeze(dim=0)
        # replaced torch.as_tensor(obs, dtype=torch.float32) to from numpy 
        a = self.actor_critic.act(obs)
        a += noise * np.random.randn(act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    def test_agent(self):
        for j in range(self.num_test_episodes):
            
            o, d, ep_ret, ep_len = self.test_env._obs_wrapper(self.test_env.reset()), False, 0, 0
            self.test_env.render()
            while not(d or (ep_len == MAX_EP_LEN)):
                # Take deterministic actions at test time (noise_scale=0)
                # print(o.shape)
                a = self.get_action(o, 0)
                o, r, d, _ = self.test_env.step(a)
                ep_ret += r
                ep_len += 1
                self.test_env.render()
            
            logger.info("Episode return : {} == Episode Length : {}".format(ep_ret, ep_len))

            # logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
            self.test_env.close()


    def train(self):
        
        total_steps = self.steps_per_epoch * self.epochs
        start_time = time.time()
        o, ep_ret, ep_len = self.env._obs_wrapper(self.env.reset()), 0, 0
        # cv2.imwrite('test.png', o*255)
        # print(o.shape)
        
        for t in range(total_steps):
            if t%500 == 0:
                logger.info('Time step : {}'.format(t))

            if t > self.start_steps:
                a = self.get_action(o, self.act_noise)
            else:
                a = self.env.action_space.sample()
            # print(a)
            o2, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1

            d = False if ep_len==MAX_EP_LEN else d
            self.replay_buffer.store(o, a, r, o2, d)
            o = o2
            # print(o.shape)
            if d or (ep_len == MAX_EP_LEN):
                logger.info("Episode return : {} == Episode Length : {}".format(ep_ret, ep_len))
                o, ep_ret, ep_len = self.env._obs_wrapper(self.env.reset()), 0, 0

            if t >= self.update_after and t % self.update_every == 0:
                for c in range(self.update_every):
                    if c%10 == 0:
                    #     # logger.info('update count : {}'.format(c))
                        batch = self.replay_buffer.sample_batch(self.batch_size)
                        self.update(data=batch)

            # End of epoch handling
            if (t+1) % self.steps_per_epoch == 0:
                self.epochs = (t+1) // self.steps_per_epoch

                # Save model

                # if (self.epoch % self.save_freq == 0) or (epoch == epochs):
                #     logger.save_state({'env': env}, None)

                # Test the performance of the deterministic version of the agent.
                logger.info("Testing....")

                self.test_agent()
                
                # logger.info("Episode return : {} == Episode Length : {}".format(ep_ret, ep_len))

                # Log info about epoch
                # logger.log_tabular('Epoch', epoch)
                # logger.log_tabular('EpRet', with_min_and_max=True)
                # logger.log_tabular('TestEpRet', with_min_and_max=True)
                # logger.log_tabular('EpLen', average_only=True)
                # logger.log_tabular('TestEpLen', average_only=True)
                # logger.log_tabular('TotalEnvInteracts', t)
                # logger.log_tabular('QVals', with_min_and_max=True)
                # logger.log_tabular('LossPi', average_only=True)
                # logger.log_tabular('LossQ', average_only=True)
                # logger.log_tabular('Time', time.time()-start_time)
                # logger.dump_tabular()




if __name__ == '__main__':
    from rl_world.env.duckietown_env import DuckietownEnv
    inp_shape, act_dim, act_lim = (80, 120, 3), 2, 1
    env = DuckietownEnv(
        seed=0,
        map_name='udem1',
        frame_skip=1,
        distortion=False,
        camera_rand=False,
    )

    test_env = DuckietownEnv(
        seed=0,
        map_name='udem1',
        frame_skip=1,
        distortion=False,
        camera_rand=False,
    )

    ac = CNNActorCritic(inp_shape, act_dim, act_lim)

    agent = ddpg(env, test_env, ac, inp_shape, wandb_logging=True)
    agent.train()


