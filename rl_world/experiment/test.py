# test 

from PIL import Image
import argparse
import sys
import random
import gym
import numpy as np
import pyglet
from pyglet.window import key

from rl_world.env.duckietown_env import DuckietownEnv
from rl_world.algo.ppo import Agent
import torch
from rl_world.utils.pytorch_utils import from_numpy, to_numpy
# from experiments.utils import save_img

import cv2
import os

video_name = './output/video.avi'
max_step = 1000
weight_path = '/home/proton/course/project/RL_world/rl_world/algo/model_agent_20.pt'
fps = 25

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default=None)
parser.add_argument("--map-name", default="udem1")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument("--camera_rand", default=False, action="store_true")
parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
parser.add_argument("--dynamics_rand", action="store_true", help="enable dynamics randomization")
parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
parser.add_argument("--seed", default=random.randint(0, 1000), type=int, help="seed")
args = parser.parse_args()


env = DuckietownEnv(
    seed=args.seed,
    map_name=args.map_name,
    draw_curve=args.draw_curve,
    draw_bbox=args.draw_bbox,
    domain_rand=args.domain_rand,
    frame_skip=args.frame_skip,
    distortion=args.distortion,
    camera_rand=args.camera_rand,
    dynamics_rand=args.dynamics_rand,
)


obs = env._obs_wrapper(env.reset())
height, width, layers = obs.shape
video = cv2.VideoWriter(video_name, 0, fps, (width,height))
video.write(cv2.cvtColor((obs*255).astype('uint8'), cv2.COLOR_BGR2RGB))
obs = from_numpy(obs).unsqueeze(dim=0)
env.render()


agent = torch.load(weight_path, map_location=torch.device('cpu'))
for i in range(max_step):
    action, _, _, _ = agent.get_action_and_value(obs)
    action = to_numpy(action)[0]
    obs, reward, done, info = env.step(action)
    video.write(cv2.cvtColor((obs*255).astype('uint8'), cv2.COLOR_BGR2RGB))
    obs = from_numpy(obs).unsqueeze(dim=0)
    env.render()
    if done == True:
        break


cv2.destroyAllWindows()
video.release()