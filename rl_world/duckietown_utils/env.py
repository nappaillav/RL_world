import logging

from duckietown_utils.wrappers.observation_wrappers import *
from duckietown_utils.wrappers.action_wrappers import *

logger = logging.getLogger(__name__)

MAPSETS = {'multimap1': ['_custom_technical_floor', '_huge_C_floor', '_huge_V_floor', '_plus_floor',
                         'small_loop', 'small_loop_cw', 'loop_empty'],
           'multimap2': ['_custom_technical_floor', '_custom_technical_grass', 'udem1', 'zigzag_dists',
                         'loop_dyn_duckiebots'],
           'multimap_lfv': ['_custom_technical_floor_lfv', 'loop_dyn_duckiebots', 'loop_obstacles', 'loop_pedestrians'],
           'multimap_lfv_dyn_duckiebots': ['_loop_dyn_duckiebots_inner', '_loop_dyn_duckiebots_outer'],
           'multimap_lfv_duckiebots': ['_loop_duckiebots', '_loop_dyn_duckiebots_inner', '_loop_dyn_duckiebots_outer'],
           'multimap_aido5': ['LF-norm-loop', 'LF-norm-small_loop', 'LF-norm-zigzag', 'LF-norm-techtrack',
                              '_custom_technical_floor', '_huge_C_floor', '_huge_V_floor', '_plus_floor',
                              'small_loop', 'small_loop_cw', 'loop_empty'
                              ],
           }

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


def launch_and_wrap_env(env_config):
    try:
        env_id = env_config.worker_index  # config is passed by rllib
    except AttributeError as err:
        logger.warning(err)
        env_id = 0

    if 'multimap' in env_config["training_map"]:
        mapset = MAPSETS[env_config["training_map"]]
        map_name_single_env = mapset[env_id % len(mapset)]
    else:
        map_name_single_env = env_config["training_map"]
        
    from gym_duckietown.simulator import Simulator
    
    env = Simulator(
        seed=1234,  # random seed
        map_name=map_name_single_env,
        max_steps=500,
        domain_rand=env_config["domain_rand"],
        dynamics_rand=env_config["dynamics_rand"],
        camera_rand=env_config["camera_rand"],
        camera_width=CAMERA_WIDTH,
        camera_height=CAMERA_HEIGHT,
        accept_start_angle_deg=env_config["accepted_start_angle_deg"],
        full_transparency=True,
        distortion=env_config["distortion"],
        frame_rate=env_config["simulation_framerate"],
        frame_skip=env_config["frame_skip"],
        # robot_speed=robot_speed
    )
    env = wrap_env(env_config, env)
    return env


def resolve_multimap_name(training_map_conf, env_id):
    if 'multimap' in training_map_conf:
        mapset = MAPSETS[training_map_conf]
        map_name_single_env = mapset[env_id % len(mapset)]
    else:
        map_name_single_env = training_map_conf
    return map_name_single_env


def wrap_env(env_config: dict, env=None):
    if env is None:
        # Create a dummy Duckietown-like env if None was passed. This is mainly necessary to easily run
        # dts challenges evaluate
        env = DummyDuckietownGymLikeEnv()

    # Observation wrappers
    if env_config["crop_image_top"]:
        env = ClipImageWrapper(env, top_margin_divider=env_config["top_crop_divider"])
    if env_config.get("grayscale_image", False):
        env = RGB2GrayscaleWrapper(env)
    env = ResizeWrapper(env, shape=env_config["resized_input_shape"])
    if env_config["frame_stacking"]:
        env = ObservationBufferWrapper(env, obs_buffer_depth=env_config["frame_stacking_depth"])
    if env_config["mode"] in ['train', 'debug'] and env_config['motion_blur']:
        env = MotionBlurWrapper(env)
    env = NormalizeWrapper(env)

    # Action wrappers
    if env_config["action_type"] == 'discrete':
        env = DiscreteWrapper(env)
    elif 'heading' in env_config["action_type"]:
        env = Heading2WheelVelsWrapper(env, env_config["action_type"])
    elif env_config["action_type"] == 'leftright_braking':
        env = LeftRightBraking2WheelVelsWrapper(env)

    # Reward wrappers
    if env_config['mode'] == 'train':
        from duckietown_utils.wrappers.reward_wrappers import DtRewardPosAngle, DtRewardWrapperDistanceTravelled, DtRewardClipperWrapper
        if env_config["reward_function"] in ['Posangle', 'posangle']:
            env = DtRewardPosAngle(env)
        elif env_config["reward_function"] == 'lane_distance':
            env = DtRewardWrapperDistanceTravelled(env)
        elif env_config["reward_function"] == 'default_clipped':
            env = DtRewardClipperWrapper(env, 2, -2)
        else:  # Also env_config['mode'] == 'default'
            logger.warning("Default Gym Duckietown reward used")

    return env


class DummyDuckietownGymLikeEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(CAMERA_HEIGHT, CAMERA_WIDTH, 3),
            dtype=np.uint8
        )
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32
        )

    def reset(self):
        logger.warning("Dummy Duckietown Gym reset() called!")
        return np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3))

    def step(self, action):
        logger.warning("Dummy Duckietown Gym step() called!")
        obs = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3))
        reward = 0.0
        done = True
        info = {}
        return obs, reward, done, info


def get_wrappers(wrapped_env):
    obs_wrappers = []
    action_wrappers = []
    reward_wrappers = []
    orig_env = wrapped_env
    while not isinstance(orig_env, type(wrapped_env.unwrapped)):
        if isinstance(orig_env, gym.ObservationWrapper):
            obs_wrappers.append(orig_env)
        elif isinstance(orig_env, gym.ActionWrapper):
            action_wrappers.append(orig_env)
        elif isinstance(orig_env, gym.ActionWrapper):
            reward_wrappers.append(orig_env)
        elif isinstance(orig_env, gym.Wrapper):
            None
        else:
            assert False, ("[duckietown_utils.env.get_wrappers] - {} Wrapper type is none of these:"
                           " gym.ObservationWrapper, gym.ActionWrapper, gym.ActionWrapper".format(orig_env))
        orig_env = orig_env.env

    return obs_wrappers[::-1], action_wrappers[::-1], reward_wrappers[::-1]


if __name__ == "__main__":
    # execute only if run as a script to test some functionality
    from config.args import get_common_args
    args = get_common_args()
    args = args.parse_args()
    dummy_env = wrap_env(args, add_reward_wrappers=False)
    obs_wrappers, action_wrappers, reward_wrappers = get_wrappers(dummy_env)
    print("Observation wrappers")
    print(obs_wrappers)
    print("Action wrappers")
    print(action_wrappers)
    print("Reward wrappers")
    print(reward_wrappers)
