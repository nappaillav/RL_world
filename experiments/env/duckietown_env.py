# coding=utf-8
import numpy as np
from gym import spaces
from dataclasses import dataclass
from gym_duckietown.simulator import Simulator
from gym_duckietown import logger
from gym_duckietown.simulator import NotInLane

@dataclass
class DoneRewardInfo:
    done: bool
    done_why: str
    done_code: str
    reward: float

class DuckietownEnv(Simulator):
    """
    Wrapper to control the simulator using velocity and steering angle
    instead of differential drive motor velocities
    """

    def __init__(self, gain=1.0, trim=0.0, radius=0.0318, k=27.0, limit=1.0, **kwargs):
        Simulator.__init__(self, **kwargs)
        logger.info("using DuckietownEnv")

        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

        # Should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = gain

        # Directional trim adjustment
        self.trim = trim

        # Wheel radius
        self.radius = radius

        # Motor constant
        self.k = k

        # Wheel velocity limit
        self.limit = limit

        # Distance travelled
        self.prev_pos = None

        # DtRewardCollisionAvoidance
        self.prev_proximity_penalty = 0.
        self.proximity_reward = 0.

        # Proximity penalty 2
        self.proximity_reward_2 = 0.

        # position
        self.max_lp_dist = 0.05
        self.max_dev_from_target_angle_deg_narrow = 10
        self.max_dev_from_target_angle_deg_wide = 50
        self.target_angle_deg_at_edge = 45
        self.scale = 1./2.
        self.orientation_reward = 0.

    def step(self, action):
        vel, angle = action

        # Distance between the wheels
        baseline = self.unwrapped.wheel_dist

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * baseline) / self.radius
        omega_l = (vel - 0.5 * angle * baseline) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])

        obs, reward, done, info = Simulator.step(self, vels)
        
        reward_dist = self._distance_reward()
        reward_velocity = self._velocity_reward()
        reward_proximity = self._proximity_reward()
        reward_orientation = self._orientation_reward()
        

        mine = {}
        mine["k"] = self.k
        mine["gain"] = self.gain
        mine["train"] = self.trim
        mine["radius"] = self.radius
        mine["omega_r"] = omega_r
        mine["omega_l"] = omega_l
        info["DuckietownEnv"] = mine 
        info["reward_dist"] = reward_dist
        info["reward_velocity"] = reward_velocity
        info["reward_proximity"] = reward_proximity
        info["reward_orientation"] = reward_orientation
        return obs, reward, done, info


    def _distance_reward(self, lane_pos_threshold=-0.05, scale_factor=50):
        # Baseline reward is a for each step
        my_reward = 0

        # Get current position and store it for the next step
        pos = self.cur_pos
        prev_pos = self.prev_pos
        self.prev_pos = pos
        if prev_pos is None:
            return 0

        # Get the closest point on the curve at the current and previous position
        angle = self.cur_angle
        curve_point, tangent = self.closest_curve_point(pos, angle)
        prev_curve_point, prev_tangent = self.closest_curve_point(prev_pos, angle)
        if curve_point is None or prev_curve_point is None:
            logger.error("self.closest_curve_point(pos, angle) returned None!!!")
            return my_reward

        # Calculate the distance between these points (chord of the curve), curve length would be more accurate
        diff = curve_point - prev_curve_point
        dist = np.linalg.norm(diff)

        try:
            lp = self.get_lane_pos2(pos, self.unwrapped.cur_angle)
        except NotInLane:
            return my_reward

        # Dist is negative on the left side of the rignt lane center and is -0.1 on the lane center.
        # The robot is 0.13 (m) wide, to keep the whole vehicle in the right lane, dist should be > -0.1+0.13/2)=0.035
        # 0.05 is a little less conservative
        if lp.dist < lane_pos_threshold:
            return my_reward

        # Check if the agent moved in the correct direction
        if np.dot(tangent, diff) < 0:
            return my_reward

        # Reward is proportional to the distance travelled at each step
        my_reward = scale_factor * dist
        if np.isnan(my_reward):
            my_reward = 0.
            logger.error("Reward is nan!!!")
        return my_reward

    def _clip_reward(self, reward):
        if np.isnan(reward):
            reward = 0.
            logger.error("Reward is nan!!!")
        return np.clip(reward, self.clip_low, self.clip_high)

    def _velocity_reward(self, scale_factor=0.25):
        self.velocity_reward = np.max(self.wheelVels) * scale_factor
        if np.isnan(self.velocity_reward):
            self.velocity_reward = 0.
            logger.error("Velocity reward is nan, likely because the action was [nan, nan]!")
        return self.velocity_reward

    def _proximity_reward(self, scale_factor=50):
        # Proximity reward is proportional to the change of proximity penalty. Range is ~ 0 - +1.5 (empirical)
        # Moving away from an obstacle is promoted, if the robot and the obstacle are close to each other.
        proximity_penalty = self.proximity_penalty2(self.cur_pos, self.cur_angle)
        self.proximity_reward = -(self.prev_proximity_penalty - proximity_penalty) * scale_factor
        if self.proximity_reward < 0.:
            self.proximity_reward = 0.
        # logger.debug("Proximity reward: {:.3f}".format(self.proximity_reward))
        self.prev_proximity_penalty = proximity_penalty
        return self.proximity_reward

    def _proximity_reward_2(self, scale_factor=2.5):
        proximity_penalty = self.proximity_penalty2(self.cur_pos, self.cur_angle)
        self.proximity_reward_2 = proximity_penalty * scale_factor
        # logger.debug("Proximity reward: {:.3f}".format(self.proximity_reward_2))
        return self.proximity_reward_2

    def calculate_pos_angle_reward(self, lp_dist, lp_angle):
        normed_lp_dist = lp_dist / self.max_lp_dist
        target_angle = - np.clip(normed_lp_dist, -1., 1.) * self.target_angle_deg_at_edge
        reward_narrow = 0.5 + 0.5 * self.leaky_cosine(
            np.pi * (target_angle - lp_angle) / self.max_dev_from_target_angle_deg_narrow)
        reward_wide = 0.5 + 0.5 * self.leaky_cosine(
            np.pi * (target_angle - lp_angle) / self.max_dev_from_target_angle_deg_wide)
        return reward_narrow, reward_wide    

    @staticmethod
    def leaky_cosine(x):
        slope = 0.05
        if np.abs(x) < np.pi:
            return np.cos(x)
        else:
            return -1. - slope * (np.abs(x)-np.pi)

    @staticmethod
    def gaussian(x, mu=0., sig=1.):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def _orientation_reward(self):
        pos = self.cur_pos
        angle = self.cur_angle
        try:
            lp = self.get_lane_pos2(pos, angle)
            # print("Dist: {:3.2f} | DotDir: {:3.2f} | Angle_deg: {:3.2f}". format(lp.dist, lp.dot_dir, lp.angle_deg))
        except NotInLane:
            return -10.

        # print("Dist: {:3.2f} | Angle_deg: {:3.2f}".format(normed_lp_dist, normed_lp_angle))
        angle_narrow_reward, angle_wide_reward = self.calculate_pos_angle_reward(lp.dist, lp.angle_deg)
        # logger.debug("Angle Narrow: {:4.3f} | Angle Wide: {:4.3f} ".format(angle_narrow_reward, angle_wide_reward))
        self.orientation_reward = self.scale * (angle_narrow_reward + angle_wide_reward)

        early_termination_penalty = 0.

        # If the robot leaves the track or collides with an other object it receives a penalty
        # if reward <= -1000.:  # Gym Duckietown gives -1000 for this
        #     early_termination_penalty = -10.
        
        return self.orientation_reward + early_termination_penalty

    def _reward_reset():
        self.velocity_reward = 0
        self.prev_pos = 0
        self.prev_proximity_penalty = 0.
        self.proximity_reward = 0.  
        self.proximity_reward_2 = 0.
        self.orientation_reward = 0.

    def _compute_done_reward(self) -> DoneRewardInfo:
        # If the agent is not in a valid pose (on drivable tiles)
        if not self._valid_pose(self.cur_pos, self.cur_angle):
            msg = "Stopping the simulator because we are at an invalid pose."
            # logger.info(msg)
            reward = REWARD_INVALID_POSE
            done_code = "invalid-pose"
            done = True
        # If the maximum time step count is reached
        elif self.step_count >= self.max_steps:
            msg = "Stopping the simulator because we reached max_steps = %s" % self.max_steps
            # logger.info(msg)
            done = True
            reward = 0
            done_code = "max-steps-reached"
        else:
            done = False
            reward = self.compute_reward(self.cur_pos, self.cur_angle, self.robot_speed)
            msg = ""
            done_code = "in-progress"
        return DoneRewardInfo(done=done, done_why=msg, reward=reward, done_code=done_code)

