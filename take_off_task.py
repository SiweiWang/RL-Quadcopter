import numpy as np
from physics_sim import PhysicsSim
import math

class Takeoff_Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # distance factor
        self.xy_distance_factor = 0.5
        self.z_distance_factor = 2.0
        self.z_distance_max = 15.0

        self.z_velocity_factor = 4.0

        # angle factor
        self.angle_pentalty = 3.0
        self.angle_velocity_pentalty = 3.0

        # Reward when reach the target hight
        self.finish_reward = 100.0

        # Reward scale factor -- the total reward factor that scales the final reward. 
        # This does not change the propertion of each part of rewards
        self.reward_scale = 1

        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """ Reward function """
        reward = - self.xy_distance_factor * (abs(self.sim.pose[:2] - self.target_pos[:2])).sum()
        reward += - self.z_distance_factor * min(abs(self.sim.pose[2] - self.target_pos[2]), self.z_distance_max)
        reward +=  self.z_velocity_factor * self.sim.v[2]

        reward += - self.angle_pentalty * (abs(self.sim.pose[3:6])).sum() - self.angle_velocity_pentalty * (abs(self.sim.angular_v[:3])).sum()

        # Terminate if reach the height
        if(self.sim.pose[2] >= self.target_pos[2]):
            reward += self.finish_reward 
            done = True
        else:
            done = False

        return self.reward_scale*reward, done

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward, height_done = self.get_reward()
            pose_all.append(self.sim.pose)

            # terminate if reach the target height
            if height_done:
                done = True

        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))