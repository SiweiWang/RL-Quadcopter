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

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """ Reward function """
        # https:/stackoverflow.com/questions/41723209/distance-between-2-points-in-3d-for-a-big-array?answertab=active#tab-top
        # http://mathworld.wolfram.com/VectorNorm.html
        # reward = np.linalg.norm(self.sim.pose[:3] - self.target_pos[:3])
        # reward = 1.5 - self.sigmoid(reward)

        # http://reference.wolfram.com/language/ref/Tanh.html
        # reward z is positive when above the target above and negative when under
        # reward_z = 0.0

        # if (self.sim.pose[2] -  self.target_pos[2] > 0.5):
        #     reward_z = - (self.sigmoid(self.sim.pose[2] + self.sim.v [2]))

        # if (self.target_pos[2] - self.sim.pose[2] > 0.5 ):
        #     reward_z = self.sigmoid((self.sim.pose[2] + 0.1 * self.sim.v [2]))

        # if ( abs(self.sim.pose[2] -  self.target_pos[2]) < 0.5):
        #     reward += 1.0
        # # Penalize crash
        # if done and self.sim.time < self.sim.runtime:
        #     return -5

        # # # Reward reaching target height
        # if self.sim.pose[2] >= self.target_pos[2] :
        #     return 1

        # # use tanh to scale the reward between -1 and 1

        # reward = 0.8 * np.tanh(self.sim.v[2]) - 0.2 * np.tanh(self.sim.v[0]) - 0.2 * np.tanh(self.sim.v[1]) 
        # -  0.5 * np.tanh(np.linalg.norm(self.sim.pose[:3] - self.target_pos[:3])) 

        # reward = np.tanh(1 - 0.0003*(abs(self.sim.pose[:3] - self.target_pos[:3])).sum())

        # reward = -min(abs(self.target_pos[2] - self.sim.pose[2] ),10.0)  # reward = zero for matching target z, -ve as you go farther, upto -20

        # if self.sim.pose[2]  >= self.target_pos[2]:  # agent has crossed the target height
        #     reward += 10.0  # bonus reward
        #     done = True

        # elif done:   # agent has run out of time
        #     reward -= 10.0  # extra penalty
        #     done = True

        reward = - 0.5*(abs(self.sim.pose[:2] - self.target_pos[:2])).sum()
        reward += - 2.0*min(abs(self.sim.pose[2] - self.target_pos[2]), 20.0)
        reward +=  4.0*self.sim.v[2]
        
        reward += - 3.0*(abs(self.sim.pose[3:6])).sum()
        reward += - 3.0*(abs(self.sim.angular_v[:3])).sum()

        done = False
        if(self.sim.pose[2] >= self.target_pos[2]):
            reward += 50.0
            done = True
        return reward, done

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward, height_done = self.get_reward()
            pose_all.append(self.sim.pose)

            # terminate if done
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