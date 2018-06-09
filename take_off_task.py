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

        # This value should be between 0 and 1, which indicate how much we want to penalize the angle. 0 mean we don't penalize angle
        self.penalty_angle_factor = 0.9
    def get_reward(self):
        """ Reward function """

        # https:/stackoverflow.com/questions/41723209/distance-between-2-points-in-3d-for-a-big-array?answertab=active#tab-top
        # http://mathworld.wolfram.com/VectorNorm.html
        reward = np.linalg.norm(self.sim.pose[:3] - self.target_pos[:3])

        reward = 0.1 - self.sigmoid(reward)

        # print ("reward")
        # print (reward)
        done = False
        # If we are higher, consider it as finish and give a positive finish reward
        # if (self.sim.pose[2] > self.target_pos[2]):
        #     reward += 3.00
        #     done = True
        # if we hit the ground, give a negative finish reward
        # if (self.sim.pose[0] < 0.0 or self.sim.pose[1] < 0.0 or self.sim.pose[2] < 0.0 ):
        #     reward -= 5.00
        #     done = True
        return reward, done

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward, taken_off = self.get_reward()
            if taken_off:
                done = True
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))