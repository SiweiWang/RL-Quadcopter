import numpy as np
from physics_sim import PhysicsSim

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
        """ Reward function

        Objective
        ===============
            1. Minimize the distance between current pos and target pos to take off as soon as possible
            2. Reward velocity in z
            2. Minimize the angles to make sure the take off is stable. take abs value
        """
        # item 1 -- distance penalty
        diff_xy = 0.5 * (abs(self.sim.pose[:2] - self.target_pos[:2]).sum())
        print ("diff_xy")
        print (diff_xy)

        # We care more about diff in z axis
        diff_z = abs(self.sim.pose[2] - self.target_pos[2])
        print ("diff_z")
        print (diff_z)

        # item 2 Reward velocity in z
        velocity_z = 2 * self.sim.v[2]
        print ("velocity_z")
        print (velocity_z)

        # item 3 -- angle  penalty
        angle = 0.5 * (abs(self.sim.pose[3:6]).sum())
        print ("angle")
        print (angle)

        # Calculate total reward
        reward = velocity_z - diff_xy - diff_z - angle
        print ("reward")
        print (reward)

        done = False
        # If we are higher, consider it as finish and give a big reward
        if (self.sim.pose[2] > self.target_pos[2]):
            reward += 1.0
            done = True
            print ("done reward")
            print (reward)
        print(done)
        return reward, done

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            new_reward, taken_off = self.get_reward()
            reward += new_reward
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