from ou_noise import OUNoise
from replay_buffer import ReplayBuffer
from agents.actor import Actor
from agents.critic import Critic
import numpy as np

class DDPG():
    """Reinforcement learning agent that learns using DDPG."""
    def __init__(self, task, train=True):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high


        # Set the learning rate suggested by paper:  https://pdfs.semanticscholar.org/71f2/03de1a53deae81a7707143f0ed564661e279.pdf
        self.actor_learning_rate = 0.001
        self.actor_decay = 0.0
        self.critic_learning_rate = 0.001
        self.critic_decay = 0.0

        # Actor Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high, self.actor_learning_rate, self.actor_decay)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high, self.actor_learning_rate, self.actor_decay)

        # Critic Model
        self.critic_local = Critic(self.state_size, self.action_size, self.critic_learning_rate, self.critic_decay)
        self.critic_target = Critic(self.state_size, self.action_size, self.critic_learning_rate, self.critic_decay)

        # initialize targets model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2

        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta,
                   self.exploration_sigma)

        # Replay memory
        self.buffer_size = 100000

        self.batch_size = 64

        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        self.best_w = None
        self.best_score = -np.inf
        # self.noise_scale = 0.7
        self.score = 0

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.01 # for soft update of target parameters


        # Indicate if we want to learn (or use to predict without learn)
        self.set_train(train)

    def reset_episode(self):
        self.total_reward = 0.0
        self.score = 0
        self.step_count = 0
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):

        self.total_reward += reward
        self.step_count += 1
        # Save experience /reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        self.score = self.total_reward / float(self.step_count) if self.step_count else 0.0
        # Update the noise factor depending on the new score value
        if  self.score >= self.best_score:
            self.best_score = self.score
            # Add mixin positive example for the agent to learn
            # self.memory.add(self.last_state, action, reward, next_state, done, positive=True)
            # if done:
                # Save the best_w so learned
                # self.best_w = self.actor_local.model.get_weights()
            # else:
        #     self.noise_scale = max(0.1 * self.noise_scale, 0.01)
        # else :
        #     self.noise_scale = min(2 * self.noise_scale, 3.5)

        # Learn, if enough samples are available in memory
        if self.train and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, done)

        # Roll over last state and action
        self.last_state= next_state

    def act(self, state):
        """Returns actions for given state(s)  as per current policy"""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample()) # add more noise for exploration

    def learn(self, experiences, done):
        """Update policy and value parameters using give batch experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards  = np.array([e.reward for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)

        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_state = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        # Q_targets_next = critic_target(next_state, actor_target(next_state))
        next_action = self.actor_target.model.predict_on_batch(next_state)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_state, next_action])

        # Compute Q targets for current states and train critic model(local)
        Q_targets = rewards + self.gamma * Q_targets_next * ( 1- dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]),
                            (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])

        # Soft-update target method

        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())


        assert len(local_weights) == len(target_weights), "Local and target model parameters mush have the same size"
        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

    def set_train(self, train):
        self.train = train
        # if not self.train:
        #     self.actor_local.model.set_weights(self.best_w)
