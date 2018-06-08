import numpy as numpy
import copy

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        self.mu = mu* np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu). """

        slef.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu -x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state