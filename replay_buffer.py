import random
from collection import namedtuple, deque

class ReplayBuffer:
    """Fixed-size buffer to stroe experience tuples."""

    def _init_(self, buffer_size, batch_size):
        """Initializer a ReplayBuffer object.
        Params
        ===== 
            buffer_size: maxium size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size) # internal memory
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current sizer of internal memory"""
        return len(self.memory)