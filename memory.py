from random import sample


class BatchMemory:
    def __init__(self, max_size):
        # deque object that we've used for 'episodic_memory' is not suitable for random sampling
        # here, we instead use a fix-size array to implement 'buffer'
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def push(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index += 1
        # self.index = (self.index + 1) % self.max_size

    def is_full(self):
        if self.index == self.max_size:
            return True
        else:
            return False

    def reset(self):
        self.buffer = [None] * self.max_size
        self.index = 0
        self.size = 0

    def sample(self, batch_size):
        return [self.buffer[index] for index in range(batch_size)]

    def __len__(self):
        return self.size