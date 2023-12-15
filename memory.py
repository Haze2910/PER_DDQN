import torch
import random

# Implementation adapted from https://github.com/rlcode/per
class PrioritizedReplayBuffer():
    def __init__(self, size, alpha=0.5, beta=0.4, beta_increment=1e-4, eps=1e-2):
        self.tree = SumTree(size)
        self.size = size
        self.eps = eps # for numerical stability 
        self.alpha = alpha # control variable for how much priorization to use (alpha = 0 is the uniform case, alpha = 1 is full priority)
        self.beta = beta # control variable for the importance-sampling correction
        self.beta_increment = beta_increment 

    def get_priority(self, td_error):
        return (torch.abs(td_error) + self.eps) ** self.alpha

    def push(self, sample, td_error):
        priority = self.get_priority(td_error).item()
        self.tree.add(priority, sample)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            p = random.uniform(a, b)
            idx, priority, data = self.tree.get(p)
            priorities.append(priority)
            batch.append(data)
            idxs.append(idx)

        probs = torch.tensor(priorities) / self.tree.total()
        weights = (self.tree.actual_size * probs) ** (-self.beta)
        weights = weights / weights.max()

        batch = map(torch.stack, zip(*batch))

        return batch, weights, idxs

    def update_priorities(self, data_idxs, td_errors):
        for data_idx, td_error, in zip(data_idxs, td_errors):
            
            priority = self.get_priority(td_error)
            
            self.tree.update(data_idx, priority.item())

class SumTree():
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = torch.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.actual_size = 0
        self.position = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.position + self.capacity - 1

        self.data[self.position] = data
        self.update(idx, priority)

        self.position = (self.position + 1) % self.capacity

        if self.actual_size < self.capacity:
            self.actual_size += 1

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])
