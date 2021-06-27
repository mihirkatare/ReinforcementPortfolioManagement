import torch
import torch.nn as nn
from torch.optim import Adam

from .network import Actor, Critic

class PPO:
    def __init__(self, num_assets):
        self.num_assets = num_assets

        self.actor = Actor(num_assets)
        self.critic = Critic(num_assets)

        self.lr = 0.001

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.buffer = []
    
    def predict(self, s, action):
        return self.actor((torch.tensor(s).double(), torch.tensor(action).double())) # needs to be passed as a tuple

    def save_transition(self, state, action, r, is_nonterminal, next_state, prev_action):
        self.buffer.append((state, action, r, is_nonterminal, next_state, prev_action))

    def reset_buffer(self):
        self.buffer = []

    def train(self):
        state, action, reward, nonterminal, next_state, curr_weights = self.get_batch()

    def get_batch(self):
        state = [torch.tensor(data[0]).double() for data in self.buffer]
        action = [torch.tensor(data[1]).double() for data in self.buffer]
        reward = [torch.tensor(data[2]).double() for data in self.buffer]
        nonterminal = [torch.tensor(data[3]).double() for data in self.buffer]
        next_state = [torch.tensor(data[4]).double() for data in self.buffer]
        curr_weights = [torch.tensor(data[5]).double() for data in self.buffer]
        return state, action, reward, nonterminal, next_state, curr_weights
