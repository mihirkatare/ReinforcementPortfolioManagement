import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

from .network import PPOActor, PPOCritic

eps=1e-10

class PPO:
    def __init__(self, num_assets):
        self.num_assets = num_assets

        self.actor = PPOActor(num_assets)
        self.critic = PPOCritic(num_assets)

        self.lr = 0.001

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.buffer = []
        self.log_probs = []
        self.output_actions = []

        self.clip = 0.2

        self.cov_mat = torch.eye(self.num_assets)
    
    def predict(self, s, weights):
        action_vec = self.actor((torch.tensor(s).double(), torch.tensor(weights).double()))
        
        distribution = MultivariateNormal(action_vec, self.cov_mat)
        sampled = distribution.sample()
        log_prob = distribution.log_prob(sampled)

        self.log_probs.append(log_prob)
        self.output_actions.append(sampled)

        return torch.softmax(sampled)

    def save_transition(self, state, action, r, is_nonterminal, next_state, prev_action):
        self.buffer.append((state, action, r, is_nonterminal, prev_action))

    def reset_buffer(self):
        self.buffer = []
        self.log_probs = []
        self.output_actions = []

    def train(self):
        state, curr_weights, action, reward, nonterminal = self.get_batch()
        batch_len = len(state)

        V, _ = self.evaluate(state, curr_weights)
        advantages = reward - V.detach()

        advantages = (advantages - advantages.mean()) / (advantages.std() + eps)

        for i in range(5):
            V, new_log_probs = self.evaluate(state, curr_weights)

            r = torch.exp(new_log_probs - self.log_probs)

            surr1 = r * advantages
            surr2 = torch.clamp(r, 1 - self.clip, 1 + self.clip) * advantages

            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = nn.MSELoss()(V, reward)
            
            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optim.step()
            
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

    def evaluate(self, state, curr_weights):
        V = self.critic((state, curr_weights)).squeeze()

        mean = self.actor((state, curr_weights))
        dist = MultivariateNormal(mean, self.cov_mat)
        new_log_probs = dist.log_prob(self.output_actions)

        return V, new_log_probs

    def get_batch(self):
        state = torch.tensor([torch.tensor(data[0]).double() for data in self.buffer])
        action = torch.tensor([torch.tensor(data[1]).double() for data in self.buffer])
        reward = torch.tensor([torch.tensor(data[2]).double() for data in self.buffer])
        nonterminal = torch.tensor([torch.tensor(data[3]).double() for data in self.buffer])
        curr_weights = torch.tensor([torch.tensor(data[4]).double() for data in self.buffer])
        return state, curr_weights, action, reward, nonterminal
