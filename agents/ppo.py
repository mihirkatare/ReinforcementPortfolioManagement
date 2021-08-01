import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

from .network import PPOActor, PPOCritic

eps=1e-10

class PPO:
    def __init__(self, num_assets, window_length, num_features):
        self.num_assets = num_assets

        self.actor = PPOActor(num_assets, window_length, num_features).double()
        self.critic = PPOCritic(num_assets, window_length, num_features).double()

        self.lr = 0.001

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.buffer = []
        self.log_probs = []
        self.output_actions = []
        self.states = []
        self.rewards = []
        self.weights = []

        self.clip = 0.2

        self.cov_mat = torch.eye(self.num_assets).double()
    
    def predict(self, s, weights):
        s = torch.flatten(torch.tensor(s))
        weights = torch.flatten(weights).double()

        action_vec = self.actor((s, weights))
        
        distribution = MultivariateNormal(action_vec, self.cov_mat)
        sampled = distribution.sample()
        log_prob = distribution.log_prob(sampled)

        self.log_probs.append(log_prob)
        self.output_actions.append(sampled)

        new_weights = torch.softmax(sampled, 0)
        new_weights = torch.unsqueeze(new_weights, 0)

        return new_weights

    def save_transition(self, state, action, reward, is_nonterminal, next_state, weight):
        self.states.append(state)
        self.rewards.append(reward)
        self.weights.append(weight)

    def reset_buffer(self):
        self.buffer = []
        self.log_probs = []
        self.output_actions = []
        self.states = []
        self.rewards = []
        self.weights = []

    def train(self):
        states, weights, rewards = torch.tensor(self.states), torch.stack(self.weights), torch.tensor(self.rewards)

        batch_size = len(self.states)
        states = states.view(batch_size, -1)
        weights = weights.view(batch_size, -1)
        rewards = rewards.view(batch_size, -1)

        V, _ = self.evaluate(states, weights)
        advantages = rewards - V.detach()

        advantages = (advantages - advantages.mean()) / (advantages.std() + eps)
        prev_log_probs = torch.stack(self.log_probs)

        # for i in range(5):
        V, new_log_probs = self.evaluate(states, weights)

        r = torch.exp(new_log_probs - prev_log_probs)

        surr1 = r * advantages
        surr2 = torch.clamp(r, 1 - self.clip, 1 + self.clip) * advantages

        actor_loss = (-torch.min(surr1, surr2)).mean()
        critic_loss = nn.MSELoss()(V, rewards)
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

    def evaluate(self, state, curr_weights):
        V = self.critic((state, curr_weights)).squeeze()

        mean = self.actor((state, curr_weights))
        dist = MultivariateNormal(mean, self.cov_mat)
        new_log_probs = dist.log_prob(torch.stack(self.output_actions))

        return V, new_log_probs
