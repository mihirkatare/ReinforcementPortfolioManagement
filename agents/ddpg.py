import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import torch
import torch.nn as nn
import dataloader
from options import parse_args
from torch.optim import Adam
from .network import Actor, Critic

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

class DDPG():
    def __init__(self, M, L, N):
        self.M = M
        self.L = L
        self.N = N
        self.actor = Actor(self.M).double()
        self.actor_target = Actor(self.M).double()
        self.critic = Critic(self.M).double()
        self.critic_target = Critic(self.M).double()
        self.actor_optim = Adam(self.actor.parameters())
        self.critic_optim = Adam(self.critic.parameters())

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        self.tau = 0.5
        self.batch_size = 1
        self.discount = 0.05
        self.buffer = list()
        
        self.criterion = nn.MSELoss()

    def predict(self, s, action):
        return self.actor((torch.tensor(s).double(), torch.tensor(action).double())) # needs to be passed as a tuple
    
    def save_transition(self, state, action, r, is_nonterminal, next_state, prev_action):
        self.buffer.append((state, action, r, is_nonterminal, next_state, prev_action))

    def train(self):
        s, a, r, isnt, s_next, w = self.get_batch()
        length = len(s_next)

        y = []
        for i in range(length):
            q_next_i = self.critic_target( (s_next[i], w[i], self.actor_target((s_next[i], w[i])) ) )
            y.append(r[i] + self.discount*isnt[i]*q_next_i[i])

        # updating critic
        self.critic.zero_grad()
        q_pred = []
        for j in range(length):
            q_pred.append(self.critic((s[j], w[j], a[j]))[0] )
        
        c_loss = self.criterion(torch.stack(q_pred), torch.stack(y) )
        print(q_pred[0], y[0])
        c_loss.backward()
        self.critic_optim.step()

        # updating actor
        self.actor.zero_grad()
        a_loss_i = []
        for k in range(length):
            a_loss_i.append( -self.critic((s[k], w[k], self.actor((s[k], w[k])))) )
        a_loss = torch.stack(a_loss_i).mean()
        a_loss.backward()
        self.actor_optim.step()

        # soft updating targets
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def get_batch(self):
        s = [torch.tensor(data[0]).double() for data in self.buffer]
        a = [torch.tensor(data[1]).double() for data in self.buffer]
        r = [torch.tensor(data[2]).double() for data in self.buffer]
        isnt = [torch.tensor(data[3]).double() for data in self.buffer]
        s_next = [torch.tensor(data[4]).double() for data in self.buffer]
        w = [torch.tensor(data[5]).double() for data in self.buffer]
        return s, a, r, isnt, s_next, w

    def reset_buffer(self):
        self.buffer = list()
# opts = parse_args()
# train_loader, val_loader = dataloader.getDataloaders(opts)

# for i, data in enumerate(train_loader):
#     datatest = data
#     break

# net = Actor(7).double()
# x = datatest["x"].squeeze(0).double()
# print(x.dtype)
# # for parameter in net.parameters():
# #     print(parameter.dtype)
# out = net(x)
# print(out.shape)

# net2 = Critic(7).double()
# out2 = net2((x,out))
# print(out2)
