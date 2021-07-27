import os,sys,inspect
from numpy.testing._private.utils import requires_memory
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import torch
import torch.nn as nn
import dataloader
from options import parse_args
from torch.optim import Adam
from .network import DDPGActor, DDPGCritic

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
        # Hyperparameters of state
        self.M = M # Number of assets = m (stock/illiquid) + 1 (cash/liquid) {M includes cash here} [dflt: M=7]
        self.L = L # Length of time-series used in one state [dflt: L=10]
        self.N = N # Number of features to use {ex: close, high} [dflt: N=5]

        # Buffer initialization
        self.s_buffer = []
        self.a_buffer = []
        self.r_buffer = []
        self.isnt_buffer = []
        self.s_next_buffer = []
        self.w_buffer = []


        # Hyper-parameters for training
        self.tau = 10e-2 # Tau for soft-update
        self.batch_size = 1 
        self.discount = 0.9 # Discount for future rewards
        self.lr_actor = 3e-5
        self.lr_critic = 1e-5

        self.buffer = list() # initializes empty list as buffer 

        # Initializing models (as double to prevent casting issues)
        self.actor = DDPGActor(self.M, self.L, self.N).double()
        self.actor_target = DDPGActor(self.M, self.L, self.N).double()
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr_actor)

        self.critic = DDPGCritic(self.M, self.L, self.N).double()
        self.critic_target = DDPGCritic(self.M, self.L, self.N).double()        
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr_critic)

        # hard updating sets parameters of target/offline model equal
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)
        
        self.criterion = nn.MSELoss() # loss function choice

    def predict(self, s, action):
        return self.actor((torch.tensor(s).double().permute(0, 3, 1, 2), action.double())) # needs to be passed as a tuple
    
    def save_transition(self, state, action, r, is_nonterminal, next_state, prev_action):
        # self.buffer.append((state, action, r, is_nonterminal, next_state, prev_action))
        self.s_buffer.append(state)
        self.a_buffer.append(action)
        self.r_buffer.append(r)
        self.isnt_buffer.append(is_nonterminal)
        self.s_next_buffer.append(next_state)
        self.w_buffer.append(prev_action)

    def train(self):
        s, a, r, isnt, s_next, w = torch.tensor(self.s_buffer).permute(0, 1, 4, 2, 3).squeeze(1), torch.stack(self.a_buffer).squeeze(1), torch.tensor(self.r_buffer).unsqueeze(1), torch.tensor(self.isnt_buffer).unsqueeze(1), torch.tensor(self.s_next_buffer).permute(0, 1, 4, 2, 3).squeeze(1), torch.stack(self.w_buffer).squeeze(1)
        length = len(s_next)

        y = r + self.discount*isnt*self.critic_target( (s_next, a.detach(), self.actor_target((s_next, a.detach())) ) )     
       
        # updating critic
        self.critic_optim.zero_grad()        
        q_pred = self.critic( (s, w.detach(), a.detach()))        
        c_loss = self.criterion(q_pred, y)       
        c_loss.backward()
        self.critic_optim.step()

        # updating actor
        self.actor_optim.zero_grad()
        a_loss = -self.critic( (s, w.detach(), self.actor((s, w.detach())) ) ).mean()
        a_loss.backward()
        self.actor_optim.step()

        # soft updating targets
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def get_batch(self):
        s = [torch.tensor(data[0]).double() for data in self.buffer]
        a = [data[1].double() for data in self.buffer]
        r = [torch.tensor(data[2]).double() for data in self.buffer]
        isnt = [torch.tensor(data[3]).double() for data in self.buffer]
        s_next = [torch.tensor(data[4]).double() for data in self.buffer]
        w = [data[5].double() for data in self.buffer]
        return s, a, r, isnt, s_next, w

    def reset_buffer(self):
        self.s_buffer = []
        self.a_buffer = []
        self.r_buffer = []
        self.isnt_buffer = []
        self.s_next_buffer = []
        self.w_buffer = []
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
