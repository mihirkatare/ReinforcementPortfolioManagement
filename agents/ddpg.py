import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import torch
import torch.nn as nn
import dataloader
from options import parse_args
from torch.optim import Adam

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

class resblock(nn.Module):
    """ Resnet block """

    def __init__(self, n_channels):        
        super(resblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = n_channels, out_channels = n_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.batchnorm1 = nn.BatchNorm2d(n_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = n_channels, out_channels = n_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.batchnorm2 = nn.BatchNorm2d(n_channels)

    def forward(self, x):
        y = self.relu(self.batchnorm1(self.conv1(x)))
        y = self.relu(x + self.batchnorm2(self.conv2(y)))
        return y

class basicNetwork(nn.Module):
    def __init__(self):
        super(basicNetwork, self).__init__()
        self.n_channels = 7
        self.conv1 = nn.Conv2d(in_channels = self.n_channels, out_channels = self.n_channels, kernel_size = 1, stride = 1, padding = 0, bias = False).float()
        self.res1 = resblock(self.n_channels).double()
        self.batchnorm1 = nn.BatchNorm2d(self.n_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = self.n_channels, out_channels = self.n_channels, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.batchnorm2 = nn.BatchNorm2d(self.n_channels)
        self.flatten = nn.Flatten()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(self.batchnorm1(y))
        y = self.res1(y)
        y = self.relu(self.batchnorm2(self.conv2(y)))
        y = self.flatten(y)
        return y

class Actor(nn.Module):
    def __init__(self, M):
        super(Actor, self).__init__()
        self.M = M
        # self.L = L
        # self.N = N
        self.basic = basicNetwork()
        self.fc1 = nn.Linear(350, 350)
        self.fc2 = nn.Linear(self.M, 350)
        self.fc3 = nn.Linear(350, self.M)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

    def forward(self, input):
        x, w = input
        y = self.relu(self.basic(x))
        y = self.fc1(y)
        y_w = self.fc2(w)
        y = self.relu(y+y_w)
        y = self.fc3(y)
        y = self.softmax(y)
        return y

class Critic(nn.Module):
    def __init__(self, M):
        super(Critic, self).__init__()
        self.M = M
        self.basic = basicNetwork()
        self.fc1 = nn.Linear(350, 350)
        self.fc2 = nn.Linear(self.M, 350)
        self.fc3 = nn.Linear(self.M, 350)
        self.fc4 = nn.Linear(350, 1)
        self.relu = nn.ReLU()

    def forward(self, input):
        x, w , action = input
        y = self.relu(self.basic(x))
        y = self.fc1(y)
        y_a = self.fc2(action)
        y_w = self.fc3(w)
        y = self.relu(y+y_a+y_w)
        y = self.fc4(y)
        return y


class DDPG():
    def __init__(self, M, L, N):
        self.M = M
        self.L = L
        self.N = N
        self.actor = Actor(self.M)
        self.actor_target = Actor(self.M)
        self.critic = Critic(self.M)
        self.critic_target = Critic(self.M)
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
        return self.actor((s, action)) # needs to be passed as a tuple
    
    def save_transition(self, state, action, r, is_nonterminal, next_state, prev_action):
        self.buffer.append((state, action, r, is_nonterminal, next_state, prev_action))

    def train(self):
        s, a, r, isnt, s_next, w = self.get_batch()
        q_next = self.critic_target((s_next, w, self.actor_target((s_next, w)) ))
        y = r + self.discount*isnt*q_next

        # updating critic
        self.critic.zero_grad()
        q_pred = self.critic((s, w, a))
        c_loss = self.criterion(q_pred, y)
        c_loss.backward()
        self.critic_optim.step()

        # updating actor
        self.actor.zero_grad()
        a_loss = -self.critic((s, w, self.actor((s, w)))).mean()
        a_loss.backward()
        self.actor_optim.step()

        # soft updating targets
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def get_batch(self):
        return self.buffer[0] # testing purposes
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
