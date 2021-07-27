import torch
import torch.nn as nn
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

class cnnNetwork(nn.Module):
    def __init__(self, M, L, N):
        super(cnnNetwork, self).__init__()
        self.M = M
        self.L = L 
        self.N = N

        self.conv2d1 = nn.Conv2d(in_channels=self.N, out_channels=32, kernel_size = (1,self.L), stride = 1, padding = "valid", bias=False)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

        self.conv2d2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size = (1,1), stride = 1, padding = "valid", bias=False)
        self.batchnorm2 = nn.BatchNorm2d(1)
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        y = self.conv2d1(x)
        y = self.relu(self.batchnorm1(y))
        y = self.conv2d2(y)
        y = self.relu(self.batchnorm2(y))
        y = self.flatten(y)
        return y

class DDPGActor(nn.Module):
    def __init__(self, M, L, N): # pass number of assets (M) for defining the network
        super(DDPGActor, self).__init__()
        self.M = M
        self.L = L 
        self.N = N
        self.basic = cnnNetwork(self.M, self.L, self.N)
        self.fc1 = nn.Linear(self.M, self.M)
        self.softmax = nn.Softmax()

    def forward(self, input):
        x, w = input
        # x = x.permute(0, 3, 1, 2)
        y = self.basic(x)
        y = self.fc1(torch.add(y,w))
        y = self.softmax(y)
        return y

class DDPGCritic(nn.Module):
    def __init__(self, M, L, N):
        super(DDPGCritic, self).__init__()
        self.M = M
        self.L = L 
        self.N = N
        self.basic = cnnNetwork(self.M, self.L, self.N)
        self.fc1 = nn.Linear(self.M, 1)

    def forward(self, input):
        x, w , action = input
        # x = x.permute(0, 3, 1, 2)
        y = self.basic(x)
        y = torch.add(y, action)
        y = torch.add(y, w)
        y = self.fc1(y)
        return y

class PPOActor(nn.Module):
    def __init__(self, num_assets, window_length, num_features):
        super(PPOActor, self).__init__()
        self.output_dim = num_assets
        # self.basic = basicNetwork()
        # self.fc1 = nn.Linear(350, 350)
        self.fc1 = nn.Linear(num_assets*window_length*num_features, 350)
        self.fc2 = nn.Linear(num_assets, 350)
        self.fc3 = nn.Linear(350, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, input):
        x, w = input
        # y = self.relu(self.basic(x))
        # y = self.fc1(y)
        y = self.relu(self.fc1(x))
        y_w = self.fc2(w)
        y = self.relu(y+y_w)
        y = self.fc3(y)
        return y

class PPOCritic(nn.Module):
    def __init__(self, num_assets, window_length, num_features):
        super(PPOCritic, self).__init__()
        self.output_dim = num_assets
        self.fc1 = nn.Linear(num_assets*window_length*num_features, 350)
        self.fc2 = nn.Linear(self.output_dim, 350)
        self.fc3 = nn.Linear(350, 1)
        self.relu = nn.ReLU()

    def forward(self, input):
        x, w = input
        y = self.relu(self.fc1(x))
        y_w = self.fc2(w)
        y = self.relu(y+y_w)
        y = self.fc3(y)
        return y
