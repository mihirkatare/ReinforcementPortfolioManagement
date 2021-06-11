import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

import os

from options import parse_args

epsilon = 10e-8

class Environment:

    def __init__(self, datapath, window_length, cost, opts):
        self.datapath = datapath
        self.window_length = window_length
        self.opts = opts

        self.files = os.listdir(self.datapath)
        self.assets = [f.split(".")[0].strip() for f in self.files]
        self.asset_dict = dict()

        for (asset, f) in zip(self.assets, self.files):
            filepath = os.path.join(self.datapath, f)
            df = pd.read_csv(filepath)

            # TODO: Change to use input for easy configuration
            df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Adj Close'])
            self.asset_dict[asset] = df

        sample_df = self.asset_dict[self.assets[-1]]
        self.portfolio_dim = len(self.assets) + 1  # all assets and cash
        self.variables = list(sample_df.columns)
        self.variables.pop(0)
        self.num_variables = len(self.variables)
        self.length = len(df) - self.window_length

        self.x = []
        self.y = []

        for i in range(self.length):
            values = []
            for j in range(self.num_variables):
                values.append(np.ones(self.window_length))

            out = np.ones(1)
            for asset in self.assets:
                df = self.asset_dict[asset]
                for j in range(self.num_variables):
                    values[j] = np.vstack(
                        (values[j], df.iloc[i:i+self.window_length, j+1]))

                retval = df['Close'][i+self.window_length] / \
                    df['Close'][i+self.window_length - 1]
                out = np.vstack((out, retval))

            values = np.stack(values, axis=2)
            values = values.reshape(
                1, self.portfolio_dim, self.window_length, self.num_variables)
            out = out.reshape(self.portfolio_dim, 1)

            self.x.append(values)
            self.y.append(out)

        self.idx = self.window_length
        self.cost = cost

    def step(self, w, a):
        if self.idx == self.window_length:
            data = {
                'reward': 0,
                'current': self.x[self.idx-1],
                'next': self.x[self.idx],
                'action': np.array([[1] + [0 for i in range(self.portfolio_dim-1)]]),
                'price': self.y[self.idx],
                'risk': 0,
                'is_nonterminal': 0 if self.idx == self.length - 1 else 1
            }
            self.idx += 1

            return data

        price = self.y[self.idx]
        mu = self.cost * (np.abs(a[0][1:] - w[0][1:])).sum()

        # std = self.states[self.t - 1][0].std(axis=0, ddof=0)
        # w2_std = (w2[0]* std).sum()

        # #adding risk
        # gamma=0.00
        # risk=gamma*w2_std

        risk = 0
        reward = (np.dot(a, price)[0] - mu)[0]
        reward = np.log(reward + epsilon)

        a = a / (np.dot(a, price) + epsilon)
        next_state = self.x[self.idx+1]
        curr_state = self.x[self.idx]
        data = {
            'reward': reward,
            'current': curr_state, 
            'next': next_state,
            'action': a,
            'price': price,
            'risk': risk,
            'is_nonterminal': 0 if self.idx == self.length - 1 else 1
        }

        self.idx += 1
        if self.idx == self.length:
            self.idx = self.window_length

        return data

def getEnvironments(opts):
    train_path = os.path.join(opts.datapath, 'train/')
    val_path = os.path.join(opts.datapath, 'val/')

    train_env = Environment(train_path, opts.window_length, opts.cost, opts)
    val_env = Environment(val_path, opts.window_length, opts.cost, opts)

    return train_env, val_env


def getTestEnvironment(opts):
    test_path = os.path.join(opts.datapath, 'test/')
    test_env = Environment(test_path, opts.window_length, opts.cost, opts)

    return test_env


if __name__ == "__main__":
    opts = parse_args()
    train_env, val_env = getEnvironments(opts)
    print(train_env.assets)
    w = [1] + [0 for _ in range(train_env.portfolio_dim-1)]
    w = np.array([w])
    i = 0

    while train_env.idx < train_env.length:
        data = train_env.step(w, w)
        print(data)

        i += 1
        if i == 5:
            break

