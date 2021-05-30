import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

import os

from options import parse_args

class StockDataset(Dataset):

    def __init__(self, datapath, window_length, opts):
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
        self.portfolio_dim = len(self.assets) + 1 # all assets and cash
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
                    values[j] = np.vstack((values[j], df.iloc[i:i+self.window_length, j+1]))

                retval =  df['Close'][i+self.window_length] / df['Close'][i+self.window_length -1]
                out = np.vstack((out, retval))

            values = np.stack(values, axis=2)
            values = values.reshape(1, self.portfolio_dim, self.window_length, self.num_variables)
            out = out.reshape(1, self.portfolio_dim, 1)

            self.x.append(values)
            self.y.append(out)

    def __getitem__(self, index: int):
        data = {
            'x': self.x[index],
            'y': self.y[index]
        }

        return data

    def __len__(self):
        return self.length


def getDataloaders(opts):
    train_path = os.path.join(opts.datapath, 'train/')
    val_path = os.path.join(opts.datapath, 'val/')

    train_dataset = StockDataset(train_path, opts.window_length, opts)
    val_dataset = StockDataset(val_path, opts.window_length, opts)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=opts.batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=opts.batch_size, shuffle=False)

    return train_loader, val_loader


# train_loader, val_loader, test_dataset = getDataloaders('./data/', 1, 0.7, 0.1)
if __name__ == "__main__":
    opts = parse_args()
    train_loader, val_loader = getDataloaders(opts)

    for i, data in enumerate(train_loader):
        print(data)

        if i == 5:
            break
