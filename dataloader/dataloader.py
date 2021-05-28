import torch
from torch import Dataset

import os

class StockDataset(Dataset):

  def __init__(self, datapath, window_length, features):
    self.datapath = datapath
    self.path_dict = dict()
    self.len_dict = dict()
    self.files = os.listdir(self.datapath)
    self.window_length = window_length
    self.features = features

    for file in self.files:
      filepath = os.path.join(self.datapath, file)
      self.path_dict[file] = filepath
      df = pd.read_csv(filepath)
      self.len_dict[file] = len(df)

  def __getitem__(self, index: int):
    filepath = None
    curr_file = None

    for file in self.files:
      if self.len_dict[file] > index:
        curr_file = file
        filepath = self.path_dict[file]
        break
      else:
        index -= self.len_dict[file]

    df = pd.read_csv(filepath)
    row = df.iloc[index, :]
    tick = curr_file.split('.')[0]
    
    data = {
        'tick': tick,
        'open': row['Open'],
        'close': row['Close'],
        'high': row['High'],
        'low': row['Low'],
        'vol': row['Volume']
    }

    return data

  def __len__(self):
    datapoints = 0

    for file in self.files:
      datapoints += self.len_dict[file]
    
    return datapoints
