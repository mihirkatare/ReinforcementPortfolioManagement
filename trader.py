import pandas as pd
import numpy as np
import math
from decimal import Decimal
import matplotlib.pyplot as plt

class StockTrader():
    def __init__(self):
        self.reset()

    def reset(self):
        self.wealth = 10e3
        self.total_reward = 0

        self.wealth_history = []
        self.reward_history = []
        self.weight_history = []
        # self.price_history = []

    def update_summary(self, reward, weights, prices):
        self.total_reward += reward
        self.reward_history.append(reward)
        self.wealth = self.wealth * math.exp(reward)
        self.wealth_history.append(self.wealth)

        self.weight_history.extend([','.join([str(Decimal(str(w0)).quantize(Decimal('0.00'))) for w0 in weights.tolist()[0]])])
        # self.price_history.extend([','.join([str(Decimal(str(p0)).quantize(Decimal('0.000'))) for p0 in prices.tolist()])])

    def write(self, filepath):
        wealth_history = pd.Series(self.wealth_history)
        reward_history = pd.Series(self.reward_history)
        weight_history = pd.Series(self.weight_history)
        # price_history = pd.Series(self.price_history)

        history = pd.concat([wealth_history, reward_history, weight_history], axis=1)
        history.to_csv(filepath)

    def print_update(self, iteration):
        total_reward=math.exp(self.total_reward) * 100
        print('*-----Iteration: {:d}, Reward: {:.6f}%, Wealth: {:.5f}-----*'.format(iteration, total_reward, self.wealth))

    def plot_result(self):
        pd.Series(self.wealth_history).plot()

        plt.show()
