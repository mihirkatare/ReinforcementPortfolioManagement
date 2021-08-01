import torch
import numpy as np
import math
import matplotlib.pyplot as plt

from options import parse_args
from dataloader import getEnvironments, getTestEnvironment
from trader import StockTrader

from agents import DDPG, PPO, UCRP, WINNER, LOSSER

def backtest(agent, label, trader, opts):
    agents = [agent, UCRP(), WINNER(), LOSSER()]
    labels = [label, 'UCRP', 'Winner', 'Losser']

    all_wealths = []
    all_rewards = []

    for x in range(len(agents)):
        test_env = getTestEnvironment(opts)

        w = [1] + [0 for _ in range(test_env.portfolio_dim-1)]
        w = torch.tensor([w])

        data = test_env.step(None, None)
        i = 0
        wealths = []
        rewards = []
        wealth = 10e3

        while True:
            a = agents[x].predict(data['current'], w)
            data = test_env.step(w.detach().numpy(), a.detach().numpy())

            i += 1
            wealth = wealth*math.exp(data['reward'])
            wealths.append(wealth)
            rewards.append(math.exp(data['reward']))
            
            trader.update_summary(data['reward'], w, data['current'])
            trader.print_update(i)

            if test_env.idx == test_env.length-1:
                break

            w = a

        trader.write(f'./backtest_output/{labels[x]}.csv')

        all_wealths.append(wealths)
        all_rewards.append(rewards)
    
    plt.figure(figsize=(8, 6), dpi=100)
    for i in range(len(agents)):
        plt.plot(all_wealths[i], label=labels[i])
        mrr=float(np.mean(all_rewards[i])*100)
        sharpe=float(np.mean(all_rewards[i])/np.std(all_rewards[i])*np.sqrt(252))
        maxdrawdown=float(max(1-min(all_wealths[i])/np.maximum.accumulate(all_wealths[i])))
        print(labels[i],'   ',round(mrr,3),'%','   ',round(sharpe,3),'  ',round(maxdrawdown,3))

    plt.legend()
    plt.savefig('./backtest_output/wealths.png')
    plt.show()

def rollout(agent, trader, opts):
    train_env, val_env = getEnvironments(opts)

    w = [1] + [0 for _ in range(train_env.portfolio_dim-1)]
    w = torch.tensor([w])

    data = train_env.step(None, None)
    i = 0

    while True:
        a = agent.predict(data['current'], w)
        data = train_env.step(w.detach().numpy(), a.detach().numpy())

        agent.save_transition(data['current'], a, data['reward']-data['risk'], data['is_nonterminal'], data['next'], w)
        # print(data)

        i += 1
        if i % opts.episode_length==0:
            loss = agent.train()
            agent.reset_buffer()
        
        # update trader for output
        trader.update_summary(data['reward'], w, data['current'])
        trader.print_update(i)

        # do validation once every n iterations

        if train_env.idx == train_env.length-1:
            break

        w = a

    trader.plot_result()
    trader.reset()

def main():
    opts = parse_args()

    trader = StockTrader()

    if opts.mode == "train":
            # initialize agent
        agent = None
        if opts.model == "ppo":
            agent = PPO(7, 10, 5)
        elif opts.model == "ddpg":
            agent = DDPG(7, 10, 5)
        else:
            raise Exception(f"{opts.model} is not currently supported")

        for epoch in range(opts.epochs):
            print("epoch: " + str(epoch+1))
            rollout(agent, trader, opts)
    elif opts.mode == "test":
        # if opts.model_dir == None:
        #     raise Exception(f"model_dir needs to be specified when running on test set")

        # load model from path and initialize agent with it

        # call backtest (initialized PPO for now) (switch with trained model later)
        backtest(PPO(7, 10, 5), 'PPO', trader, opts)

    else:
        raise Exception("mode can only be one of 'test' or 'train'")

if __name__ == "__main__":
    main()
