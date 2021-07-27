import torch
import numpy as np

from options import parse_args
from dataloader import getEnvironments, getTestEnvironment

from agents.ddpg import DDPG
from agents.ppo import PPO

# yet to implement
def backtest(agent, env):
    pass

def rollout(agent, opts):
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

        # do validation once every n iterations

        if train_env.idx == train_env.length-1:
            print("Weights: ", w[0].tolist())
            value = np.dot(w.detach().numpy(), data["price"])
            print("Portfolio Value: ", value.item())
            break

        w = a

def main():
    opts = parse_args()

    # initialize agent
    agent = DDPG(7, 10, 5)

    for epoch in range(opts.epochs):
        print("epoch: " + str(epoch+1))
        rollout(agent, opts)
        

if __name__ == "__main__":
    main()
