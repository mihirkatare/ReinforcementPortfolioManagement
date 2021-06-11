import torch
import numpy as np

from options import parse_args
from dataloader import getEnvironments, getTestEnvironment

from agents.ddpg import DDPG
# yet to implement
def backtest(agent, env):
    pass

def rollout(agent, opts):
    train_env, val_env = getEnvironments(opts)

    w = [1] + [0 for _ in range(train_env.portfolio_dim-1)]
    w = np.array([w])

    data = train_env.step(None, None)
    i = 0

    while train_env.idx < train_env.length:
        a = agent.predict(data['current'], w)
        data = train_env.step(w, a.detach().numpy())

        agent.save_transition(data['current'], a, data['reward']-data['risk'], data['is_nonterminal'], data['next'], w)
        # print(data)

        i += 1
        if i % opts.episode_length:
            loss = agent.train()
            agent.reset_buffer()
        
        # update trader for output

        # do validation once every n iterations

        w = a
def main():
    opts = parse_args()

    # initialize agent
    agent = DDPG(7, 10, 5)

    for epoch in range(opts.epochs):
        rollout(agent, opts)


if __name__ == "__main__":
    main()
