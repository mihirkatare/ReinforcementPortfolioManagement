# Reinforcement Learning for Portfolio Management

Contributors: @[nitishp1812](https://github.com/nitishp1812) @[mihirkatare](https://github.com/mihirkatare)

Implementing an agent for portfolio management using reinforcement learning algorithms. Our implementation is based on the following paper: [Adversarial Deep Reinforcement Learning for Portfolio Management](https://arxiv.org/pdf/1808.09940v3.pdf)

### Algorithms
We are currently using the following algorithms to train the agent:
- Deep Deterministic Policy Gradient (DDPG)
- Proximal Policy Gradient (PPO)
 
 ### Dataset
The agent is trained on a data set we created for a market of the following 6 stocks:
- Apple (AAPL)
- Amazon (AMZN)
- Ford (FORD)
- JP Morgan Chase (JPM)
- KO (Coca-Cola)
- XOM (Exxon Mobil)

We gathered daily open, close, high, low and trade volume data for these stocks from 2010 to 2020 to be used as inputs for our model.

 The model is trained to optimize profits by trading on the following stocks while considering factors like returns and transaction costs in its optimization objective.
