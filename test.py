import gym

# import gym_anytrading
# from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions 
# from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL

from gym_anytrading_master.gym_anytrading import datasets
from gym_anytrading_master.gym_anytrading.envs import  ForexEnv

import matplotlib.pyplot as plt
from copy import deepcopy
import log_init
import logging


def test():
    log_init.log_init("test.log")
    logging.debug("enter")
    # env = gym.make('forex-v0', frame_bound=(50, 100), window_size=10)
    # env = gym.make('stocks-v0', frame_bound=(50, 100), window_size=10)
    # env = ForexEnv(df = deepcopy(datasets.FOREX_EURUSD_1H_ASK),window_size=24,frame_bound= (24, len(datasets.FOREX_EURUSD_1H_ASK)))
    env = ForexEnv(df = deepcopy(datasets.FOREX_EURUSD_1H_ASK),window_size=10,frame_bound= (50, 100))
    observation = env.reset()
    while True:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        # env.render()
        if done:
            print("info:", info)
            break

    plt.cla()
    env.render_all()
    plt.show()


if __name__=="__main__":
    test()