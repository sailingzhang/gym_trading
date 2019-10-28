import gym
import time
# import gym_anytrading
# from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions 
# from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL

from gym_anytrading_master.gym_anytrading import datasets
# from gym_anytrading_master.gym_anytrading.envs import  ForexEnv

import matplotlib.pyplot as plt
from copy import deepcopy
from envs import forex_env
import numpy as np
import log_init
import logging

# FOREC_DATA="C:\mydata\develop\mygit\gym_trading\data\FOREX_EURUSD_1H_ASK.csv"
# FOREC_DATA="data/FOREX_EURUSD_1H_ASK.csv"
FOREC_DATA="data/FOREX_EURUSD_1H_ASK_CLOSE.csv"
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
        env.render()
        if done:
            print("info:", info)
            break

    plt.cla()
    # env.render_all()
    plt.show()


def test2():
    log_init.log_init("../tmp/test.log",'INFO')
    logging.debug("enter")
    # npdata = np.loadtxt(FOREC_DATA,delimiter=",",skiprows=1,usecols=(1,2,3,4,5)).astype("float32")
    env = forex_env.forex_candle_env(FOREC_DATA,window_size=600,initCapitalPoint=20000,feePoint=20)
    observation = env.reset()
    step = 0
    allrewards = 0
    while True:
        # a = forex_env.Actions.Sell.value
        # a = env.action_space.sample()
        if step == 0:
            a = forex_env.Actions.Sell.value
        elif step == 3420:
            a = forex_env.Actions.Buy.value
        else:
            a =  forex_env.Actions.Hold.value
        logging.info("tyep(a)={},a={}".format(type(a),a))
        observation, reward, done, info = env.step(a)
        step = info["tick"]
        allrewards += reward
        # env.render()
        if done:
            logging.info("allrewards={},info={}".format(allrewards,info))
            break
    env.render_all()
    # plt.cla()
    # env.render_all()
    # plt.show()

if __name__=="__main__":
    # test()
    test2()
    # time.sleep(50)