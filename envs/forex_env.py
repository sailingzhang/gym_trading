import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt
import logging


class Actions(Enum):
    Sell = 0
    Buy = 1
    Hold = 2


class Positions(Enum):
    Short = 0
    Long = 1
    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long


class forex_candle_env(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, datapath, window_size,onlyClose=False,initCapitalPoint=2000,feePoint=20):
        """
        pointProfit:the gain or loss when  up or down 1 point every unit
        initCapitalPoint:how many  points the capital can cost every unit
        """
        self.basequate = 100000 
        self._seed()
        self._df = self._load_dataset(datapath)
        self._window_size = window_size
        self._onlyClose = onlyClose
        self._initCapitalPoint = initCapitalPoint
        self._feePoint = feePoint

        if self._onlyClose:
            #(closeprice[window_size],feePoint,holdPosition,holdPrice,floattingCaptalPoint)
            self._shape = (self._window_size+4,)
        else:
            #((openprice,highprice,lowprice,closeprice)[window_size],feePoint,holdPosition,holdPrice,floattingCaptalPoint)
            self._shape = (self._window_size*4+4,)

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=np.inf, high=np.inf, shape=self._shape, dtype=np.float32)

        self.reset()
    def reset(self):
        self._holdPosition = 0#the uint is 1 or -1
        self._holdPrice = 0
        self._capitalPoint =self._initCapitalPoint
        self._current_tick = self._window_size
        self._done = False
        return self._get_observation()


    def step(self, action):
        oldFloattingCapitalPoint = self._floattingCapitalPoint()
        self._updateStep(action)
        observation = self._get_observation()
        info = {}
        step_reward = self._floattingCapitalPoint() - oldFloattingCapitalPoint
        return observation, step_reward, self._done, info

    def _currentPrice(self):
        return self._df.loc[self._current_tick, 'Close']
    def _floattingCapitalPoint(self):
        return self.basequate * (self._currentPrice() - self._holdPrice)* self._holdPosition + self._capitalPoint
    def _updateStep(self,action):
        isClose = None
        oldHoldPosition = self._holdPrice
        pointPrice = 0
        getProfitPoint = 0
        self._current_tick += 1
        if action == Actions.Buy.value:
            if self._holdPosition < 0:
               isClose = True
               getProfitPoint = 1*(self._currentPrice() - self._holdPrice)*self.basequate 
            else:
                isClose = False
                pointPrice =  1*self._feePoint
            self._holdPosition +=1
        if action == Actions.Sell.value:
            if self._holdPosition > 0:
                isClose = True
                getProfitPoint = -1*(self._currentPrice() - self._holdPrice)*self.basequate 
            else:
                isClose = False
                pointPrice = -1 * self._feePoint 
            self._holdPosition -= 1
        
        if isClose == True:
            self._capitalPoint += getProfitPoint
        else:
            self._holdPrice = (self._holdPrice * oldHoldPosition + (self._currentPrice()+feePoint)*1)/self._holdPosition
        if self._floattingCapitalPoint() < 0 or self._current_tick >= len(self._df):
            self._done = True
        

    def _get_observation(self):
        if self._onlyClose:
            prices = self._df.loc[self._current_tick-self._window_size:self._current_tick, 'Close'].tolist()
        else:
            prices = self._df.loc[self._current_tick-self._window_size:self._current_tick,['Open', 'High', 'Low','Close']].tolist()
        observationlist = prices.append(self._feePoint,self._holdPosition,self._holdPrice,self._floattingCapitalPoint())
        return np.array(observationlist)
    
    def _load_dataset(filepath, index_name):
        return pd.read_csv(filepath, index_col=index_name)
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def render(self, mode='human'):
        logging.debug("floatprofitPoint={}".format(self._floattingCapitalPoint()))
    def render_all(self, mode='human'):
        pass
    def save_rendering(self, filepath):
        pass




