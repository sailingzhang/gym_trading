print("load gym")
import gym
from gym import spaces
from gym.utils import seeding
print("load np")
import numpy as np
# print("load pd")
# import pandas as pd
print("load enum")
from enum import Enum
print("load plt")
# import matplotlib.pyplot as plt
print("load logging")
import logging
print("load over")

class Actions(Enum):
    Sell = 0
    Buy = 1
    Hold = 2


class forex_candle_env(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, npdata, window_size,initCapitalPoint=2000,feePoint=20):
        """
        pointProfit:the gain or loss when  up or down 1 point every unit
        initCapitalPoint:how many  points the capital can cost every unit
        """
        
        self.basequate = 100000 
        self._np = npdata
        self._window_size = window_size
        self._initCapitalPoint = initCapitalPoint
        self._feePoint = feePoint
        self._shape = (self._window_size *self._np.shape[1]+4,)


        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self._shape, dtype=np.float32)
        logging.debug("enter,self._np.shape={},self._np.dtype={},self._shape={}".format(self._np.shape,self._np.dtype,self._shape))
        self.reset()
        self.seed()
    def reset(self):
        logging.debug("enter")
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
        logging.debug("step={},step_reward={}".format(self._current_tick,step_reward))
        return observation, step_reward, self._done, info

    def _currentPrice(self):
        # return self._df.loc[self._current_tick, 'Close']
        cur_price = self._np[self._current_tick][3]
        logging.debug("cur_trick={},cur_price={}".format(self._current_tick,cur_price))
        return cur_price
    def _floattingCapitalPoint(self):
        ret = self.basequate * (self._currentPrice() - self._holdPrice)* self._holdPosition + self._capitalPoint
        logging.debug("base={},cur_price={},hold_price={},hold_position={},capitalpoint={},ret={}".format(self.basequate,self._currentPrice(),self._holdPrice,self._holdPosition,self._capitalPoint,ret))
        return ret
    def _updateStep(self,action):
        # logging.debug("enter")
        isClose = False
        isOpen = False
        oldHoldPosition = self._holdPosition
        oldholdprice = self._holdPrice
        pointPrice = 0
        getProfitPoint = 0
        self._current_tick += 1
        if action == Actions.Buy.value:
            if self._holdPosition < 0:
               isClose = True
               getProfitPoint = -1*(self._currentPrice() - self._holdPrice)*self.basequate
               logging.debug("close short,position={},action={}".format(self._holdPosition,action))
            else:
                isOpen = True
                pointPrice =  1*self._feePoint/self.basequate
                logging.debug("open long,position={},action={}".format(self._holdPosition,action))
            self._holdPosition +=1
        elif action == Actions.Sell.value:
            if self._holdPosition > 0:
                isClose = True
                getProfitPoint = 1*(self._currentPrice() - self._holdPrice)*self.basequate
                logging.debug("close long,position={},action={}".format(self._holdPosition,action))
            else:
                isOpen = True
                pointPrice = -1 * self._feePoint/self.basequate
                logging.debug("open short,position={},action={}".format(self._holdPosition,action))
            self._holdPosition -= 1
        else:
            logging.debug("just hold")
        
        if isClose == True:
            self._capitalPoint += getProfitPoint
        if isOpen == True:
            self._holdPrice = (self._holdPrice * abs(oldHoldPosition) + (self._currentPrice()+pointPrice)*1)/abs(self._holdPosition)
            logging.debug("oldholdposition={},newposition={},c_price={},pointprice={} oldholdprice={},newholdprice={},self._floattingCapitalPoint()={}".format(oldHoldPosition,self._holdPosition,self._currentPrice(),pointPrice,oldholdprice,self._holdPrice,self._floattingCapitalPoint()))
        if self._floattingCapitalPoint() < 0 or self._current_tick >= len(self._np)-1:
            self._done = True
        logging.debug("cur_tick={},len(self._np)={}".format(self._current_tick,len(self._np)))

    def _get_observation(self):
        # logging.debug("startindex={},endindex={},self._np.shape={}".format(self._current_tick-self._window_size,self._current_tick,self._np.shape))
        obs1 = self._np[self._current_tick-self._window_size:self._current_tick,:]
        obs2 = np.array([self._feePoint,self._holdPosition,self._holdPrice,self._floattingCapitalPoint()])
        obs = np.append(obs1,obs2).astype("float32")
        # logging.debug("obs1.shape={},obs2.shape={},obs.shape={},obs.dtype={}".format(obs1.shape,obs2.shape,obs.shape,obs.dtype))
        logging.debug("self._done={},self._feePoint={},self._holdPosition={},self._holdPrice={},self._floattingCapitalPoint()={}".format(self._done,self._feePoint,self._holdPosition,self._holdPrice,self._floattingCapitalPoint()))
        return obs
    
    def _load_dataset(self,filepath, index_name):
        return pd.read_csv(filepath, index_col=index_name)
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def render(self, mode='human'):
        logging.debug("self._done={},self._feePoint={},self._holdPosition={},self._holdPrice={},self._floattingCapitalPoint()={}".format(self._done,self._feePoint,self._holdPosition,self._holdPrice,self._floattingCapitalPoint()))
    def render_all(self, mode='human'):
        pass
    def save_rendering(self, filepath):
        pass




