print("load gym")
import gym
from gym import spaces
from gym.utils import seeding
print("load np")
import numpy as np
print("load pd")
import pandas as pd
print("load enum")
from enum import Enum

print("load logging")
import logging
print("load over")

import pyecharts
from pyecharts.charts import Bar

print("load plt")
import matplotlib.pyplot as plt
# import plotly.graph_objects as go



class Actions(Enum):
    Sell = 0
    Buy = 1
    Hold = 2


class forex_candle_env(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,filepath,window_size,initCapitalPoint=2000,feePoint=20):
        """
        pointProfit:the gain or loss when  up or down 1 point every unit
        initCapitalPoint:how many  points the capital can cost every unit
        """
        
        self.basequate = 100000 
        self._pd = pd.read_csv(filepath)
        self._CloseIndexName = "Close"
        
        self._window_size = window_size
        self._initCapitalPoint = initCapitalPoint
        self._feePoint = feePoint
        self._shape = (self._window_size *(self._pd.shape[1]-1)+4,)


        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self._shape, dtype=np.float32)
        # logging.debug("self._pd.shape={},[1][1]={}".format(self._pd.shape,self._pd.iat[1,1]))
        # logging.debug("(0,Open,Close)={}".format(self._pd.loc[0,['Open','Close']].to_numpy()))
        # df.loc[:10, ['AAPL.Low','AAPL.Close']].to_numpy()

        logging.debug("shape={}".format(self._shape))
        self.reset()
        self.seed()
    def reset(self):
        logging.debug("enter")
        self._holdPosition = 0#the uint is 1 or -1
        self._holdPrice = 0
        self._capitalPoint =self._initCapitalPoint
        self._current_tick = self._window_size
        self._done = False

        self._OpenLongTicks=[]
        self._OPenShortTicks=[]
        self._CloseLongTicks=[]
        self._CloseShortTicks=[]

        return self._get_observation()


    def step(self, action):
        oldFloattingCapitalPoint = self._floattingCapitalPoint()
        oldprice = self._currentPrice()
        self._updateStep(action)
        observation = self._get_observation()
        step_reward = self._floattingCapitalPoint() - oldFloattingCapitalPoint
        info = {"action":action,"oldprice":oldprice,"curprice":self._currentPrice(),"reward":step_reward,"floattingCaption":self._floattingCapitalPoint()}
        logging.debug("step={},step_reward={}".format(self._current_tick,step_reward))
        return observation, step_reward, self._done, info

    def _currentPrice(self):
        cur_price = self._pd.loc[self._current_tick][self._CloseIndexName]
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
               self._CloseShortTicks.append(self._current_tick)
               logging.debug("close short,position={},action={},,cur_tick={}".format(self._holdPosition,action,self._current_tick))
            else:
                isOpen = True
                pointPrice =  1*self._feePoint/self.basequate
                self._OpenLongTicks.append(self._current_tick)
                logging.debug("open long,position={},action={},cur_tick={}".format(self._holdPosition,action,self._current_tick))
            self._holdPosition +=1
        elif action == Actions.Sell.value:
            if self._holdPosition > 0:
                isClose = True
                getProfitPoint = 1*(self._currentPrice() - self._holdPrice)*self.basequate
                self._CloseLongTicks.append(self._current_tick)
                logging.debug("close long,position={},action={},cur_tick={}".format(self._holdPosition,action,self._current_tick))
            else:
                isOpen = True
                pointPrice = -1 * self._feePoint/self.basequate
                self._OPenShortTicks.append(self._current_tick)
                logging.debug("open short,position={},action={},cur_tick={}".format(self._holdPosition,action,self._current_tick))
            self._holdPosition -= 1
        else:
            logging.debug("just hold")
        
        if isClose == True:
            self._capitalPoint += getProfitPoint
        if isOpen == True:
            self._holdPrice = (self._holdPrice * abs(oldHoldPosition) + (self._currentPrice()+pointPrice)*1)/abs(self._holdPosition)
            logging.debug("oldholdposition={},newposition={},c_price={},pointprice={} oldholdprice={},newholdprice={},self._floattingCapitalPoint()={}".format(oldHoldPosition,self._holdPosition,self._currentPrice(),pointPrice,oldholdprice,self._holdPrice,self._floattingCapitalPoint()))
        if self._floattingCapitalPoint() < 0 or self._current_tick >= len(self._pd)-1:
            self._done = True
        logging.debug("cur_tick={},len(self._df)={}".format(self._current_tick,len(self._pd)))

    def _get_observation(self):
        logging.debug("startindex={},endindex={},self._pd.shape={}".format(self._current_tick-self._window_size,self._current_tick,self._pd.shape))
        obs1 = self._pd.iloc[self._current_tick-self._window_size:self._current_tick,1:].to_numpy()
        obs2 = np.array([self._feePoint,self._holdPosition,self._holdPrice,self._floattingCapitalPoint()])
        obs = np.append(obs1,obs2).astype("float32")
        # logging.debug("obs1.shape={},obs2.shape={},obs.shape={},obs.dtype={}".format(obs1.shape,obs2.shape,obs.shape,obs.dtype))
        logging.debug("obs.shape={},self._done={},self._feePoint={},self._holdPosition={},self._holdPrice={},self._floattingCapitalPoint()={}".format(obs.shape,self._done,self._feePoint,self._holdPosition,self._holdPrice,self._floattingCapitalPoint()))
        return obs
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def render(self, mode='human'):
        logging.debug("self._done={},self._feePoint={},self._holdPosition={},self._holdPrice={},self._floattingCapitalPoint()={}".format(self._done,self._feePoint,self._holdPosition,self._holdPrice,self._floattingCapitalPoint()))


    def render_all(self, mode='human'):
        drawprice = self._pd.loc[0:self._current_tick,[self._CloseIndexName]].to_numpy()
        logging.debug("len(drawprice)={},drawprice={}".format(len(drawprice),drawprice))
        window_ticks = np.arange(len(drawprice))
        plt.plot(drawprice)

        plt.scatter(self._OpenLongTicks, self._pd.loc[self._OpenLongTicks,[self._CloseIndexName]], c='r',marker='^')
        plt.scatter(self._OPenShortTicks, self._pd.loc[self._OPenShortTicks,[self._CloseIndexName]],c='r', marker='v')
        plt.scatter(self._CloseLongTicks, self._pd.loc[self._CloseLongTicks,[self._CloseIndexName]],c='g' ,marker='^', alpha=0.5)
        plt.scatter(self._CloseShortTicks, self._pd.loc[self._CloseShortTicks,[self._CloseIndexName]],c='g', marker='v', alpha=0.5)

        plt.suptitle(
            "Floatting Capital: %.6f" % self._floattingCapitalPoint() + ' ~ ' +
            "Step:%d" % self._current_tick
        )

    def render_all_bak(self):
        logging.debug("begin plotly")
        # fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
        # fig.show()


        # df = pd.read_csv('data/finance-charts-apple.csv')
        # # # df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')
        # trace0 = go.Figure([go.Scatter(x=df['Date'], y=df['AAPL.High'])])
        # trace0.show()
        # # data = [trace0]
        # # py.plot(data)

        # df = pd.read_csv("data/FOREX_EURUSD_1H_ASK.csv")
        # trace0 = go.Figure([go.Scatter(x=df['Time'], y=df['Close'])])
        # data = [trace0]
        # py.plot(data)

        # print(pyecharts.__version__)
        # bar = Bar()
        # bar.add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
        # bar.add_yaxis("商家A", [5, 20, 36, 10, 75, 90])
        # # render 会生成本地 HTML 文件，默认会在当前目录生成 render.html 文件
        # # 也可以传入路径参数，如 bar.render("mycharts.html")
        # bar.render()

        # df = pd.read_csv('data/finance-charts-apple.csv')
        # readnumpy = df.loc[:10, ['AAPL.Low','AAPL.Close']].to_numpy()
        # logging.debug("readnumpy={}".format(readnumpy))
    def save_rendering(self, filepath):
        pass



def ValidationRun(env, net, episodes=10, device="cpu", epsilon=0.02, comission=0.1):
    stats = {
        'episode_reward': [],
        'order_profits': [],
    }

    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0.0
        position = None
        position_steps = None
        episode_steps = 0

        while True:
            # obs_v = torch.tensor([obs]).to(device)
            obs_v = np.array([obs])
            out_v = net(obs_v)
            action_idx = out_v.max(dim=1)[1].item()
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = action_idx
                # action = Actions(action_idx)
            obs, reward, done, info= env.step(action)
            total_reward += reward
            logging.info("info={}".format(info))
            # logging.info("action={},reward={},toal_reward={},floattingCaption={}".format(action,reward,total_reward,env._floattingCapitalPoint()))
            if done:
                stats['episode_reward'].append(total_reward)
                stats['order_profits'].append(env._floattingCapitalPoint())
                break
    logging.info("valid:stats={}".format(stats))
    return stats




