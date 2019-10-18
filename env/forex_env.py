import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt


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

    def __init__(self, df, window_size,onlyClose=False,initCapital=100,fee):
        self._seed()
        self._df = df
        self._window_size = window_size
        self._initCapital = initCapital
        self._onlyClose = onlyClose
        self._fee = fee

        self._holdPosition = 0
        self._holdPrice = 0
        self._floattingCapital = self._initCapital
        self._capital =self._initCapital

        self._current_tick = self._window_size
        self._done = None

        if self._onlyClose:
            #(closeprice[window_size],spread,holdPosition,holdPrice,capital)
            self._shape = (self._window_size+3,)
        else:
            #((openprice,highprice,lowprice,closeprice)[window_size],spread,holdPosition,holdPrice,capital)
            self._shape = (self._window_size*4+3,)

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=np.inf, high=np.inf, shape=self.shape, dtype=np.float32)
    def reset(self):
        self._holdPosition = 0
        self._holdPosition = 0
        self._capital =self._initCapital
        self._floattingCapital = self._initCapital
        self._current_tick = self._window_size
        self._done = False

        return self._get_observation()


    def step(self, action):


        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade:
            self._position = self._position.opposite()
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = dict(
            total_reward = self._total_reward,
            total_profit = self._total_profit,
            position = self._position.value
        )
        return observation, step_reward, self._done, info

    def _updateCation(self,action):
        ValidTrade = False
        if action == Actions.Buy.value:
            self._holdPosition +=1
        if action == Actions.Sell.value:
            self._holdPosition -= 1
        self._capital -= self._
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True
        self._current_tick += 1

    def _get_observation(self):
        if self._onlyClose:
            prices = self._df.loc[self._current_tick-self._window_size:self._current_tick, 'Close'].tolist()
        else:
            prices = self._df.loc[self._current_tick-self._window_size:self._current_tick,['Open', 'High', 'Low','Close']].tolist()
        observationlist = prices.append(self._spread,self._holdPosition,self._holdPrice,self._floattingCapital)
        return np.array(observationlist)
    



    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        plt.pause(0.01)


    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self.prices))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Short:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long:
                long_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )


    def _load_dataset(filepath, index_name):
        return pd.read_csv(filepath, index_col=index_name)
    def save_rendering(self, filepath):
        plt.savefig(filepath)


    def pause_rendering(self):
        plt.show()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _process_data(self):
        raise NotImplementedError


    def _calculate_reward(self, action):
        raise NotImplementedError


    def _update_profit(self, action):
        raise NotImplementedError


    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError
