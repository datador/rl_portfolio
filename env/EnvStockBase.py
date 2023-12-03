import numpy as np
import pandas as pd
from gymnasium.utils import seeding
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os

TRANSACTION_FEE_PERCENT = 0.001

class StockEnvBase(gym.Env):
    """Base class for stock trading environment"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, day=0, problem='portfolio'):
        self.day = day
        self.df = df
        self.problem = problem
        self.stock_dim = self._get_stock_dim()
        self.indicators = self._get_indicator_dim()
        self.indicator_names = self._get_indicator_names()

    def _get_stock_dim(self):
        stock_dim = self.df['tic'].nunique()
        return stock_dim
    
    def _get_indicator_dim(self):
        indicator_dim = self.df.drop(['datadate','tic','adjcp','open','high','low','volume', 'turbulence'], axis=1).shape[1]
        indicator_dim += 2 # add share and price indicators
        return indicator_dim
    
    def _get_indicator_names(self):
        indicator_names = self.df.drop(['datadate','tic','adjcp','open','high','low','volume', 'turbulence'], axis=1).columns.tolist()
        return indicator_names

    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action
        if self.state[index+self.stock_dim+1] > 0:
            #update balance
            self.state[0] += \
            self.state[index+1]*min(abs(action),self.state[index+self.stock_dim+1]) * \
             (1- TRANSACTION_FEE_PERCENT)

            self.state[index+self.stock_dim+1] -= min(abs(action), self.state[index+self.stock_dim+1])
            self.cost +=self.state[index+1]*min(abs(action),self.state[index+self.stock_dim+1]) * \
             TRANSACTION_FEE_PERCENT
            self.trades+=1
        else:
            pass

    
    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        available_amount = self.state[0] // self.state[index+1]
        # print('available_amount:{}'.format(available_amount))

        #update balance
        self.state[0] -= self.state[index+1]*min(available_amount, action)* \
                          (1+ TRANSACTION_FEE_PERCENT)

        self.state[index+self.stock_dim+1] += min(available_amount, action)

        self.cost+=self.state[index+1]*min(available_amount, action)* \
                          TRANSACTION_FEE_PERCENT
        self.trades+=1


    def _update_state(self):
        # Update price array from day's data
        price_array = self.data.adjcp.values.tolist()
        # Update owned shares array from current state
        shares_array = list(self.state[(self.stock_dim+1):(self.stock_dim*2+1)])
        # Combine balance, prices, and owned shares
        self.state = [self.state[0]] + price_array + shares_array
        # Update indicator values from day's data
        for indicator in self.indicator_names:
            self.state += self.data[indicator].values.tolist()
        # Ensure the state has the correct shape
        assert len(self.state) == self.observation_space.shape[0], f"State shape mismatch: {len(self.state)} vs {self.observation_space.shape[0]}"

    def _apply_portfolio_action(self, actions):
        if np.sum(actions) == 0:
            # If sum of actions is 0, repeat the last actions (hold the current state)
            actions = self.last_actions if hasattr(self, 'last_actions') else np.zeros_like(actions)
        else:
            # Normalize the action to enforce that weights sum up to 1
            actions = actions / np.sum(actions)
            self.last_actions = actions

        # Calculate total portfolio value
        total_portfolio_value = self.state[0] + np.sum(np.array(self.state[1:self.stock_dim + 1]) * np.array(self.state[self.stock_dim + 1:self.stock_dim * 2 + 1]))

        for index in range(self.stock_dim):
            # Calculate target value for this stock
            target_value = total_portfolio_value * actions[index]  # actions[0] is for cash
            current_value = self.state[index + 1] * self.state[index + self.stock_dim + 1]

            if target_value > current_value:
                # Buy stock
                self._buy_stock_portfolio(index, target_value - current_value)
            elif target_value < current_value:
                # Sell stock
                self._sell_stock_portfolio(index, current_value - target_value)

        # Update cash position for the remaining cash portion
        self.state[0] = total_portfolio_value * actions[-1]




    def _get_current_weights(self):
        current_prices = np.array(self.state[1:1+self.stock_dim])
        current_holdings = np.array(self.state[1+self.stock_dim:1+self.stock_dim*2])
        portfolio_value = self.state[0] + sum(current_holdings * current_prices)

        if portfolio_value <= 0:
            # Handling case where portfolio value is zero or negative
            # This might happen due to some extreme market conditions or bugs in the simulation
            return np.zeros(self.stock_dim + 1)  # All weights are zero

        weights = np.where(current_holdings > 0, (current_holdings * current_prices) / portfolio_value, 0)
        cash_weight = self.state[0] / portfolio_value if self.state[0] > 0 else 0
        return np.append(weights, cash_weight)

    def _sell_stock_portfolio(self, index, value):
        # Calculate the amount of stock to sell based on value
        stock_price = self.state[index + 1]
        amount_to_sell = value / stock_price

        if amount_to_sell > self.state[index + self.stock_dim + 1]:
            amount_to_sell = self.state[index + self.stock_dim + 1]

        # Update cash balance and stock holding
        self.state[0] += amount_to_sell * stock_price * (1 - TRANSACTION_FEE_PERCENT)
        self.state[index + self.stock_dim + 1] -= amount_to_sell
        self.cost += amount_to_sell * stock_price * TRANSACTION_FEE_PERCENT
        self.trades += 1

    def _buy_stock_portfolio(self, index, value):
        # Calculate the amount of stock to buy based on value
        stock_price = self.state[index + 1]
        amount_to_buy = value / stock_price

        if self.state[0] < amount_to_buy * stock_price * (1 + TRANSACTION_FEE_PERCENT):
            # Not enough cash to buy the desired amount, buy as much as possible
            amount_to_buy = self.state[0] / (stock_price * (1 + TRANSACTION_FEE_PERCENT))

        # Update cash balance and stock holding
        self.state[0] -= amount_to_buy * stock_price * (1 + TRANSACTION_FEE_PERCENT)
        self.state[index + self.stock_dim + 1] += amount_to_buy
        self.cost += amount_to_buy * stock_price * TRANSACTION_FEE_PERCENT
        self.trades += 1