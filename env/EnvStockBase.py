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
        self.daily_returns = []
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

        
    def _apply_portfolio_action(self, actions):
        # Convert actions to a numpy array for element-wise multiplication
        actions = np.array(actions)

        # Calculate total assets
        total_assets = self.state[0] + np.sum(np.array(self.state[1:self.stock_dim + 1]) *
                                            np.array(self.state[self.stock_dim + 1:2*self.stock_dim + 1]))
        
        # Calculate target portfolio values based on actions
        target_values = total_assets * actions
        
        # Determine which stocks to sell
        current_prices = np.array(self.state[1:self.stock_dim + 1])
        current_holdings = np.array(self.state[self.stock_dim + 1:2*self.stock_dim + 1])
        current_values = current_prices * current_holdings
        stocks_to_sell = np.where(current_values > target_values[:-1], current_values - target_values[:-1], 0)
        
        # Execute sell actions
        self._selling_stocks(stocks_to_sell)
        
        # Determine which stocks to buy
        cash_available = self.state[0]
        stocks_to_buy = np.where(target_values[:-1] > current_values, target_values[:-1] - current_values, 0)
        
        # Execute buy actions with the available cash
        self._buying_stocks(stocks_to_buy, cash_available)
        
        # Ensure the final portfolio is within acceptable range of target values
        self._verify_rebalancing(target_values)

    def _selling_stocks(self, stocks_to_sell):
        for index, value_to_sell in enumerate(stocks_to_sell):
            if value_to_sell > 0:
                stock_price = self.state[index + 1]
                # Calculate the number of shares to sell
                shares_to_sell = value_to_sell / stock_price
                self._sell_stocks_rebalance(index, shares_to_sell)

    def _sell_stocks_rebalance(self, index, shares_to_sell):
        stock_price = self.state[index + 1]
        # Calculate the transaction fee
        transaction_fee = shares_to_sell * stock_price * TRANSACTION_FEE_PERCENT
        # Calculate cash increase after selling shares
        cash_from_sale = shares_to_sell * stock_price - transaction_fee
        # Update the state
        self.state[0] += cash_from_sale
        self.state[index + self.stock_dim + 1] -= shares_to_sell
        self.cost += transaction_fee
        self.trades += 1

    def _buying_stocks(self, stocks_to_buy, cash_available):
        for index, value_to_buy in enumerate(stocks_to_buy):
            if value_to_buy > 0 and cash_available > 0:
                stock_price = self.state[index + 1]
                # Calculate the number of shares to buy without exceeding cash available
                shares_to_buy = min(value_to_buy / stock_price, cash_available / (stock_price * (1 + TRANSACTION_FEE_PERCENT)))
                self._buy_stocks_rebalance(index, shares_to_buy)
                cash_available -= shares_to_buy * stock_price * (1 + TRANSACTION_FEE_PERCENT)

    def _buy_stocks_rebalance(self, index, shares_to_buy):
        stock_price = self.state[index + 1]
        # Calculate the transaction fee
        transaction_fee = shares_to_buy * stock_price * TRANSACTION_FEE_PERCENT
        # Update the state
        self.state[0] -= shares_to_buy * stock_price + transaction_fee
        self.state[index + self.stock_dim + 1] += shares_to_buy
        self.cost += transaction_fee
        self.trades += 1

    def _verify_rebalancing(self, target_values):
        """
        Verify that the portfolio has been rebalanced to closely match the target weights.
        """
        # Convert the slices of the state to NumPy arrays for element-wise multiplication
        current_prices = np.array(self.state[1:self.stock_dim + 1])
        current_holdings = np.array(self.state[self.stock_dim + 1:2*self.stock_dim + 1])

        # Recalculate the current values of the stocks
        current_values = current_prices * current_holdings
        # Add the cash to the current values
        current_values_with_cash = np.append(current_values, self.state[0])

        # Calculate the differences between the target and current values
        differences = target_values - current_values_with_cash
        
        # Check if the differences are within an acceptable range/tolerance
        tolerance = self.state[0] * 0.01  # for example, 1% of the cash value
        # if all(abs(difference) <= tolerance for difference in differences):
        #     print("Rebalancing successful within the specified tolerance.")
        # else:
        #     print("Rebalancing discrepancies found:", differences)
        #     # Additional logic can be added here to handle discrepancies.


    def calculate_sharpe_ratio(self):
        """
        Calculate the Sharpe Ratio based on the list of daily returns.
        """
        if len(self.daily_returns) > 1:
            mean_return = np.mean(self.daily_returns)
            std_return = np.std(self.daily_returns)
            sharpe_ratio = (252**0.5) * mean_return / std_return if std_return != 0 else 0
            return sharpe_ratio
        return 0