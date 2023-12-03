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

# shares normalization factor
# 100 shares per trade
HMAX_NORMALIZE = 100
# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE=1000000
# total number of stocks in our portfolio
#STOCK_DIM = 30
# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001
REWARD_SCALING = 1e-4
# Get the directory of the current file (__file__)
current_dir = os.path.dirname(__file__)
# Move up one directory to get to the 'finrl_modified' directory
root_dir = os.path.dirname(current_dir)
# Construct the path to the 'results' directory under 'finrl_modified'
RESULTS_DIR = os.path.join(root_dir, 'results')

class StockEnvTrain(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df,day = 0):
        #super(StockEnv, self).__init__()
        #money = 10 , scope = 1
        self.day = day
        self.df = df
        self.stock_dim = self._get_stock_dim()
        self.indicators = self._get_indicator_dim()
        self.indicator_names = self._get_indicator_names()

        # action_space normalization and shape is STOCK_DIM
        self.action_space = spaces.Box(low = -1, high = 1,shape = (self.stock_dim,)) 
        # If 2+4 techncal indicators
        self.observation_space = spaces.Box(
            low=0, high=np.inf, 
            shape=(1 + self.indicators * self.stock_dim,)
        )
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day,:]
        self.terminal = False  
           
        # initalize state
        self.state = [INITIAL_ACCOUNT_BALANCE] + self.data.adjcp.values.tolist() + [0] * self.stock_dim
        for indicator in self.indicator_names:
            self.state += self.data[indicator].values.tolist()

        # initialize reward
        self.reward = 0
        self.cost = 0
        # memorize all the total balance change
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self.trades = 0
        #self.reset()
        self._seed()

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

        
    def step(self, actions):

        info = {}
        truncated = False
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique())-1
        # print(actions)

        if self.terminal:
            plt.plot(self.asset_memory,'r')
            plt.savefig(f'{RESULTS_DIR}/account_value_train.png')
            plt.close()
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            
            #print("end_total_asset:{}".format(end_total_asset))
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv(f'{RESULTS_DIR}/account_value_train.csv')
            #print("total_reward:{}".format(self.state[0]+sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):61]))- INITIAL_ACCOUNT_BALANCE ))
            #print("total_cost: ", self.cost)
            #print("total_trades: ", self.trades)
            df_total_value.columns = ['account_value']
            df_total_value['daily_return']=df_total_value.pct_change(1)
            sharpe = (252**0.5)*df_total_value['daily_return'].mean()/ \
                  df_total_value['daily_return'].std()
            #print("Sharpe: ",sharpe)
            #print("=================================")
            df_rewards = pd.DataFrame(self.rewards_memory)
            #df_rewards.to_csv('results/account_rewards_train.csv')
            
            
            return self.state, self.reward, self.terminal, truncated, info

        else:
            # print(np.array(self.state[1:29]))

            actions = actions * HMAX_NORMALIZE
            #actions = (actions.astype(int))
            
            begin_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            #print("begin_total_asset:{}".format(begin_total_asset))
            
            argsort_actions = np.argsort(actions)
            
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = self.df.loc[self.day,:]         
            
            self.state = [self.state[0]] + \
                        self.data.adjcp.values.tolist() + \
                        list(self.state[(self.stock_dim+1):(self.stock_dim*2+1)])
            for indicator in self.indicator_names:
                self.state += self.data[indicator].values.tolist()
            # self._update_state()

            
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            self.asset_memory.append(end_total_asset)
            #print("end_total_asset:{}".format(end_total_asset))
            
            self.reward = end_total_asset - begin_total_asset            
            # print("step_reward:{}".format(self.reward))
            self.rewards_memory.append(self.reward)
            
            self.reward = self.reward*REWARD_SCALING


            return self.state, self.reward, self.terminal, truncated, info


    def reset(self, seed=None): 
        reset_info = None 
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        self.rewards_memory = []
        #initiate state
        self.state = [INITIAL_ACCOUNT_BALANCE] + self.data.adjcp.values.tolist() + [0] * self.stock_dim
        for indicator in self.indicator_names:
            self.state += self.data[indicator].values.tolist()
        # self._update_state()



        # iteration += 1 
        return self.state, reset_info
    
    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]