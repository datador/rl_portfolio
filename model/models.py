# common library
import pandas as pd
import numpy as np
import time
import gymnasium as gym
import os

# RL models from stable-baselines
from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import TD3

from stable_baselines3.ddpg.policies import MlpPolicy
#from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from preprocessing.preprocessors import *
from config import config

# customized env
from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_validation import StockEnvValidation
from env.EnvMultipleStock_trade import StockEnvTrade

current_dir = os.path.dirname(__file__)
# Move up one directory to get to the 'finrl_modified' directory
root_dir = os.path.dirname(current_dir)
# Construct the path to the 'results' directory under 'finrl_modified'
RESULTS_DIR = os.path.join(root_dir, 'results')

def train_A2C(env_train, model_name, timesteps=25000):
    """A2C model"""

    start = time.time()
    model = A2C('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model

def train_DDPG(env_train, model_name, timesteps=10000):
    """DDPG model"""

    # add the noise objects for DDPG
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    start = time.time()
    model = DDPG('MlpPolicy', env_train, action_noise=action_noise)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (DDPG): ', (end-start)/60,' minutes')
    return model

def train_PPO(env_train, model_name, timesteps=50000):
    """PPO model"""

    start = time.time()
    model = PPO('MlpPolicy', env_train, ent_coef = 0.005, batch_size = 2048)
    #model = PPO2('MlpPolicy', env_train, ent_coef = 0.005)

    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model

def DRL_prediction(df,
                   model,
                   name,
                   last_state,
                   iter_num,
                   unique_trade_date,
                   rebalance_window,
                   turbulence_threshold,
                   initial,
                   start_date=None,
                   end_date=None):
    ### make a prediction based on trained model###

    # If start_date and end_date are provided, use them directly
    if start_date is None and end_date is None:
        start_date=unique_trade_date[iter_num - rebalance_window]
        end_date=unique_trade_date[iter_num]

    trade_data = data_split(df, start=start_date, end=end_date)

    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data,
                                                   turbulence_threshold=turbulence_threshold,
                                                   initial=initial,
                                                   previous_state=last_state,
                                                   model_name=name,
                                                   iteration=iter_num)])
    obs_trade = env_trade.reset()

    for i in range(len(trade_data.index.unique())):
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
        if i == (len(trade_data.index.unique()) - 2):
            # print(env_test.render())
            last_state = env_trade.render()

    # df_last_state = pd.DataFrame({'last_state': [last_state]})
    # df_last_state.to_csv(f'{RESULTS_DIR}/last_state_{name}_{i}.csv', index=False)
    return last_state


def DRL_validation(model, test_data, test_env, test_obs) -> None:
    ###validation process###
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)


def get_validation_sharpe(iteration):
    ###Calculate Sharpe ratio based on validation results###
    df_total_value = pd.read_csv(f'{RESULTS_DIR}/account_value_validation_{iteration}.csv', index_col=0)
    df_total_value.columns = ['account_value_train']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / \
             df_total_value['daily_return'].std()
    return sharpe


def run_ppo_strategy(df, unique_trade_date, rebalance_window, validation_window) -> None:
    """Strategy that trains and uses only PPO agent"""
    print("============Start PPO Strategy============")

    # based on the analysis of the in-sample data
    insample_turbulence = df[(df.datadate < 20151000) & (df.datadate >= 20090000)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)

    start = time.time()
    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        print("============================================")
        ## initial state is empty
        end_date_index = df.index[df["datadate"] == unique_trade_date[i - rebalance_window - validation_window]].to_list()[-1]
        start_date_index = end_date_index - validation_window*30 + 1
        initial = True if i - rebalance_window - validation_window == 0 else False

        # Tuning turbulence index based on historical data
        # Turbulence lookback window is one quarter
        historical_turbulence = df.iloc[end_date_index - validation_window * 30 + 1 : end_date_index + 1, :]
        historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])
        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

        turbulence_threshold = insample_turbulence_threshold if historical_turbulence_mean > insample_turbulence_threshold else np.quantile(insample_turbulence.turbulence.values, 1)
        print("turbulence_threshold: ", turbulence_threshold)

        ############## Environment Setup starts ##############
        ## training env
        train = data_split(df, start=20090000, end=unique_trade_date[i - rebalance_window - validation_window])
        env_train = DummyVecEnv([lambda: StockEnvTrain(train, problem='portfolio')])

        ## validation env
        validation = data_split(df, start=unique_trade_date[i - rebalance_window - validation_window], end=unique_trade_date[i - rebalance_window])
        env_val = DummyVecEnv([lambda: StockEnvValidation(validation, turbulence_threshold=turbulence_threshold, iteration=i)])
        obs_val = env_val.reset()
        ############## Environment Setup ends ##############

        ############## Training and Validation starts ##############
        print("======PPO Training========")
        model_ppo = train_PPO(env_train, model_name="PPO_100k_dow_{}".format(i), timesteps=100000)
        print("======PPO Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ", unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_ppo, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_ppo = get_validation_sharpe(i)
        print("PPO Sharpe Ratio: ", sharpe_ppo)
        ############## Training and Validation ends ##############

        ############## Trading starts ##############
        print("======Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])
        last_state_ensemble = DRL_prediction(df=df, model=model_ppo, name="ppo", last_state=[], iter_num=i, unique_trade_date=unique_trade_date, rebalance_window=rebalance_window, turbulence_threshold=turbulence_threshold, initial=initial)
        ############## Trading ends ##############

    end = time.time()
    print("PPO Strategy took: ", (end - start) / 60, " minutes")

def run_ppo_strategy_full(df, unique_trade_date, rebalance_window, validation_window):
    """Strategy that trains the PPO agent and validates on a hold-out set"""
    print("============Start PPO Strategy with Validation============")
    val_name = ''

    turbulence_threshold = 150
    print("turbulence_threshold: ", turbulence_threshold)

    # Define start dates for training, validation, and test sets
    train_start = df.datadate.min()
    validation_start = unique_trade_date[-(validation_window + rebalance_window)]
    test_start = unique_trade_date[-rebalance_window]

    # Define training, validation, and test sets
    train = data_split(df, start=train_start, end=validation_start)
    validation = data_split(df, start=validation_start, end=test_start)
    test = data_split(df, start=test_start, end=unique_trade_date[-1])

    # Setup the training and validation environments
    env_train = DummyVecEnv([lambda: StockEnvTrain(train, problem='portfolio')])
    env_val = DummyVecEnv([lambda: StockEnvValidation(validation, problem='portfolio')])

    # Train the PPO model
    print(f"======PPO Training from: {train_start} to {validation_start}======")
    model_ppo = train_PPO(env_train, model_name="PPO_train", timesteps=100000)

    # Validate the model
    print(f"======PPO Validation from: {validation_start} to {test_start}======")
    DRL_validation(model=model_ppo, test_data=validation, test_env=env_val, test_obs=env_val.reset())
    sharpe_ratio = get_validation_sharpe(val_name)

    # Trading
    print(f"======Trading from: {test_start} to {unique_trade_date[-1]}======")
    env_trade = DummyVecEnv([lambda: StockEnvTrade(test, problem='portfolio')])
    last_state = DRL_prediction(
        df=df,
        model=model_ppo,
        name="ppo",
        last_state=[],
        iter_num=None, 
        unique_trade_date=unique_trade_date,
        rebalance_window=None,
        turbulence_threshold=turbulence_threshold,
        initial=False,
        start_date=test_start,
        end_date=unique_trade_date[-1]
    )
    
    # Print the Sharpe Ratio
    print("Validation Sharpe Ratio: ", sharpe_ratio)
