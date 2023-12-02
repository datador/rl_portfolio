# common library
import pandas as pd
import numpy as np
import time
from stable_baselines3.common.vec_env import DummyVecEnv

# preprocessor
from preprocessing.preprocessors import *
# config
from config.config import *
# model
from model.models import *
import os

dow_jones = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GS', 'HD',
       'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT',
       'NKE', 'PFE', 'PG', 'RTX', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT',
       'XOM', 'AMZN'] 

tickers=['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'PG']

omx=['MAREL.IC', 'SJOVA.IC', 'EIM.IC', 'EIK.IC', 'HAGA.IC']
# , 'ARION.IC', 'FESTI.IC', 'EIM.IC', 'EIK.IC', 'HAGA.IC', 'SJOVA.IC'
def run_model() -> None:
    """Train the model."""

    # read and preprocess data
    preprocessed_path = "done_data.csv"
    if os.path.exists(preprocessed_path):
        print(f'Path exists: {preprocessed_path}')
        data = pd.read_csv(preprocessed_path, index_col=0)
    else:
        print(f"Path does not exists: {preprocessed_path}")
        data = preprocess_data(new_data=True, tickers=tickers, start_date='2009-01-01')
        print('Backfilling and adding traded dummy')
     #   data = backfill_and_mark_traded(data)
        print('Adding turbulence')
        data = add_turbulence(data)
        print(f'Saving data as: {preprocessed_path}')
        data.to_csv(preprocessed_path)
        

    print(data.head())
    print(data.size)

    # 2015/10/01 is the date that validation starts
    # 2016/01/01 is the date that real trading starts
    # unique_trade_date needs to start from 2015/10/01 for validation purpose
    unique_trade_date = data[(data.datadate > 20171001)&(data.datadate <= 20230707)].datadate.unique()
    print(unique_trade_date)
    print(data.datadate.max())

    # rebalance_window is the number of months to retrain the model
    # validation_window is the number of months to validation the model and select for trading
    rebalance_window = 63
    validation_window = 63
    
    ## Ensemble Strategy
    # run_ensemble_strategy(df=data, 
    #                       unique_trade_date= unique_trade_date,
    #                       rebalance_window = rebalance_window,
    #                       validation_window=validation_window)

    ### ONLY PPO ###
    run_ppo_strategy(df=data, 
                          unique_trade_date= unique_trade_date,
                          rebalance_window = rebalance_window,
                          validation_window=validation_window)

    #_logger.info(f"saving model version: {_version}")

if __name__ == "__main__":
    run_model()