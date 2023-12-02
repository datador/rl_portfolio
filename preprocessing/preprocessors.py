import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
from config import config
import yfinance as yf

def load_new_data(tickers: list, start_date: str = '2009-01-01') -> pd.DataFrame:
    """
    Fetch the latest data from Yahoo Finance and convert it to a long format.
    
    :param tickers: List of ticker symbols to fetch data for.
    :param new_data: Boolean to decide whether to fetch new data or not.
    :return: (pd.DataFrame) pandas dataframe with stock data in long format.
    """
    if not tickers:
        raise ValueError("Ticker list must be provided if new_data is True.")
        
    # Fetch the latest data from Yahoo Finance for all tickers at once
    data = yf.download(tickers, start=start_date)
    data_long = data.stack(level=1).reset_index(level=1).rename(columns={'level_1': 'Ticker'})
    data_long = data_long.reset_index()

    data_long.rename(columns={
            'Date': 'datadate',
            'Ticker': 'tic',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adjcp',
            'Volume': 'volume'
            }, inplace=True)
    
    data_long['datadate'] = pd.to_datetime(data_long['datadate']).dt.strftime('%Y%m%d').astype(int)
    
    data_long = data_long[['datadate', 'tic', 'adjcp', 'open', 'high', 'low', 'volume']]
    data_long = data_long.sort_values(['tic', 'datadate'], ignore_index=True)

    return data_long


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    #_data = pd.read_csv(file_name)
    return

def data_split(df,start,end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.datadate >= start) & (df.datadate < end)]
    data=data.sort_values(['datadate','tic'],ignore_index=True)
    #data  = data[final_columns]
    data.index = data.datadate.factorize()[0]
    return data

def calcualte_price(df):
    """
    calcualte adjusted close price, open-high-low price and volume
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    data = df.copy()
    data = data[['datadate', 'tic', 'prccd', 'ajexdi', 'prcod', 'prchd', 'prcld', 'cshtrd']]
    data['ajexdi'] = data['ajexdi'].apply(lambda x: 1 if x == 0 else x)

    data['adjcp'] = data['prccd'] / data['ajexdi']
    data['open'] = data['prcod'] / data['ajexdi']
    data['high'] = data['prchd'] / data['ajexdi']
    data['low'] = data['prcld'] / data['ajexdi'] 
    data['volume'] = data['cshtrd']

    data = data[['datadate', 'tic', 'adjcp', 'open', 'high', 'low', 'volume']]
    data = data.sort_values(['tic', 'datadate'], ignore_index=True)
    return data

def add_technical_indicator(df):
    """
    calcualte technical indicators
    use stockstats package to add technical inidactors
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    stock = Sdf.retype(df.copy())

    stock['close'] = stock['adjcp']
    unique_ticker = stock.tic.unique()

    macd = pd.DataFrame()
    rsi = pd.DataFrame()
    cci = pd.DataFrame()
    dx = pd.DataFrame()

    #temp = stock[stock.tic == unique_ticker[0]]['macd']
    for i in range(len(unique_ticker)):
        ## macd
        temp_macd = stock[stock.tic == unique_ticker[i]]['macd']
        temp_macd = pd.DataFrame(temp_macd)
        macd = pd.concat([macd, temp_macd]).reset_index(drop=True)
        ## rsi
        temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
        temp_rsi = pd.DataFrame(temp_rsi)
        rsi = pd.concat([rsi, temp_rsi]).reset_index(drop=True)
        ## cci
        temp_cci = stock[stock.tic == unique_ticker[i]]['cci_30']
        temp_cci = pd.DataFrame(temp_cci)
        cci = pd.concat([cci, temp_cci]).reset_index(drop=True)
        ## adx
        temp_dx = stock[stock.tic == unique_ticker[i]]['dx_30']
        temp_dx = pd.DataFrame(temp_dx)
        dx = pd.concat([dx, temp_dx]).reset_index(drop=True)


    df['macd'] = macd
    df['rsi'] = rsi
    df['cci'] = cci
    df['adx'] = dx
    df['test_add_indicator'] = dx**2

    return df


def preprocess_data(tickers, new_data=False, start_date='2009-01-01'):
    """data preprocessing pipeline"""
    if new_data:
        df = load_new_data(tickers=tickers, start_date=start_date)
        df_final=add_technical_indicator(df)
        # fill the missing values at the beginning
        df_final.fillna(method='bfill',inplace=True)
        return df_final
    else:
        df = load_dataset(file_name=config.TRAINING_DATA_FILE)
        # get data after 2009
        df = df[df.datadate>=20090000]
        # calcualte adjusted price
        df_preprocess = calcualte_price(df)
        # add technical indicators using stockstats
        df_final=add_technical_indicator(df_preprocess)
        # fill the missing values at the beginning
        df_final.fillna(method='bfill',inplace=True)
        return df_final

def backfill_and_mark_traded(df):
    # Convert 'datadate' to datetime and sort
    df['datadate'] = pd.to_datetime(df['datadate'], format=('%Y%m%d'))
    df = df.sort_values(by=['datadate', 'tic'])

    # Remove rows where 'tic' is NaN
    df = df[df['tic'].notna()]

    # Get unique dates from the dataset
    unique_dates = df['datadate'].unique()

    # Add a 'traded' column to the original DataFrame
    df['traded'] = 1

    # Function to backfill based on unique dates and add 'traded' column
    def backfill_based_on_unique_dates(group):
        group = group.set_index('datadate')
        # Reindex to unique dates
        group_reindexed = group.reindex(unique_dates)
        # Mark backfilled rows as not traded
        group_reindexed['traded'] = group_reindexed['traded'].fillna(0).astype(int)
        # Backfill other columns
        group_reindexed.update(group_reindexed.drop(columns=['traded']).bfill())
        return group_reindexed

    # Apply the function to each group
    filled_df = pd.concat([backfill_based_on_unique_dates(group) for _, group in df.groupby('tic')])

    # Drop any dates that do not have complete data for all tickers
    complete_dates = filled_df.groupby('datadate').count().min(axis=1)
    filled_df = filled_df[filled_df.index.isin(complete_dates[complete_dates == complete_dates.max()].index)]

    # Convert 'datadate' back to int format (YYYYMMDD)
    filled_df = filled_df.reset_index()
    filled_df['datadate'] = filled_df['datadate'].dt.strftime('%Y%m%d').astype(int)

    # Sort
    filled_df = filled_df.sort_values(by=['datadate', 'tic'])

    return filled_df

def add_turbulence(df):
    """
    add turbulence index from a precalcualted dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    turbulence_index = calcualte_turbulence(df) # df[df['traded'] == 1]
    df = df.merge(turbulence_index, on='datadate')
    df = df.sort_values(['datadate','tic']).reset_index(drop=True)
    return df



def calcualte_turbulence(df):
    """calculate turbulence index based on dow 30"""
    # can add other market assets
    
    df_price_pivot=df.pivot(index='datadate', columns='tic', values='adjcp')
    unique_date = df.datadate.unique()
    # start after a year
    start = 252
    turbulence_index = [0]*start
    #turbulence_index = [0]
    count=0
    for i in range(start,len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index ]]
        cov_temp = hist_price.cov()
        current_temp=(current_price - np.mean(hist_price,axis=0))
        temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
        if temp>0:
            count+=1
            if count>2:
                turbulence_temp = temp[0][0]
            else:
                #avoid large outlier because of the calculation just begins
                turbulence_temp=0
        else:
            turbulence_temp=0
        turbulence_index.append(turbulence_temp)
    
    
    turbulence_index = pd.DataFrame({'datadate':df_price_pivot.index,
                                     'turbulence':turbulence_index})
    return turbulence_index










