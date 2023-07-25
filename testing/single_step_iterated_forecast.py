import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import pickle
import json
import tqdm
from tqdm import tqdm
from tensorflow.keras.models import model_from_json
from LSTNet.lstnet_util import GetArguments, LSTNetInit
from LSTNet.lstnet_model import PreSkipTrans, PostSkipTrans, PreARTrans, PostARTrans, LSTNetModel, ModelCompile

import yfinance as yf
import pandas as pd

tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'JPM', 'V', 'JNJ', 'PG', 'XOM', 'T', 'BAC', 'WMT', 'INTC', 'PFE',
           'VZ', 'KO', 'TSLA', 'MRK', 'DIS', 'UNH', 'HD', 'ADBE', 'CMCSA', 'PEP', 'CSCO', 'NVDA', 'NFLX',
           'ABT', 'NKE', 'CVX', 'ACN', 'TMUS', 'BMY', 'LLY', 'TMO', 'IBM', 'MCD', 'ORCL', 'UPS', 'MDT', 'COST',
           'PM', 'AVGO', 'SAP', 'HON', 'NEE', 'TXN', 'MO']

data = yf.download(tickers, period="1000d",group_by='ticker', auto_adjust=True)

open_prices = pd.DataFrame({ticker: data[ticker]['Open'] for ticker in tickers})
open_prices


custom_objects = {"PreSkipTrans": PreSkipTrans,
                  "PostSkipTrans": PostSkipTrans,
                  "PreARTrans": PreARTrans,
                  "PostARTrans": PostARTrans,
                  }

json_file = open('save\large_portfolio_horizon1\portfolio.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json, custom_objects=custom_objects)
loaded_model.load_weights('save\large_portfolio_horizon1\portfolio.h5')

model = loaded_model

def lstnet_forecast(model, forecast_steps, series, time_steps):
    forecast_steps=forecast_steps
    last_batch=series[-time_steps:,:]
    forecast=[]
    for step in range(forecast_steps):
        pred=model.predict(np.array(last_batch, dtype='float32').reshape(1,time_steps,series.shape[1]))
        forecast.append(pred)
        last_batch=np.append(last_batch[1:,:],pred, axis=0)
    forecast=np.array(forecast)
    
    return forecast

def plot_lstnet_forecast(model, data='LSTNet/data/large_portfolio.csv', forecast_steps=15, series_index=24):
    series=np.array(pd.read_csv(data))[:,1:]
    forecast=lstnet_forecast(model=model, series=series, forecast_steps=15, time_steps=28)
    forecast_df=pd.DataFrame(forecast.reshape(forecast_steps,series.shape[1]))
    forecast_df.columns=tickers 

    series_df=pd.DataFrame(series)
    series_df.columns=tickers
    cumulative_df=pd.concat([series_df,forecast_df], axis=0)
    cumulative_df=pd.DataFrame(np.array(cumulative_df))
    cumulative_df.iloc[-(100+len(forecast_df)):-len(forecast_df),series_index].plot(color='blue')
    cumulative_df.iloc[-len(forecast_df):,series_index].plot(color='red')
    output_df=cumulative_df.iloc[-55:,:]
   

    json_data = output_df.to_json(orient='index')

    with open('LargePortfolioLSTNet_forecast.json', 'w') as f:
        f.write(json_data)
    print('JSON data saved to', 'LargePortfolioLSTNet_forecast.json')


    plt.legend()
    title_ticker=tickers[series_index]
    plt.title(title_ticker)
    plt.show()

plot_lstnet_forecast(model=model)
    