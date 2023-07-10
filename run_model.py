
# FETCHING DATA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import pickle
import json
import yfinance as yf
import pandas as pd

tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'JPM', 'V', 'JNJ', 'PG', 'XOM', 'T', 'BAC', 'WMT', 'INTC', 'PFE',
           'VZ', 'KO', 'TSLA', 'MRK', 'DIS', 'UNH', 'HD', 'ADBE', 'CMCSA', 'PEP', 'CSCO', 'NVDA', 'NFLX',
           'ABT', 'NKE', 'CVX', 'ACN', 'TMUS', 'BMY', 'LLY', 'TMO', 'IBM', 'MCD', 'ORCL', 'UPS', 'MDT', 'COST',
           'PM', 'AVGO', 'SAP', 'HON', 'NEE', 'TXN', 'MO']

data = yf.download(tickers, period="1000d",group_by='ticker', auto_adjust=True)

open_prices = pd.DataFrame({ticker: data[ticker]['Open'] for ticker in tickers})
# open_prices.to_csv('large_portfolio.csv')
# np.savetxt('large_portfolio.txt', np.array(open_prices), delimiter=',')


import subprocess

def run_on_data():
    command = 'python main.py --data="data/large_portfolio.txt" --horizon=1 --save="save/portfolio" --test --savehistory --logfilename="log/lstnet" --debuglevel=20 --predict="all" --plot --save-plot="save/plots" --window=28 --validpercent=0.40 --batchsize=16 --skip=7 --epochs=20'  
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    if process.returncode == 0:
        print(output.decode())
    else:
        print("Error:", error.decode())
run_on_data()
