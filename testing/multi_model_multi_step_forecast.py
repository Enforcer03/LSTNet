import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import pickle
import json
import yfinance as yf
from run_model import train_model
import tqdm
from tqdm import tqdm

from tensorflow.keras.models import model_from_json
from LSTNet.lstnet_util import GetArguments, LSTNetInit
from LSTNet.lstnet_model import PreSkipTrans, PostSkipTrans, PreARTrans, PostARTrans, LSTNetModel, ModelCompile

custom_objects = {"PreSkipTrans": PreSkipTrans,
                  "PostSkipTrans": PostSkipTrans,
                  "PreARTrans": PreARTrans,
                  "PostARTrans": PostARTrans,
                  }

def multi_train(forecast_steps=50):
    for step in tqdm(range(forecast_steps)):
        train_model(horizon=step, save_name=f"large_portfolio_{step}")
    
def load_multiple_models(n_models=50):
    models=[]
    for step in tqdm(range(n_models)):
        json_file = open(f'save\large_portfolio{step}\large_portfolio.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json, custom_objects=custom_objects)
        loaded_model.load_weights(f"save\large_portfolio{step}\portfolio.h5")
        models.append(loaded_model)

def lstnet_forecast(models, forecast_steps, series, time_steps):
    forecast_steps=forecast_steps
    last_batch=series[-time_steps:,:]
    forecast=[]
    for step in tqdm(range(forecast_steps)):
        pred=models[step].predict(np.array(last_batch, dtype='float32').reshape(1,time_steps,series.shape[1]))
        forecast.append(pred)
    forecast=np.array(forecast)
    
    return forecast

multi_train()
load_multiple_models()
lstnet_forecast()