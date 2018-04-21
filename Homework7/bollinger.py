# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 14:45:03 2018

@author: andy
"""
import numpy as np
import pandas as pd

def bollinger_indicator(df_close,look_back,unit_of_std):
    #df_close.columns = ['price']
	mid_series = pd.Series(np.ones((len(df_close),))*np.NaN, index=df_close.index)
	upper_series = pd.Series(np.ones((len(df_close),))*np.NaN, index=df_close.index)
	lower_series = pd.Series(np.ones((len(df_close),))*np.NaN, index=df_close.index)
	indicator_series = pd.Series(np.ones((len(df_close),))*np.NaN, index=df_close.index)
	for i in range(look_back-1,len(df_close)):
	    mid = np.mean(df_close.iloc[i-look_back+1:i+1])
	    std = np.std(df_close.iloc[i-look_back+1:i+1])
	    upper = mid+unit_of_std*std
	    lower = mid-unit_of_std*std
	    mid_series.iloc[i] = mid
	    upper_series.iloc[i] = upper
	    lower_series.iloc[i] = lower
	    indicator_series.iloc[i] = (df_close.iloc[i] - mid) / (std * unit_of_std)
	return indicator_series
