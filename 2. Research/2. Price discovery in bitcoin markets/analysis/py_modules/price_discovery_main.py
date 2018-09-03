# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:53:41 2017

@author: Rocku
"""


import pandas as pd
import matplotlib.pyplot as plt
from price_discovery import price_discovery, lead_lag_indicator
from price_discovery_data_preprocess import price_data_generator
import datetime as dt


#%%
data_root = 'usd/'
root = 'bitfinex_1r/'


# Data generation process
startdate = dt.datetime(2016,7,15)
enddate = dt.datetime(2015,3,15)
price_data_generator(data_root, root, startdate, enddate, interval='1Min')


#%% Price discovery process
file = root + 'exc_1min.csv'
to_csv = root + 'is.csv'
price_discovery(file, to_csv, 10, interval='1D')


# Data aggregation by iteration
df_full = pd.read_csv(to_csv)
df_full['date'] = pd.to_datetime(df_full['date'])
df_full['date_month'] = df_full['date'] - pd.offsets.DateOffset(1)

df_grouped = df_full.groupby(['exchange','date']).mean()
df_grouped.reset_index(inplace=True)
df_grouped['date_month'] = df_grouped['date'] - pd.offsets.DateOffset(1)

df_grouped.to_csv(root +'agg_is_25.csv')


#%% Lead lag analysis process
file = root + 'exc_5min.csv'
to_csv = root + 'lead_lag.csv'
lead_lag_indicator(file, to_csv,10)

#%% Price discovery - Hasbrouck



#%%
df_grouped_by_ex = df_grouped.groupby(['exchange'])
graph_lines = ['rd--', 'go-', 'bv--']
for exchange, data in df_grouped_by_ex:
    plt.hold()
    plt.plot(data['date_month'],data['IS'],markersize=10, label=exchange)
    plt.legend()

plt.show()


#%%
#df_mtgox = df_grouped.loc[df_grouped['exchange'] == 'p_mtgox']
#df_mtgox.reset_index(inplace=True,drop=True)
##df_mtgox.drop(0,inplace=True)
#
#df_bitfinex = df_grouped.loc[df_grouped['exchange'] == 'p_bitfinex']
#df_bitfinex.reset_index(inplace=True,drop=True)
##df_bitfinex.drop(0,inplace=True)
#
#df_bitstamp = df_grouped.loc[df_grouped['exchange'] == 'p_bitstamp']
#df_bitstamp.reset_index(inplace=True,drop=True)
##df_bitstamp.drop(0,inplace=True)
#
#df_btce = df_grouped.loc[df_grouped['exchange'] == 'p_btce']
#df_btce.reset_index(inplace=True,drop=True)
##df_btce.drop(0,inplace=True)
#
#plt.plot(df_mtgox['date_month'],df_mtgox['IS'],'rd--',markersize=10, label='mtgox')
#plt.hold
#plt.plot(df_bitfinex['date_month'],df_bitfinex['IS'],'go-', markersize=10, label='bitfinex')
#plt.plot(df_bitstamp['date_month'],df_bitstamp['IS'],'bv--',markersize=10, label='bitstamp')
#plt.plot(df_btce['date_month'],df_btce['IS'],'k+-',markersize=10, label='btce')
#plt.legend(loc=0)
#plt.grid()
#
##%%
#plt.plot(df_mtgox['date_month'],df_mtgox['IS/AS'],'rd--', label='mtgox',markersize=10)
#plt.hold
#plt.plot(df_bitfinex['date_month'],df_bitfinex['IS/AS'],'go-', label='bitfinex',markersize=10)
#plt.plot(df_bitstamp['date_month'],df_bitstamp['IS/AS'],'bv--', label='bitstamp',markersize=10)
#plt.plot(df_btce['date_month'],df_btce['IS/AS'],'k+-', label='btce',markersize=10)
#plt.legend(loc=0)
#plt.grid()


##%%
#
#df_psi = df_full.loc[df_full['exchange']=='p_mtgoxUSD']
#df_psi.reset_index(inplace=True,drop=True)
#plt.plot(df_psi['psi_i'])
