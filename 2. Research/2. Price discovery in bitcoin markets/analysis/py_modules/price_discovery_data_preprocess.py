
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab


def price_data_generator(data_root, root, startdate, enddate, interval='5Min'):
    file = 'summary_stat_bitcoin_exchanges.csv'
    
    ex_ss = pd.read_csv(file)
    ex_ss['First_date'] = pd.to_datetime(ex_ss['First_date'])
    ex_ss['Last_date'] = pd.to_datetime(ex_ss['Last_date'])
    
    ex_ss.sort_values('Total_volume',inplace=True,ascending=False)
    
    ex_dated = ex_ss.loc[(ex_ss['First_date'] <= startdate) &
                         (ex_ss['Last_date'] >= enddate) & 
                         (ex_ss['Total_trade'] > 1000000) &
                         (ex_ss['TradePerDay'] > 1000)]
    #ex_dated = ex_dated[:-1]
    
    #ex_dated = ex_ss[:10]
    #%% extract 5min price data
    
    usd_full_files = (data_root + ex_dated['Exchange'] + '.csv').tolist()
    
    success = 0
    failed = 0
    failed_name = []
    
    for i, file in enumerate(usd_full_files):
        print(file)
        try:
            data = pd.read_csv(file, date_parser=['date'])
            data['date'] = pd.to_datetime(data['date'])
            data['date_origin'] = data['date']
            data_within = data[(data['date'] >= startdate) & (data['date'] <= enddate)]
            data_within.sort_values(by='date', inplace=True, ascending=False)
    
            data_within.set_index('date',inplace=True, drop=True)
            data_within2 = data_within.groupby(pd.TimeGrouper(interval))
#            data_counted = data_within2.count()
#            data_summed = data_within2.sum()
#            data_counted.to_csv(root + 'counted_' + file.split('/')[-1], header='column_names', encoding='utf-8')
#            data_summed.to_csv(root + 'summed_' + file.split('/')[-1], header='column_names', encoding='utf-8')
            
            data_within3 = data_within2.last()
            data_within3.to_csv(root + file.split('/')[-1], header='column_names', encoding='utf-8')
            
            print('Success : ', file)
    
            del data
        except:
            failed += 1
            failed_name.append(file)
            print('failed : ', file)
            
    print('end')
    
    #%% concat 5min price data
    
    indi_5min_files = (root + ex_dated['Exchange'] + '.csv').tolist()
    
    df_full = pd.DataFrame()
    
    for i, file in enumerate(indi_5min_files):
        print(i,file)
        data = pd.read_csv(file, date_parser=['date'], index_col=None)
        data['date'] = pd.to_datetime(data['date'])
        data.sort_values(by='date', inplace=True, ascending=False)
        data['price'].fillna(method='bfill', inplace=True)
        data['date_origin'].fillna(method='bfill', inplace=True)
        data['volume'].fillna(value=0, inplace=True)
        data['return'] = np.log(1+(data['price'] - data['price'].shift(-1))/data['price'].shift(-1))
        data['return'].fillna(value=0, inplace=True)
        
        data.rename(columns={
                'price': 'p_'+file.split('/')[-1].split('.')[0],
                'volume': 'v_'+file.split('/')[-1].split('.')[0],
                'date_origin': 'd_'+file.split('/')[-1].split('.')[0],
                'return': 'r_'+file.split('/')[-1].split('.')[0]
                }, inplace= True)
                
        if i == 0:
            df_full = data
        else:
            df_full.sort_values(by='date', inplace=True, ascending=False)
            data.sort_values(by='date', inplace=True, ascending=False)
            df_full = pd.merge(left=df_full, right=data, on='date', how='inner')
    
    print('end')
    
    df_full.to_csv(root + 'exc_1min.csv', header='column_names', encoding='utf-8', index=False)
