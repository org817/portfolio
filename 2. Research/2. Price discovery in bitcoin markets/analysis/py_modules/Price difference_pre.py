import pandas as pd
import glob
import sys
#%% 

files = glob.glob('C:/Users/Rocku/Dropbox/Data_analysis/mtgox/usd/*.csv')
nfile = len(files)
columns =['Exchange', 'Currency', 'Datapoints', 'Total_volume', 'Min_volume', 'Max_volume',
          'First_date', 'Last_date','Dates', 'VolumePerDay', 'VolumePerTrade','TradePerDay']

df_ss = pd.DataFrame(columns=columns)

success = 0
failed = 0
failed_name = []

print('------------start -------------')

for i, file in enumerate(files):
#     print(gz_file)
    if (i%10 == 1):
        print(str(nfile), 'files total:', success+failed, 'success:', success, 'failed:', failed)
    
    try:            
        data = pd.read_csv(file)
        data['date'] = pd.to_datetime(data['date'])
        data['datadate'] = data['date'].dt.date
        data['time'] = data['date'].dt.time
        
        data.sort_values(by=['datadate','time'], ascending=False, inplace=True)
        described = data.describe()
        
        groupped = data.groupby('datadate')
        counted = groupped[['volume']].count()
        counted.reset_index(inplace=True)
        summed = groupped[['volume']].sum()
        summed.reset_index(inplace=True)
        
        merged = pd.merge(summed, counted, how='inner', on='datadate')
        merged.rename(columns={'volume_x':'sum_volume', 'volume_y':'total_trade'}, inplace=True)
        merged['VolPerTrade'] = merged['sum_volume'] / merged['total_trade']
        
        data_dict = {}
        data_dict['Exchange'] = file.split('\\')[-1].split('.')[0]
        data_dict['Currency'] = data_dict['Exchange'][-3:]
        data_dict['Datapoints'] = described.loc['count','price']
        data_dict['Total_volume'] = data['volume'].sum()
        data_dict['Min_volume'] = described.loc['min','volume']
        data_dict['Max_volume'] = described.loc['max','volume']
        data_dict['First_date'] = str(data.loc[0,'date'])
        data_dict['Last_date'] = str(data.loc[len(data)-1,'date'])
        data_dict['Dates'] = (data.loc[len(data)-1,'date'] - data.loc[0,'date']).days
        data_dict['TradePerDay'] = merged['total_trade'].mean()
        data_dict['VolumePerDay'] = merged['sum_volume'].mean()
        data_dict['VolumePerTrade'] = merged['VolPerTrade'].mean()

        df_new = pd.DataFrame([data_dict],columns=columns)
        df_ss = df_ss.append(df_new)
        success += 1
        
        print('****',file, "Success ****")

        del data, data_dict, described, merged, summed, counted
        
    except:
        failed_name.append(file)
        e = sys.exc_info()[1]

        print('****',file, "Failed **** : ", e)
        failed += 1

print('------------end -------------')

#%%
df_ss.to_csv('summary_stat_bitcoin_exchanges.csv', header='column_names', index=False, encoding='utf-8')
