# -*- coding: utf-8 -*-
"""
For Calculating information share on Bitcoin exchanges
"""

import pandas as pd
import numpy as np
import scipy.optimize as optimize

#%% import data
#file = '5min_without_gox/exc_5min.csv'
def price_discovery(file, to_csv, attempt=1, interval='1M'):
#    file = 'dated/exc_5min.csv'
    
    price_data = pd.read_csv(file)
    price_data['date'] = pd.to_datetime(price_data['date'])
    price_data.sort_values(by='date', inplace=True, ascending=True)
    
    price_columns = list(price_data.columns)
    
    p_columns = [p[:-3] for p in price_columns if p[0] == 'p']
#    v_columns = [v[:-3] for v in price_columns if v[0] == 'v']
    nex = len(p_columns)
        
    p_col = [p for p in list(price_data.columns) if p[0] == 'p']
    v_col = [v for v in list(price_data.columns) if v[0] == 'v']
    d_col = [d for d in list(price_data.columns) if d[0:2] == 'd_']
    for d_c in d_col:
        price_data[d_c] = pd.to_datetime(price_data[d_c])

    price_data['last_ex'] = 'p_'+price_data[d_col].idxmax(axis=1).str[2:]
    
    for idx, row in price_data.iterrows():
        price_data.loc[idx,'m'] = row[row['last_ex']]
#        price_data.loc[idx,'m'] = price_data.loc[idx, row['last_ex']]
        
    price_data.set_index('date', inplace=True, drop=True)
    
    del price_data['last_ex']
    
    #%% Initialze full dataframe for the monthly information share
    col = ['date', 'rho_ii','pi','sigma', 'E[yj_t,i_t-1]', 'E[yj_t,i_t]', 'roll',
            'gamma_i', 'gamma_j', 'psi_i', 'psi_j', 'omega_i_j', 'omega^e_i', 'IS', 'IS/AS']
    
    df_full = pd.DataFrame(columns=col)
    
    # Calculate IS
    data_monthly = price_data.groupby(pd.TimeGrouper(interval))
    
    for date, price_data in data_monthly:
        price_data.reset_index(inplace=True)
    #    price_data.sort_values(by='date', inplace=True, ascending=False)
        
        trading_volume = price_data.loc[:,v_col]
        v_sum = trading_volume.sum(axis=0)
        v_sum = v_sum / v_sum.sum(axis=0)
        v_freq = trading_volume[trading_volume > 0].count(axis = 0)
        v_freq = v_freq/v_freq.sum(axis=0)
        weights = 0.5*v_sum + 0.5*v_freq
        pi = np.array(weights)
        
        # Market price formation
#        def market_price(exchange, v_col, weights):
#            other_weights = {'p_'+item[2:] : weights.ix[item] / (1 - weights.ix[exchange]) for item in v_col if item != exchange}
#            market_price = sum([price_data[exchange] * other_weights[exchange] for exchange in other_weights.keys()])
#            return market_price
#        
#        for exchange in v_col:
#            other = market_price(exchange, v_col, weights)
#            price_data['m_'+exchange[2:]] = other    
            
        price_data.set_index('date', inplace=True, drop=True)
        
        p_col = [p for p in list(price_data.columns) if (p[0] !='v') & (p[0] !='d')]
        # Calculate lag variables of log price difference
        for price in p_col:
            price_data[price] = np.log(price_data[price])
            price_data['diff_'+price] = price_data[price] - price_data[price].shift(1)
            price_data['diff_lag1_'+price] = price_data['diff_'+price].shift(1)
            price_data['diff_lag2_'+price] = price_data['diff_'+price].shift(2)
            
        price_data.fillna(method='bfill',inplace=True)
        
        E_yj_yi = []
        E_yj_yi_lag1_t = []
        E_yj_yi_lag2_t = []
        E_yi_yj_lag2_t = []
        rho = []
        rho2 = []
        mean_ = []
        stdev_ = []
        roll = []
        
        # calculate E[yj_t,i_t], E[yj_t,i_t-1], rho_ii
        p_col = [p for p in list(price_data.columns) if p[0] == 'p']
        
        for col in p_col:
#            diff = price_data[['diff_m_'+ col[2:] ,'diff_'+col]]
            diff = price_data[['diff_m','diff_'+col]]
            E_yj_yi.append(np.cov(diff.T)[0,1])
        
#            diff = price_data[['diff_m'+ col[2:], 'diff_lag1_'+col]]
            diff = price_data[['diff_m', 'diff_lag1_'+col]]
            E_yj_yi_lag1_t.append(np.cov(diff.T)[0,1])
        
#            diff = price_data[['diff_m_'+ col[2:], 'diff_lag2_'+col]]
            diff = price_data[['diff_m', 'diff_lag2_'+col]]
            E_yj_yi_lag2_t.append(np.cov(diff.T)[0,1])
        
#            diff = price_data[['diff_'+col, 'diff_lag2_m_' + col[2:]]]
            diff = price_data[['diff_'+col, 'diff_lag2_m']]
            E_yi_yj_lag2_t.append(np.cov(diff.T)[0,1])
            
            diff = price_data[['diff_'+ col, 'diff_lag1_'+col]]
            roll.append(2*np.sqrt(abs(np.cov(diff.T)[0,1])))
        
            rho.append(price_data['diff_'+col].autocorr(1))
            rho2.append(price_data['diff_'+col].autocorr(2))
            mean_.append(price_data['diff_'+col].mean())
            stdev_.append(price_data['diff_'+col].std())
            
        E_yj_yi = np.array(E_yj_yi)
        E_yj_yi_lag1_t = np.array(E_yj_yi_lag1_t)
        gamma_j = -np.array(E_yj_yi_lag2_t)
        gamma_i = -np.array(E_yi_yj_lag2_t)
        rho = np.array(rho)
        rho2 = np.array(rho2)
        mean_ = np.array(mean_)
        stdev_ = np.array(stdev_)
        roll = np.array(roll)
        
        df = pd.DataFrame()
        df['E[yj_t,i_t]'] = E_yj_yi
        df['E[yj_t,i_t-1]'] = E_yj_yi_lag1_t
        df['rho_ii'] = rho
        df['rho_2'] = rho2
        p_col = ['diff_'+p for p in list(price_data.columns) if p[0] == 'p']
        df['sigma'] = price_data[p_col].sum(axis=0).var()
        df['gamma_i'] = gamma_i
        df['gamma_j'] = gamma_j
        df['pi'] = pi
        df['mean'] = mean_
        df['stdev'] = stdev_
        df['roll'] = roll
        #### from paper ####
        #pi_paper = np.array([0.352,0.240,0.215,0.109])
        #df['pi'] = pi_paper/pi_paper.sum()
        
        
        # Define opmization constraints and objective function
        def toVector(x_mat):
            return x_mat.reshape(nex*4,1)
        
        def objective(x):
            sum_value1 = 0
            for i in range(0,nex):
                sum_value1 += (1 + (x[(4*i)]/df['sigma'][i])) * df['pi'][i] 
            return abs(sum_value1 - 1)
        
        def constraint1(x):
            sum_value = 0
            for i in range(0,nex):
                sum_value += x[(4*i)]*df['pi'][i]
            return sum_value
        
    #    def constraint2(x):
    #        sum_value = 0
    #        for i in range(0,nex):
    #            sum_value += abs(x[(4*i)]*df['pi'][i])
    #        return sum_value
        
        def constraint3(x):
            sum_up1 = 0
            for i in range(0,nex):
                sum_up1 += abs(2*x[(4*i)+2] + x[(4*i)+1] + x[(4*i)] + df['sigma'][i] - df['E[yj_t,i_t]'][i])
            return sum_up1
        
        def constraint4(x):
            sum_up2 = 0
            for i in range(0,nex):
                sum_up2 += abs(x[(4*i)+1] + x[(4*i)+2] + df['E[yj_t,i_t-1]'][i] - df['gamma_j'][i])
            return sum_up2
        
        def constraint5(x):
            sum_up3 = 0
            for i in range(0,nex):
                sum_up3 += abs(((x[(4*i)+3] + x[(4*i)+0] + df['gamma_i'][i])/(df['sigma'][i] + 2*(x[(4*i)+3]+x[(4*i)+0]))) + df['rho_ii'][i])
            return sum_up3
        
        def constraint6(x):  
    #        return x[(4*1)-1]
            x = x.reshape(nex,4)
            return x[:,3]
    
            
        con1 = {'type': 'eq', 'fun': constraint1} 
    #    con2 = {'type': 'eq', 'fun': constraint2} 
        con3 = {'type': 'eq', 'fun': constraint3}
        con4 = {'type': 'eq', 'fun': constraint4} 
        con5 = {'type': 'eq', 'fun': constraint5} 
    
        con6 = {'type': 'ineq', 'fun': constraint6}
    
        
        cons = ([con1, con3, con4, con5, con6])
        
        print(date)
        for att in range(1,attempt+1):
            if (att % 10) == 1:
                print(att)
            
            flag = True
            sol = np.zeros([nex,4])
            try_no = 1
            
            while flag:
    #            print(date)
                try_no += 1
                
                x0 = np.zeros([nex,4])
                for i in range(nex):
                    for j in range(4):
                        x0[i,j] = 0.1*np.random.rand()*df['sigma'][0]
            
                solution = optimize.minimize(fun=objective, x0=toVector(x0), options={'disp': False, 'maxiter': 10}, constraints=cons, method='SLSQP', tol=1e-10)
                sol = solution.x.reshape(nex,4)
            
                if (
                    (abs(sol[:,3]/df['sigma']) < 1).all() &
#                    (abs(sol[:,0]/df['sigma']) > 0.05).all() &
                    (sol[:,3] > 0).all() &
                    (((1 + sol[:,0]/df['sigma']) * df['pi']) > 0).all()):
                    print(sol)
                    
                    df['psi_i'] = sol[:,0]
                    df['psi_j'] = sol[:,1]
                    df['omega_i_j'] = sol[:,2]
                    df['omega^e_i'] = sol[:,3]
                
                    df['IS'] = (1 + df['psi_i']/df['sigma']) * df['pi']
                    df['IS/AS'] = df['IS']/df['pi']
                    df['date'] = date
                    df['exchange'] = p_columns
                    df['attempt'] = att
                    break
                
                elif (try_no > 100):
                    df['psi_i'] = np.NaN
                    df['psi_j'] = np.NaN
                    df['omega_i_j'] = np.NaN
                    df['omega^e_i'] = np.NaN
                
                    df['IS'] = np.NaN
                    df['IS/AS'] = np.NaN
                    df['date'] = date
                    df['exchange'] = p_columns
                    df['attempt'] = att
                    break
                
            df_full = df_full.append(df, ignore_index=False)
        
        del df, diff, price_data, E_yj_yi, E_yj_yi_lag1_t, E_yj_yi_lag2_t, E_yi_yj_lag2_t, rho, pi, trading_volume, gamma_i, gamma_j, sol, solution, v_sum, x0 
        
    df_full.to_csv(to_csv, header='column_names', index=False, encoding='utf-8')



#%% lead lag indicator
def lead_lag_indicator(file, to_csv, lag=5):
    price_data = pd.read_csv(file)
    price_data['date'] = pd.to_datetime(price_data['date'])
    price_data.sort_values(by='date', inplace=True, ascending=True)
            
    p_col = [p for p in list(price_data.columns) if p[0] == 'p']
    v_col = [v for v in list(price_data.columns) if v[0] == 'v']
    
    price_data.set_index('date', inplace=True, drop=True)
    
    data_monthly = price_data.groupby(pd.TimeGrouper('1M'))
    
    df_full=pd.DataFrame()
    
    for date, price_data in data_monthly:
        price_data.reset_index(inplace=True)        
    
        trading_volume = price_data.loc[:,v_col]
        v_sum = trading_volume.sum(axis=0)
        v_sum = v_sum / v_sum.sum(axis=0) 
        v_freq = trading_volume[trading_volume > 0].count(axis = 0)
        v_freq = v_freq/v_freq.sum(axis=0)
        weights = 0.5*v_sum + 0.5*v_freq
                
        # Market price formation
        def market_price(exchange, v_col, weights):
            other_weights = {'p_'+item[2:] : weights.ix[item] / (1 - weights.ix[exchange]) for item in v_col if item != exchange}
            market_price = sum([price_data[exchange] * other_weights[exchange] for exchange in other_weights.keys()])
            return market_price
        
        for exchange in v_col:
            other = market_price(exchange, v_col, weights)
            price_data['m_'+exchange[2:]] = other    
            
        price_data.set_index('date', inplace=True, drop=True)
        
        p_col = [p for p in list(price_data.columns) if p[0] !='v']
        for price in p_col:
            price_data[price] = np.log(price_data[price])
            
        p_col = [p for p in list(price_data.columns) if p[0] == 'p']
        for price in p_col:
            for i in range(-1*lag,lag+1):
                shifted = price_data['m_' + price[2:]].shift(i*100)
                shifted.fillna(value=0, inplace = True)
                df_full.loc[i+1,price] = np.corrcoef(price_data[price], shifted)[0,1]
                
    df_full.to_csv(to_csv, header='column_names', index=False, encoding='utf-8')



def hasbrouck():
    pass