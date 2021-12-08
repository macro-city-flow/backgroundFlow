# -*- coding: utf-8 -*-
import pandas as pd
from pickle import load
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy.linalg as la
from numpy import max
import math
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from xgboost import train as xgb_train,DMatrix
from tqdm import trange

def MAPE(true,pred):
    return mean_absolute_error(true,pred)/max(true)*100

def preprocess_data(data, time_len, rate, seq_len, pre_len):
    train_size = int(time_len * rate)
    train_data = data[0:train_size]
    test_data = data[train_size:time_len]
    
    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0 : seq_len])
        trainY.append(a[seq_len : seq_len + pre_len])
    for i in range(len(test_data) - seq_len -pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0 : seq_len])
        testY.append(b[seq_len : seq_len + pre_len])
    return trainX, trainY, testX, testY
    
###### evaluation ######
def evaluation(a,b):
    mse = mean_squared_error(a,b)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(a, b)
    mape = MAPE(a,b)

    return rmse,mse,mae,mape
 
path = r'chengdu_data.pkl'
with open(path,'rb') as f:
    data =  load(f)

data = data.reshape(-1,625)
time_len = data.shape[0]
train_rate = 0.8
seq_len = 4
pre_len = 1

trainX,trainY,testX,testY = preprocess_data(data, time_len, train_rate, seq_len, pre_len)
####HA, SVR , ARIMA, SARIMA or xgboost
method = 'SVR' 
########### HA #############
if method == 'HA':
    result = []
    for i in range(len(testX)):
        a = testX[i]
        a1 = np.mean(a, axis=0) 
        result.append(a1)
    result1 = np.array(result)
    result1 = np.reshape(result1, [-1,625])
    testY1 = np.array(testY)
    testY1 = np.reshape(testY1, [-1,625])
    rmse,mse, mae, mape = evaluation(testY1, result1)  
    print(  'HA_rmse:%r\n'%rmse,
            'HA_mse:%r\n'%mse,
            'HA_mae:%r\n'%mae,
            'HA_mape:%r\n'%rmse,)


############ SVR #############
if method == 'SVR':  
    total_rmse, total_mae, total_acc, result = [], [],[],[]
    for i in trange(625):
        data1 = np.mat(data)
        a = data1[:,i]
        a_X, a_Y, t_X, t_Y = preprocess_data(a, time_len, train_rate, seq_len, pre_len)
        a_X = np.array(a_X)
        a_X = np.reshape(a_X,[-1, seq_len])
        a_Y = np.array(a_Y)
        a_Y = np.reshape(a_Y,[-1, pre_len])
        a_Y = np.mean(a_Y, axis=1)
        t_X = np.array(t_X)
        t_X = np.reshape(t_X,[-1, seq_len])
        t_Y = np.array(t_Y)
        t_Y = np.reshape(t_Y,[-1, pre_len])    
       
        svr_model=SVR(kernel='linear')
        svr_model.fit(a_X, a_Y)
        pre = svr_model.predict(t_X)
        pre = np.array(np.transpose(np.mat(pre)))
        pre = pre.repeat(pre_len ,axis=1)
        result.append(pre)
    result1 = np.array(result)
    result1 = np.reshape(result1, [625,-1])
    result1 = np.transpose(result1)
    testY1 = np.array(testY)


    testY1 = np.reshape(testY1, [-1,625])
    total = np.mat(total_acc)
    total[total<0] = 0
    rmse,mse,mae,mape = evaluation(testY1, result1)
    print('SVR_rmse:%r'%rmse,
          'SVR_mse:%r'%mse,
          'SVR_mae:%r'%mae,
          'SVR_mape:%r'%mape,
            )

######## XGBoost #########

if method == 'xgboost':
    params = {
        'verbosity':2,
        'objective':'reg:squarederror',
        'feature_selector':'thrifty',
        'booster':'gblinear',
        'top_k':10
    }
    for _ in trange(25):
        ...
    # TODO trans the data into xhboost accepted way
    # TODO fit
    # TODO give prediction and evaluate
    
######## ARIMA #########    
if method == 'ARIMA':
    rng = pd.date_range('1/11/2016', periods=5664, freq='15min')
    a1 = pd.DatetimeIndex(rng)
    data.index = a1
    num = data.shape[1]   
    rmse,mae,acc,r2,var,pred,ori = [],[],[],[],[],[],[]
    for i in range(156):
        ts = data.iloc[:,i]
        ts_log=np.log(ts)    
        ts_log=np.array(ts_log,dtype=np.float)
        where_are_inf = np.isinf(ts_log)
        ts_log[where_are_inf] = 0
        ts_log = pd.Series(ts_log)
        ts_log.index = a1
        model = ARIMA(ts_log,order=[1,0,0])
        properModel = model.fit()
        predict_ts = properModel.predict(4, dynamic=True)
        log_recover = np.exp(predict_ts)
        ts = ts[log_recover.index]
        er_rmse,er_mae,er_acc,r2_score,var_score = evaluation(ts,log_recover)
        rmse.append(er_rmse)
        mae.append(er_mae)
        acc.append(er_acc)
        r2.append(r2_score)
        var.append(var_score)
    print('arima_rmse:%r'%(np.mean(rmse)),
          'arima_mse:%r'%(np.mean(mae)),
          'arima_acc:%r'%(np.mean(acc1)),
          'arima_r2:%r'%(np.mean(r2)),
          'arima_var:%r'%(np.mean(var)))


if method == 'SARIMA':
    ...

