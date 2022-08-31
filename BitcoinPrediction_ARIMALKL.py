# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARMA, ARIMA
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def adf_test(ts):
    adftest = adfuller(ts)
    adf_res = pd.Series(adftest[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
    for key, value in adftest[4].items():
        adf_res['Critical Value (%s)' % key] = value
    return adf_res

# 参数调优：AIC
def get_pdq(time_series):
    pmax=10
    qmax=10
    aic_matrix=[]
    for p in range(pmax+1):
        tmp = []
        for q in range(qmax+1):
            try:
                tmp.append(ARIMA(time_series,order=(p, 1, q)).fit().aic)
                print("times", "p", p, "q", q)
            except:
                tmp.append(None)

        aic_matrix.append(tmp)
    aic_matrix=pd.DataFrame(aic_matrix)
    p,q=aic_matrix.stack().idxmin() #最小值的索引
    print('用AIC方法得到最优的p值是%d,q值是%d'%(p,q))

# 加入差分的 ARMA 模型
def ts_arima(ts, p, d, q):
    arima = ARIMA(ts, (p, 1, q)).fit(disp=0)
    # ts_predict_arima = arima.predict(2, 120, dynamic=True) # 预测
    ts_predict_arima, e, p = arima.forecast(8)  # 预测未来
    print(ts_predict_arima)
    return ts_predict_arima


file_path = 'bitcoin_2012-01-01_to_2018-10-31.csv'
data = pd.read_csv(file_path)

data['Timestamp'] = pd.to_datetime(data.Timestamp)
data.set_index(data.Timestamp, inplace=True)
print(data.head())

# 分别按照月、季度和年进行降采样
data_month = data.resample(rule='M').mean()
data_Q = data.resample(rule='Q-DEC').mean()
data_year = data.resample(rule='A-DEC').mean()

ad = adf_test(data_month.Weighted_Price)
print(ad)

Weighted_Price = data_month['Weighted_Price'].diff(1).dropna()
ad1 = adf_test(Weighted_Price)
print(ad1)

get_pdq(Weighted_Price)

pre = ts_arima(data_month['Weighted_Price'], 1, 1, 2)
# data_shift = data_month['Weighted_Price'].shift(1)
# pre_recover = pre.add(data_shift)
print(pre)

# 结果可视化
plt.figure(figsize=(15, 15))
# plt.plot(data_month['Weighted_Price'], label='实际值')
# plt.legend()
plt.plot(pre, color='r', linestyle='dashed', label='预测值')
plt.legend()
plt.title('比特币金额（月）')
plt.xlabel('时间')
plt.ylabel('美金')
plt.show()