# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from itertools import product
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


file_path = 'bitcoin_2012-01-01_to_2018-10-31.csv'
data = pd.read_csv(file_path)

data['Timestamp'] = pd.to_datetime(data.Timestamp)
data.set_index(data.Timestamp, inplace=True)
print(data.head())


# 分别按照月、季度和年进行降采样
data_month = data.resample(rule='M').mean()
data_Q = data.resample(rule='Q-DEC').mean()
data_year = data.resample(rule='A-DEC').mean()

# # 对于不同时间频率，分别可视化
fig = plt.figure(figsize=[15, 15])
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.suptitle('比特币走势', fontsize=20)
plt.subplot(221)
plt.plot(data.Weighted_Price, '-', label='按天')
plt.legend()
plt.subplot(222)
plt.plot(data_month.Weighted_Price, '-', label='按月')
plt.legend()
plt.subplot(223)
plt.plot(data_Q.Weighted_Price, '-', label='按季度')
plt.legend()
plt.subplot(224)
plt.plot(data_year.Weighted_Price, '-', label='按年')
plt.legend()
plt.show()

# ARMA参数范围设定
p_arrange = range(0, 3)
q_arrange = range(0, 3)

para = list(product(p_arrange, q_arrange))
best_aic = float('inf')
reslut = []
for parameter in para:
    try:
        model = ARMA(data_month.Weighted_Price, order=(parameter[0], parameter[1])).fit()
    except ValueError:
        print('参数错误')
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_para = parameter
    reslut.append([parameter, aic])

print('最优模型：', best_model.summary())

# 预测未来走势
future_date_list = [datetime(2018, 11, 30), datetime(2018, 12, 31), datetime(2019, 1, 31), datetime(2019, 2, 28),
                    datetime(2019, 3, 31), datetime(2019, 4, 30), datetime(2019, 5, 31), datetime(2019, 6, 30)]
df_future = best_model.predict(start=len(data_month)-1, end=90)
# df_future = best_model.forecast(10)

# 结果可视化
plt.figure(figsize=(15, 15))
plt.plot(data_month['Weighted_Price'], label='实际值')
plt.legend()
plt.plot(df_future, color='r', linestyle='dashed', label='预测值')
plt.legend()
plt.title('比特币金额（月）')
plt.xlabel('时间')
plt.ylabel('美金')
plt.show()