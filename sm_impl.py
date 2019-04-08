import numpy as np
import pandas as pd
from statsmodels.tsa.statespace import sarimax as smx
import math
import pyflux as pf

from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime
import matplotlib.pyplot as plt

from dateutil.parser import parse

data_raw = pd.read_csv("input/series.csv", dayfirst=True)  # , index_col=0, dayfirst=True, parse_dates=True)
# data_raw['IPC'] = data_raw['IPC'].apply(lambda x: x/10000)
# data_raw['IPC_BASE'] = data_raw['IPC_BASE'].apply(lambda x: x/10000)
# data_raw['IPC_ADVERSO'] = data_raw['IPC_ADVERSO'].apply(lambda x: x/10000)

data_raw['fecha'] = data_raw['fecha'].apply(lambda x: parse(x, dayfirst=True))

# select development sample  comercial
# dev_sample = (data_raw['fecha'] >= '2014-01')&(data_raw['fecha'] <= '2018-03')
# # select data to foreacasting
# forecast = (data_raw['fecha'] >= '2018-04')&(data_raw['fecha'] <= '2020-12')


# select development sample hipotecario
# dev_sample = (data_raw['fecha'] >= '2014-07')&(data_raw['fecha'] <= '2018-03')
# # select data to foreacasting
# forecast = (data_raw['fecha'] >= '2018-04')&(data_raw['fecha'] <= '2020-12')

# select development sample consumo
dev_sample = (data_raw['fecha'] >= '2016-01')&(data_raw['fecha'] <= '2018-03')
# select data to foreacasting
forecast = (data_raw['fecha'] >= '2018-04')&(data_raw['fecha'] <= '2020-12')


# exog variables
exog_vars = ['CETES28', 'CETES364', 'TIPO_CAMBIO_USD', 'TASA_DESEMPLEO', 'IPC']


BASE = ['CETES28_BASE', 'CETES364_BASE', 'TIPO_CAMBIO_USD_BASE', 'TASA_DESEMPLEO_BASE', 'IPC_BASE']

ADVERSE = ['CETES28_ADVERSO', 'CETES364_ADVERSO', 'TIPO_CAMBIO_USD_ADVERSO', 'TASA_DESEMPLEO_ADVERSO', 'IPC_ADVERSO']

BASE_MAPPER = {
    'CETES28_BASE': 'CETES28',
    'CETES364_BASE': 'CETES364',
    'TIPO_CAMBIO_USD_BASE': 'TIPO_CAMBIO_USD',
    'TASA_DESEMPLEO_BASE': 'TASA_DESEMPLEO',
    'IPC_BASE': 'IPC'
}

ADVERSE_MAPPER = {
    'CETES28_ADVERSO': 'CETES28',
    'CETES364_ADVERSO': 'CETES364',
    'TIPO_CAMBIO_USD_ADVERSO': 'TIPO_CAMBIO_USD',
    'TASA_DESEMPLEO_ADVERSO': 'TASA_DESEMPLEO',
    'IPC_ADVERSO': 'IPC'
}


targets = ['comercial_calif', 'hipotecario_tot', 'consumo_puro_sh']
target = targets[2]


data_X = data_raw.loc[dev_sample, exog_vars]
data_y = data_raw.loc[dev_sample, [target]]

data_y[target] = data_y[target].apply(lambda x: math.log(x, math.e))

data = pd.concat([data_y, data_X], axis=1)

# data_ylog = data_y[target].apply(math.log10)
# data_Xlog = data_X.loc[:,exog_vars].apply(math.log10)

exog = data_X


# comercial
# order = (1, 1, 0)
# seasonal_order = (1, 1, 0, 12)

# hipotecario
# order = (1, 0, 0)
# seasonal_order = (1, 1, 0, 12)

model = smx.SARIMAX(data_y, exog=data_X, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0),
                    enforce_stationarity=False, enforce_invertibility=True)

# model_log = smx.SARIMAX(data_ylog, exog=data_Xlog, order=(1, 1, 0), seasonal_order=(1, 0, 1, 12),
#                     enforce_stationarity=False, enforce_invertibility=False)

# model_wox = smx.SARIMA(data_y, order=(1, 1, 0), seasonal_order=(1, 1, 0, 12))

model_fit = model.fit()

# model_log_fit = model_log.fit()

# model_wox_fit = model_wox.fit()

# yhat = model_fit.forecast()

summary = model_fit.summary()

# summary_wox = model_wox_fit.summary()

print(summary)
# print(summary_wox)

data_base_forecast_X = data_raw.loc[forecast, BASE].rename(columns=BASE_MAPPER)
data_adverse_forecast_X = data_raw.loc[forecast, ADVERSE].rename(columns=ADVERSE_MAPPER)

# data_base_forecast_X_log = data_base_forecast_X.apply(lambda x: math.log10(x))
# data_adverse_forecast_X_log = data_adverse_forecast_X.apply(lambda x: math.log10(x))

# comercial
# end = 83

# consumo
end = 59

# hipotecraio
# end = 64


base_preds = model_fit.get_prediction(start=0, end=end, exog=data_base_forecast_X)
adverse_preds = model_fit.get_prediction(start=0, end=end, exog=data_adverse_forecast_X)

# baselog_preds = model_log_fit.get_prediction(start=0, end=71, exog=data_base_forecast_X_log)
# adverselog_preds = model_log_fit.get_prediction(start=0, end=71, exog=data_adverse_forecast_X_log)
#
# baselog_preds_pred_mean_transformed = baselog_preds.predicted_mean.apply(lambda x: math.pow(10, x))
# adverselog_preds_pred_mean_transformed = adverselog_preds.predicted_mean.apply(lambda x: math.pow(10, x))

# wox_preds = model_fit.get_prediction(start=0, end=83)


# data_y0 = data_y.reset_index()
# plt.figure(figsize=(15, 5))
# plt.plot(range(0,39), base_preds.predicted_mean[1:40], 'r')
# plt.plot(base_preds.predicted_mean[39:], 'g')
# plt.plot(adverse_preds.predicted_mean[39:], 'c')
# plt.plot(data_y0[target], 'b')
# plt.show()

# comercial
# init = 51

# consumo
init = 27

# hipotecario
# init = 32

var_vect = preds.se_mean[init:]
adverse_var_vect = adverse_preds.se_mean[init:]

base_var = np.var(var_vect)
adverse_var = np.var(adverse_var_vect)
var1 = math.exp(base_var/2)
var2 = math.exp(adverse_var/2)

dev_data = base_preds.predicted_mean[1:init].apply(math.exp)

data_y[target] = data_y[target].apply(math.exp)
base_pred = base_preds.predicted_mean[init:].apply(lambda x: math.exp(x)*var1)
adverse_pred = adverse_preds.predicted_mean[init:].apply(lambda x: math.exp(x)*var2)

data_y0 = data_y.reset_index()
plt.figure(figsize=(15, 5))
plt.plot(dev_data, 'r')
plt.plot(base_pred, 'g')
plt.plot(adverse_pred, 'c')
plt.plot(data_y0[target], 'b')
plt.show()

# plt.plot(range(0,39), baselog_preds_pred_mean_transformed[1:40], 'r')
# plt.plot(baselog_preds_pred_mean_transformed[39:], 'g')
# plt.plot(adverselog_preds_pred_mean_transformed[39:], 'c')
# plt.plot(data_y0[target], 'b')
# plt.show()



print('finish')
