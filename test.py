import numpy as np
import math
import pyflux as pf
import pandas as pd
from statsmodels.tsa.statespace import sarimax as smx
from pandas_datareader import DataReader
from datetime import datetime
import matplotlib.pyplot as plt
from dateutil.parser import parse

targets = ['Reserva_Consumo', 'Reserva_Comercial', 'Reserva_Hipotecario']
target = targets[2]

exog_vars = [
    'DOLAR_TC',
    'EURO_TC',
    'CETES_28',
    'CETES_91',
    'CETES_182',
    'CETES_364',
    'IPC',
    'TIIE_28',
    'TIIE_91',
    'TIIE_182',
    'T_DESEMPLEO',
    'LIBOR_1',
    'LIBOR_3',
    'LIBOR_6',
    'T-BILL_3',
    'T-BILL_12',
    'VIX',
    'S_P500',
    'INFLACION_ANUAL',
]

exog_adv_vars = [
    'DOLAR_TC_ADV',
    'EURO_TC_ADV',
    'CETES_28_ADV',
    'CETES_91_ADV',
    'CETES_182_ADV',
    'CETES_364_ADV',
    'IPC_ADV',
    'TIIE_28_ADV',
    'TIIE_91_ADV',
    'TIIE_182_ADV',
    'T_DESEMPLEO_ADV',
    'LIBOR_1_ADV',
    'LIBOR_3_ADV',
    'LIBOR_6_ADV',
    'T-BILL_3_ADV',
    'T-BILL_12_ADV',
    'VIX_ADV',
    'S_P500_ADV',
    'INFLACION_ANUAL_ADV',
]

mapper = {
    'DOLAR_TC_ADV': 'DOLAR_TC',
    'EURO_TC_ADV': 'EURO_TC',
    'CETES_28_ADV': 'CETES_28',
    'CETES_91_ADV': 'CETES_91',
    'CETES_182_ADV': 'CETES_182',
    'CETES_364_ADV': 'CETES_364',
    'IPC_ADV': 'IPC',
    'TIIE_28_ADV': 'TIIE_28',
    'TIIE_91_ADV': 'TIIE_91',
    'TIIE_182_ADV': 'TIIE_182',
    'T_DESEMPLEO_ADV': 'T_DESEMPLEO',
    'LIBOR_1_ADV': 'LIBOR_1',
    'LIBOR_3_ADV': 'LIBOR_3',
    'LIBOR_6_ADV': 'LIBOR_6',
    'T-BILL_3_ADV': 'T-BILL_3',
    'T-BILL_12_ADV': 'T-BILL_12',
    'VIX_ADV': 'VIX',
    'S_P500_ADV': 'S_P500',
    'INFLACION_ANUAL_ADV': 'INFLACION_ANUAL',
}

target_exog_vars = {
    'Reserva_Consumo': ['DOLAR_TC', 'EURO_TC', 'LIBOR_1', 'LIBOR_3', 'S_P500',],
    'Reserva_Comercial': ['DOLAR_TC', 'VIX', 'S_P500', 'INFLACION_ANUAL', 'T_DESEMPLEO', 'CETES_364'],
    'Reserva_Hipotecario': ['CETES_91', 'T_DESEMPLEO', 'DOLAR_TC', 'T-BILL_12', 'IPC' ]
}

target_adv_exog_vars = {
    'Reserva_Consumo': ['{}_ADV'.format(var)  for var in target_exog_vars['Reserva_Consumo']],
    'Reserva_Comercial': ['{}_ADV'.format(var)  for var in target_exog_vars['Reserva_Comercial']],
    'Reserva_Hipotecario': ['{}_ADV'.format(var)  for var in target_exog_vars['Reserva_Hipotecario']]
}

target_fecha_inicial = {
    'Reserva_Consumo': '2014-07',
    'Reserva_Comercial': '2014-01',
    'Reserva_Hipotecario': '2016-01'
}


target_arima_params = {
    'Reserva_Consumo': {
        'order': (1, 0, 1),
        'seasonal_order': (1, 0, 1, 12)
    },
    'Reserva_Comercial': {
        'order': (1, 0, 1),
        'seasonal_order': (1, 0, 1, 12)
    },
    'Reserva_Hipotecario': {
        'order': (1, 0, 1),
        'seasonal_order': (1, 0, 1, 12)
    }
}

data_raw = pd.read_csv("input/Base_Trabajo.csv", dayfirst=True)

data_raw['IPC'] = data_raw['IPC'].apply(lambda x: x/10000)
data_raw['IPC_ADV'] = data_raw['IPC_ADV'].apply(lambda x: x/10000)
data_raw['S_P500'] = data_raw['S_P500'].apply(lambda x: x/1000)
data_raw['S_P500_ADV'] = data_raw['S_P500_ADV'].apply(lambda x: x/1000)

data_raw['fecha'] = data_raw['fecha'].apply(lambda x: parse(x, dayfirst=True))

dev_sample = (data_raw['fecha'] >= target_fecha_inicial[target])&(data_raw['fecha'] <= '2019-02')
# select data to foreacasting
forecast = (data_raw['fecha'] >= '2019-03')&(data_raw['fecha'] <= '2021-12')


end = dev_sample.sum() + forecast.sum() - 1

init = dev_sample.sum()




data_X = data_raw.loc[dev_sample, target_exog_vars[target]]

data_y = data_raw.loc[dev_sample, [target]]
data_y[target] = data_y[target].apply(lambda x: math.log(x, math.e))

# data = pd.concat([data_y, data_X], axis=1)
model = smx.SARIMAX(data_y, exog=data_X, order=target_arima_params[target]['order'],
                    seasonal_order=target_arima_params[target]['seasonal_order'],
                    enforce_stationarity=False, enforce_invertibility=False)

model_fit = model.fit()

summary = model_fit.summary()
print(summary)


data_forecast_X = data_raw.loc[forecast, target_exog_vars[target]]
data_adverse_forecast_X = data_raw.loc[forecast, target_adv_exog_vars[target]].rename(columns=mapper)


preds = model_fit.get_prediction(start=0, end=end, exog=data_forecast_X)
adverse_preds = model_fit.get_prediction(start=0, end=end, exog=data_adverse_forecast_X)


var_vect = preds.se_mean[init:]
adverse_var_vect = adverse_preds.se_mean[init:]

base_var = np.var(var_vect)
adverse_var = np.var(adverse_var_vect)
var1 = math.exp(base_var/2)
var2 = math.exp(adverse_var/2)

dev_data = preds.predicted_mean[13:init].apply(math.exp)

data_y[target] = data_y[target].apply(math.exp)
pred = preds.predicted_mean[init:].apply(lambda x: math.exp(x)*var1)
adverse_pred = adverse_preds.predicted_mean[init:].apply(lambda x: math.exp(x)*var2)

data_y0 = data_y.reset_index()
plt.figure(figsize=(15, 5))
plt.plot(dev_data, 'r')
plt.plot(pred, 'g')
plt.plot(adverse_pred, 'c')
plt.plot(data_y0[target], 'b')
plt.title('Predicción {}'.format(target))
plt.show()

print('finish!')