from pmdarima.arima import auto_arima
import math
import numpy as np
import pandas as pd
import pmdarima as pm
from sklearn import preprocessing as prp
import matplotlib.pyplot as plt

from dateutil.parser import parse


targets = ['Reserva_Consumo', 'Reserva_Comercial', 'Reserva_Hipotecario',
           'Reserva_Comercial_Empresa', 'Reserva_Comercial_Negocio', 'Reserva_Consumo_TDC']
target = targets[4]

# 'DOLAR_TC','CETES_91','IPC','TIIE_91','T_DESEMPLEO','LIBOR_6','T-BILL_12','VIX','S_P500','INFLACION_ANUAL',

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

init_exog_vars = ['DOLAR_TC', 'CETES_91', 'IPC',
                  'TIIE_91', 'T_DESEMPLEO', 'LIBOR_6',
                  'T-BILL_12', 'VIX', 'S_P500', 'INFLACION_ANUAL']

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
    'Reserva_Consumo': ['T-BILL_12'],
    'Reserva_Comercial': init_exog_vars,  # ['T_DESEMPLEO', 'VIX',],
    'Reserva_Hipotecario': ['DOLAR_TC','CETES_91','TIIE_91',
                            'T_DESEMPLEO','LIBOR_6',],
    'Reserva_Comercial_Empresa': ['T_DESEMPLEO','VIX',],
    'Reserva_Comercial_Negocio': ['DOLAR_TC','CETES_91','T_DESEMPLEO',],
    'Reserva_Consumo_TDC': ['TIIE_91',]
}

target_adv_exog_vars = {
    'Reserva_Consumo': ['{}_ADV'.format(var) for var in target_exog_vars['Reserva_Consumo']],
    'Reserva_Comercial': ['{}_ADV'.format(var) for var in target_exog_vars['Reserva_Comercial']],
    'Reserva_Hipotecario': ['{}_ADV'.format(var) for var in target_exog_vars['Reserva_Hipotecario']],
    'Reserva_Comercial_Empresa': ['{}_ADV'.format(var) for var in target_exog_vars['Reserva_Comercial_Empresa']],
    'Reserva_Comercial_Negocio': ['{}_ADV'.format(var) for var in target_exog_vars['Reserva_Comercial_Negocio']],
    'Reserva_Consumo_TDC': ['{}_ADV'.format(var) for var in target_exog_vars['Reserva_Consumo_TDC']]
}

target_fecha_inicial = {
    'Reserva_Consumo': '2014-07',
    'Reserva_Comercial': '2014-01',
    'Reserva_Hipotecario': '2015-07',  # '2016-01'
    'Reserva_Comercial_Empresa': '2014-01',  # 2015-01
    'Reserva_Comercial_Negocio': '2014-01',  # 2015-01
    'Reserva_Consumo_TDC': '2015-01',
}

target_fecha_final = {
    'Reserva_Consumo': '2018-02',
    'Reserva_Comercial': '2018-02',
    'Reserva_Hipotecario': '2018-02',
    'Reserva_Comercial_Empresa': '2018-02',
    'Reserva_Comercial_Negocio': '2018-02',
    'Reserva_Consumo_TDC': '2019-02',
}


target_arima_params = {
    'Reserva_Consumo': {
        'order': (1, 0, 1),
        'seasonal_order': (0, 0, 0, 0)
    },
    'Reserva_Comercial': {
        'order': (2, 1, 1),
        'seasonal_order': (1, 0, 0, 12)
    },
    'Reserva_Hipotecario': {
        'order': (1, 0, 1),
        'seasonal_order': (0, 0, 0, 0)
    },
    'Reserva_Comercial_Empresa': {
        'order': (1, 0, 0),
        'seasonal_order': (0, 0, 0, 0)
    },
    'Reserva_Comercial_Negocio': {
        'order': (1, 0, 1),
        'seasonal_order': (0, 0, 0, 0)
    },
    'Reserva_Consumo_TDC': {
        'order': (1, 0, 1),
        'seasonal_order': (1, 0, 0, 12)
    },
}

data_raw = pd.read_csv("input/Base_Trabajo.csv", dayfirst=True)


# data_raw['IPC'] = data_raw['IPC'].apply(lambda x: x/10000)
# data_raw['IPC_ADV'] = data_raw['IPC_ADV'].apply(lambda x: x/10000)
# data_raw['S_P500'] = data_raw['S_P500'].apply(lambda x: x/1000)
# data_raw['S_P500_ADV'] = data_raw['S_P500_ADV'].apply(lambda x: x/1000)

data_raw['fecha'] = data_raw['fecha'].apply(lambda x: parse(x, dayfirst=True))

dev_sample = (data_raw['fecha'] >= target_fecha_inicial[target])&(data_raw['fecha'] <= target_fecha_final[target])
# select data to foreacasting
forecast = (data_raw['fecha'] > target_fecha_final[target])&(data_raw['fecha'] <= '2021-12')

scaler = prp.StandardScaler()
# scaler_adv = prp.StandardScaler()

data_raw.loc[dev_sample, exog_vars] = scaler.fit_transform(data_raw.loc[dev_sample, exog_vars].values)
data_raw.loc[dev_sample, exog_adv_vars] = scaler.fit_transform(data_raw.loc[dev_sample, exog_adv_vars].values)

data_raw.loc[forecast, exog_vars] = scaler.transform(data_raw.loc[forecast, exog_vars].values)
data_raw.loc[forecast, exog_adv_vars] = scaler.transform(data_raw.loc[forecast, exog_adv_vars].values)


data_X = data_raw.loc[dev_sample, target_exog_vars[target]]

data_y = data_raw.loc[dev_sample, [target]]
data_y_log = data_y[target].apply(lambda x: math.log(x, math.e))



# SARIMAX Model
sxmodel = pm.auto_arima(data_y_log,
                        start_p=1, start_q=1, test='adf',
                        max_p=3, max_q=3, m=12, start_P=0,
                        seasonal=True, d=None, D=1,
                        trace=True, error_action='ignore',
                        suppress_warnings=True, stepwise=True)
print(sxmodel.summary())

# Forecast
n_periods = 34
fc, confint = sxmodel.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(data_y[target].index[-1]+1, data_y[target].index[-1]+1+n_periods)

# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)
fc_series = fc_series.apply(lambda x: math.exp(x))
lower_series = pd.Series(confint[:, 0], index=index_of_fc).apply(lambda x: math.exp(x))
upper_series = pd.Series(confint[:, 1], index=index_of_fc).apply(lambda x: math.exp(x))

# Plot
plt.plot(data_y[target])
plt.plot(fc_series, color='darkgreen')
# plt.fill_between(lower_series.index,
#                  lower_series,
#                  upper_series,
#                  color='k', alpha=.15)

plt.title("Predicción de {}".format(target))
plt.show()


output = pd.concat([data_raw['fecha'], data_y,fc_series,lower_series,upper_series], axis=1)
output.to_csv('output/{}_output_fc_after_{}.csv'.format(target,target_fecha_final[target]))


print('finish!')