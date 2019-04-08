import numpy as np
import pandas as pd
import pyflux as pf

from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime
import matplotlib.pyplot as plt

from dateutil.parser import parse

data_raw = pd.read_csv("input/series.csv", dayfirst=True)  # , index_col=0, dayfirst=True, parse_dates=True)

data_raw['fecha'] = data_raw['fecha'].apply(lambda x: parse(x, dayfirst=True))

# select development sample
dev_sample = (data_raw['fecha'] >= '2014-01')&(data_raw['fecha'] <= '2018-03')
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


def plot_prediction(data_raw, target, exog_vars, formula,
                    BASE, ADVERSE, BASE_MAPPER, ADVERSE_MAPPER,
                    AR=1, MA=0, INTEG=0, past_values=100):
    data_X = data_raw.loc[dev_sample, exog_vars]
    data_y = data_raw.loc[dev_sample, [target]]
    data = pd.concat([data_y, data_X], axis=1)

    model = pf.ARIMAX(data=data, formula=formula,
                      ar=AR, ma=MA, integ=INTEG, family=pf.Normal())
    x = model.fit("PML")
    x.summary()
    model.plot_fit(figsize=(15, 5))

    data_base_forecast_X = data_raw.loc[forecast, BASE].rename(columns=BASE_MAPPER)
    data_adverse_forecast_X = data_raw.loc[forecast, ADVERSE].rename(columns=ADVERSE_MAPPER)
    data_forecast_y = data_raw.loc[forecast, [target]]
    data_forecast_y.loc[:] = 0

    data_base = pd.concat([data_forecast_y, data_base_forecast_X], axis=1)
    data_adverse = pd.concat([data_forecast_y, data_adverse_forecast_X], axis=1)

    model.plot_predict(h=24, oos_data=data_base, past_values=past_values, figsize=(15, 5))
    model.plot_predict(h=24, oos_data=data_adverse, past_values=past_values, figsize=(15, 5))




targets = ['comercial_calif', 'hipotecario_tot', 'consumo_puro_sh']

armas = [[1, 0], [1, 1]]

for target in targets:
    # regressors
    formula = '{}~1+CETES28+CETES364+TIPO_CAMBIO_USD+TASA_DESEMPLEO'.format(target)

    plot_prediction(data_raw, target, exog_vars, formula,
                    BASE, ADVERSE, BASE_MAPPER, ADVERSE_MAPPER,
                    AR=1, MA=0, INTEG=0, past_values=49)


# formula = '{}~1+CETES28+CETES364+TIPO_CAMBIO_USD+TASA_DESEMPLEO'.format('comercial_calif')
# plot_prediction(data_raw, 'comercial_calif', exog_vars, formula,
#                 BASE, ADVERSE, BASE_MAPPER, ADVERSE_MAPPER,
#                 AR=0, MA=0, INTEG=1)
#
# formula = '{}~1+CETES28+CETES364+TIPO_CAMBIO_USD+TASA_DESEMPLEO'.format('comercial_calif')
# plot_prediction(data_raw, 'comercial_calif', exog_vars, formula,
#                 BASE, ADVERSE, BASE_MAPPER, ADVERSE_MAPPER,
#                 AR=0, MA=0, INTEG=1)


print('finish!')
