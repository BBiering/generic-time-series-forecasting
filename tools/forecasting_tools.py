import pandas as pd
from pandas.tseries.offsets import *
import numpy as np
import teradata
import sklearn
import statsmodels.tsa.x13 as smx13
import statsmodels.tsa.statespace.sarimax as smsar
import statsmodels.tsa.seasonal as sm
import statsmodels.tsa.stattools as smst
import os
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt

def mape_score(y_true, y_pred, offset):

    y_t = y_true[offset:]
    y_p = y_pred[offset:]
    return np.mean(np.abs((y_t.values - y_p.values) / y_t.values)) * 100


def mase_score(y_true, y_pred, seasonality_idx, offset):

    y_t = y_true[offset:]
    y_p = y_pred[offset:]
    dp_diff_num = np.absolute(y_p.values - y_t.values)
    df_num = dp_diff_num.sum()
    dp_diff_denom = np.absolute(y_t[seasonality_idx[0]:].values - y_t[:y_t.shape[0]-seasonality_idx[0]].values)
    df_denom = dp_diff_denom.sum()
    coeff_denom = float(y_t.shape[0]-seasonality_idx[0])/float(y_t.shape[0])
    return coeff_denom*df_num/df_denom


def rmse_score(y_true, y_pred, offset):

    y_t = y_true[offset:].copy()
    y_p = y_pred[offset:].copy()
    return sklearn.metrics.mean_squared_error(y_t, y_p)


def test_stationarity(timeseries):

    # print 'Results of Dickey-Fuller Test:'
    dftest = smst.adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    # print dfoutput


def timeserie_decomposition(timeseries, path=None):
    res = sm.seasonal_decompose(x=timeseries, two_sided=False)
    df = pd.concat([res.trend, res.seasonal, res.resid], axis=1)
    df.columns = ['trend', 'seasonal', 'residuals']
    if path is not None:
        df.to_csv(path)
    # Uncomment the following if you want plot the decomposition
    # res.plot()
    # plt.show()

def import_mltpl_timeserie(path, sheet_name=None):

    filename, file_extension = os.path.splitext(path)
    raw_data = pd.DataFrame()
    if file_extension == '.csv':
        raw_data = pd.read_csv(path)
    elif file_extension == '.xlsx':
        raw_data = pd.read_excel(io=path, sheetname=sheet_name)
    else:
        print("Not supported file format")
    te = raw_data['Date'].min()
    ts = raw_data['Date'].max()
    raw_data.index = pd.date_range(start=te, end=ts, freq='MS')
    raw_data.drop('Date', axis=1, inplace=True)
    ts = raw_data.astype(float)
    return ts


def select_model_order(timeseries, x13aspath, freq):
    mdl_order = smx13.x13_arima_select_order(timeseries, freq=freq, x12path=x13aspath)
    return mdl_order


def sarimax_model(timeseries, seasonality_idx, mdl_order, fcst_window, ts_start, ts_end, verbose, exog=None):

    if exog is None:

        try:
            mod = smsar.SARIMAX(endog=timeseries,
                                trend='n',
                                order=mdl_order.order,
                                seasonal_order=mdl_order.sorder+seasonality_idx)

            sarimax_mdl = mod.fit(disp=False)
            if verbose is True:
                print("Default Params - Endogenous Mode")

        except ValueError:
            mod = smsar.SARIMAX(endog=timeseries,
                                trend='n',
                                order=mdl_order.order,
                                seasonal_order=(0, 1, 0, seasonality_idx[0]))
            sarimax_mdl = mod.fit(disp=False)
            if verbose is True:
                print("Custom Params - Endogenous Mode")

        print("SARIMAX Endogenous Model, regular order: ({0},{1},{2}), "
              "seasonal order: ({3},{4},{5}), "
              "seasonality level: {6}".format(mdl_order.order[0],
                                              mdl_order.order[1],
                                              mdl_order.order[2],
                                              mdl_order.sorder[0],
                                              mdl_order.sorder[1],
                                              mdl_order.sorder[2],
                                              seasonality_idx[0]))

        sarimax_rslt = sarimax_mdl.predict(alpha=0.05,
                                           start=0,
                                           end=(len(timeseries)-1)+fcst_window)
        sarimax_rslt[12] = np.mean([sarimax_rslt[12-1], sarimax_rslt[12+1]])  # PATCH for buggy value
        sarimax_rslt_info = sarimax_mdl.get_prediction(end=(len(timeseries)-1)+fcst_window)
        sarimax_ci = sarimax_rslt_info.conf_int(alpha=0.05)
        sarimax_ci.columns = ['lower', 'upper']
        sarimax_ci.lower[12] = np.mean([sarimax_ci.lower[12 - 1], sarimax_ci.lower[12 + 1]])
        sarimax_ci.upper[12] = np.mean([sarimax_ci.upper[12 - 1], sarimax_ci.upper[12 + 1]])  # End of PATCH
        sarimax_fcst = pd.concat([timeseries, sarimax_rslt, sarimax_ci], axis=1)
        sarimax_fcst.columns = ['Actual', 'Forecast', 'CI Lower Bound', 'CI Upper Bound']
        # display the confidence intervals spread
        # ci_spread = ((float(sarimax_fcst['CI Upper Bound'].tail(1)) - float(sarimax_fcst['CI Lower Bound'].tail(1))) /
        #              float(sarimax_fcst['Forecast'].tail(1)))*100.0
        # print("CI spread = {0}".format(ci_spread))
        # display the performance scores
        y_pred = sarimax_fcst['Forecast']
        y_pred = y_pred[ts_start:ts_end]
        sarimax_mase = mase_score(timeseries, y_pred, seasonality_idx, 1)  # remove first buggy value
        sarimax_mape = mape_score(timeseries, y_pred, 1)  # remove first buggy value
        print("MASE Score = {0:.2f}, MAPE Score = {1:.2f}".format(sarimax_mase, sarimax_mape))

    elif exog is not None:
        # Shape exog variable
        exog = exog[['var1', 'var2', 'var13', 'var4']].copy()
        ts_start_exog = pd.to_datetime([ts_start])
        ts_end_exog = pd.to_datetime([ts_end])
        ts_exog_past = exog[ts_start_exog[0]:ts_end_exog[0]]
        if verbose is True:
            print(ts_exog_past.shape)

        try:
            mod = smsar.SARIMAX(endog=timeseries,
                                exog=ts_exog_past,
                                trend='n',
                                order=mdl_order.order,
                                seasonal_order=mdl_order.sorder + seasonality_idx)

            sarimax_mdl = mod.fit(disp=False)
            if verbose is True:
                print("Default Params - Exogenous Mode")

        except ValueError:
            mod = smsar.SARIMAX(endog=timeseries,
                                exog=ts_exog_past,
                                trend='n',
                                order=mdl_order.order,
                                seasonal_order=(0, 1, 0, seasonality_idx[0]))

            sarimax_mdl = mod.fit(disp=False)
            if verbose is True:
                print("Custom Params - Exogenous Mode")

        print("SARIMAX Exogenous Model, regular order: ({0},{1},{2}), "
              "seasonal order: ({3},{4},{5}), "
              "seasonality level: {6}".format(mdl_order.order[0],
                                              mdl_order.order[1],
                                              mdl_order.order[2],
                                              mdl_order.sorder[0],
                                              mdl_order.sorder[1],
                                              mdl_order.sorder[2],
                                              seasonality_idx[0]))
        # Shape the exogenous time serie
        ts_start_exog = pd.to_datetime([ts_end]) + DateOffset(months=1)
        ts_end_exog = pd.to_datetime([ts_end]) + DateOffset(months=fcst_window)
        ts_exog_future = exog[ts_start_exog[0]:ts_end_exog[0]]
        np_exog = np.array(ts_exog_future)
        if verbose is True:
            print(np_exog.shape)

        # sarimax_rslt = sarimax_mdl.predict(alpha=0.05, start=0, end=(len(timeseries) - 1) + fcst_window, exog=np.reshape(np_exog,(fcst_window,1)))
        sarimax_rslt = sarimax_mdl.predict(alpha=0.05,
                                           start=0,
                                           end=(len(timeseries) - 1) + fcst_window,
                                           exog=np_exog)
        sarimax_rslt[12] = np.mean([sarimax_rslt[12-1], sarimax_rslt[12+1]])  # PATCH for buggy value
        # sarimax_test = sarimax_mdl.forecast(steps=fcst_window, exog=np.reshape(np_exog,(fcst_window,1)))
        sarimax_fcst = pd.concat([timeseries, sarimax_rslt], axis=1)
        sarimax_fcst.columns = ['Actual', 'Forecast']
        # display the performance scores
        y_pred = sarimax_fcst['Forecast']
        y_pred = y_pred[ts_start:ts_end]
        sarimax_mase = mase_score(timeseries, y_pred, seasonality_idx, 1)  # remove first buggy value
        sarimax_mape = mape_score(timeseries, y_pred, 1)  # remove first buggy value
        print("MASE Score = {0:.2f}, MAPE Score = {1:.2f}".format(sarimax_mase, sarimax_mape))

    return sarimax_fcst


def print_forecast_results(timeseries_true, timeseries_pred, col_name, conf_int=None):

    # Display a graph of the observed and forecasted time series along with the confidence intervals if available

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(timeseries_true.index, timeseries_true, 'b', label='Actual Traffic')
    ax.plot(timeseries_pred.index, timeseries_pred, 'g--', label='Forecast Traffic')
    if conf_int is not None:
        ax.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='g', alpha=0.1)
    ax.set(title=col_name, xlabel='Date', ylabel='Visits')
    ax.legend(loc='lower right')
    plt.show()
