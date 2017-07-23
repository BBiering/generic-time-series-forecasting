import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import pandas as pd
    from pandas.tseries.offsets import *
    from sklearn.gaussian_process import GaussianProcess
import datetime as dt
from datetime import date
import os
import numpy as np
import teradata
import statsmodels.tsa.x13 as smx13
import statsmodels.tsa.statespace.sarimax as smsar
import statsmodels.tsa.seasonal as sm
from fbprophet import Prophet
import getpass as gp
import pickle


def update_traffic_actuals(df, channel, root_path, data_path, site_name, ga_profiles, site_id):
    input_file_name = '/input_total_traffic.xlsx'
    if channel == 'non-paid':
        input_file_name = '/input_non_paid_traffic.xlsx'

    # check if we need to query the traffic data from Teradata
    next_month_start = df.index.max() + DateOffset(months=1)
    next_month = next_month_start.month
    this_month = dt.date.today().month

    # query the latest traffic data
    if next_month != this_month:
        last_month_end = dt.date.today().replace(day=1) - dt.timedelta(days=1)
        ts_start = str(next_month_start.date())
        ts_end = str(last_month_end)
        all_df = get_mnthly_traffic(root_path,
                                    channel,
                                    ga_profiles,
                                    ts_start,
                                    ts_end,
                                    site_id)
        df = df.append(all_df)
        # save the results in the original excel file
        writer_organic = pd.ExcelWriter(root_path +
                                        data_path +
                                        site_name +
                                        input_file_name)
        df.to_excel(writer_organic, sheet_name='Monthly Traffic - Actuals')
        writer_organic.save()
        writer_organic.close()
        print('Info: Traffic input file successfully updated')
    else:
        print('Info: Traffic input file already up-to-date')

    return df


def get_mnthly_traffic(root_path, channel, ga_profiles, ts_start, ts_end, site_id):
    uda_sess = teradata.UdaExec()
    df = pd.DataFrame()
    f = open(root_path + '/data/sql_queries/total_traffic.pckl', 'rb')
    sql_query = pickle.load(f)
    if channel == 'non-paid':
        f = open(root_path + '/data/sql_queries/non_paid_traffic.pckl', 'rb')
        sql_query = pickle.load(f)

    f.close()
    td_query = sql_query.format(ga_profiles, site_id, ts_start, ts_end)
    username_td = input('Enter your Teradata username:')
    password_td = gp.getpass("Enter your Teradata password:")
    with uda_sess.connect("${dataSourceName}", username=username_td, password=password_td) as session:
        for row in session.execute(td_query):
            df = df.append({
                "Date_DT": row[0],
                "Platform": row[1],
                "Device": row[2],
                "Sessions": row[3]
            }, ignore_index=True)

    # aggregate the data per month
    df['Date'] = pd.to_datetime(df['Date_DT'], errors='coerce')
    df.drop('Date_DT', axis=1, inplace=True)
    df = df.sort_values(by='Date')
    df = df.set_index('Date')
    df = df.groupby([pd.TimeGrouper(freq='MS'), 'Platform', 'Device']).sum()
    df.reset_index(inplace=True)
    df = pd.pivot_table(df, values='Sessions', index='Date', columns=['Platform', 'Device'], aggfunc=np.sum)
    df.columns = [' - '.join(col).strip() for col in df.columns.values]
    df.index.name = 'Date'
    return df


def device_split(root_path, ga_profiles, ts_start, ts_end, site_id):
    sql_query_str = {'Adwords': 'adwords_traffic.pckl',
                     'Criteo': 'criteo_traffic.pckl',
                     'Bing': 'bing_traffic.pckl'}
    username_td = input('Enter your Teradata username:')
    password_td = gp.getpass("Enter your Teradata password:")
    res_df = pd.DataFrame()
    for key in sql_query_str:
        uda_sess = teradata.UdaExec()
        df = pd.DataFrame()
        f = open(root_path+'/data/sql_queries/'+sql_query_str[key], 'rb')
        sql_query = pickle.load(f)
        f.close()
        td_query = sql_query.format(ga_profiles, site_id, ts_start, ts_end)
        with uda_sess.connect("${dataSourceName}", username=username_td, password=password_td) as session:
            for row in session.execute(td_query):
                df = df.append({
                    "Platform": row[0],
                    "Device": row[1],
                    "Sessions": row[2]
                    }, ignore_index=True)
        for i in range(len(df)):
            df.loc[i, 'Share'] = round(df.loc[i, 'Sessions'] /
                                       df.Sessions.sum(axis=0),
                                       3)
            # df.loc[i, 'Share'] = round(df.loc[i, 'Sessions'] /
            #                            df[df.Platform == df.loc[i, 'Platform']].Sessions.sum(axis=0),
            #                            2)
        df['Channel'] = key
        res_df = res_df.append(df)
    return res_df


def create_result_df(fcst_window, ts_pred_start):
    columns_labels = ['Android - mobile',
                      'Android - tablet',
                      'Web - desktop',
                      'Web - mobile',
                      'Web - tablet',
                      'iOS - mobile',
                      'iOS - tablet']
    df = pd.DataFrame(np.zeros(shape=(fcst_window, len(columns_labels))),
                      columns=columns_labels)
    date_rng = pd.date_range(ts_pred_start,
                             periods=fcst_window,
                             freq='MS')
    df['Date'] = pd.to_datetime(date_rng, errors='coerce')
    df = df.sort_values(by='Date')
    df = df.set_index('Date')
    df.index.name = 'Date'
    return df


def fill_result_df(ts, dvic_split, fcst_window, ts_pred_start):
    rslt_df = create_result_df(fcst_window, ts_pred_start)
    for j in range(len(ts)):
        for i in range(len(dvic_split)):
            column_name = str(dvic_split.loc[i, 'Platform']) + ' - ' + str(dvic_split.loc[i, 'Device'])
            rslt_df[column_name][j] = float(dvic_split.loc[i, 'Share']) * float(ts[j])
    return rslt_df


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


def timeserie_decomposition(timeseries, path=None):
    res = sm.seasonal_decompose(x=timeseries, two_sided=False)
    df = pd.concat([res.trend, res.seasonal, res.resid], axis=1)
    df.columns = ['trend', 'seasonal', 'residuals']
    if path is not None:
        df.to_csv(path)


def import_mltpl_timeserie(path, sheet_name=None):

    filename, file_extension = os.path.splitext(path)
    raw_data = pd.DataFrame()
    if file_extension == '.csv':
        raw_data = pd.read_csv(path)
    elif file_extension == '.xlsx':
        raw_data = pd.read_excel(io=path, sheetname=sheet_name)
    else:
        print("Not supported file format")
    ts = raw_data['Date'].min()
    te = raw_data['Date'].max()
    raw_data.index = pd.date_range(start=ts, end=te, freq='MS')
    raw_data.drop('Date', axis=1, inplace=True)
    raw_data.index.name = 'Date'
    df_ts = raw_data.astype('float64')
    return df_ts


def select_model_order(timeseries, x13aspath, freq):
    mdl_order = smx13.x13_arima_select_order(timeseries, freq=freq, x12path=x13aspath)
    return mdl_order


def sarimax_model(timeseries, seasonality_idx, mdl_order, fcst_window, ts_start, ts_end, verbose, exog=None):

    sarimax_fcst = pd.DataFrame()

    if exog is None:

        try:
            mod = smsar.SARIMAX(endog=timeseries,
                                trend='n',
                                order=mdl_order.order,
                                seasonal_order=mdl_order.sorder+seasonality_idx)

            sarimax_mdl = mod.fit(disp=False)
            if verbose is True:
                print("SARIMA Info: Default Params - Endogenous Mode")

        except ValueError:
            mod = smsar.SARIMAX(endog=timeseries,
                                trend='n',
                                order=mdl_order.order,
                                seasonal_order=(0, 1, 0, seasonality_idx[0]))
            sarimax_mdl = mod.fit(disp=False)
            if verbose is True:
                print("SARIMA Info: Custom Params - Endogenous Mode")

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
        # y_pred = sarimax_fcst.loc[ts_start:ts_end]
        # sarimax_mase = mase_score(timeseries, y_pred.Forecast, seasonality_idx, 1)  # remove first buggy value
        # sarimax_mape = mape_score(timeseries, y_pred.Forecast, 1)  # remove first buggy value
        # print("MASE Score = {0:.2f}, MAPE Score = {1:.2f}".format(sarimax_mase, sarimax_mape))
        # if sarimax_mase < 1 and sarimax_mape < 10:
        #     print("SARIMA Info: Forecasting Accuracy is OK")
        # else:
        #     print("SARIMA Info: Forecasting Accuracy is not OK, check the forecast results")

    elif exog is not None:
        # shape exogenous time serie for past values
        ts_start_exog = pd.to_datetime([ts_start])
        ts_end_exog = pd.to_datetime([ts_end])
        ts_exog_past = exog[ts_start_exog[0]:ts_end_exog[0]]

        try:
            mod = smsar.SARIMAX(endog=timeseries,
                                exog=ts_exog_past,
                                trend='n',
                                order=mdl_order.order,
                                seasonal_order=mdl_order.sorder + seasonality_idx)

            sarimax_mdl = mod.fit(disp=False)
            if verbose is True:
                print("SARIMA Info: Default Params - Exogenous Mode")

        except ValueError:
            mod = smsar.SARIMAX(endog=timeseries,
                                exog=ts_exog_past,
                                trend='n',
                                order=mdl_order.order,
                                seasonal_order=(0, 1, 0, seasonality_idx[0]))

            sarimax_mdl = mod.fit(disp=False)
            if verbose is True:
                print("SARIMA Info: Custom Params - Exogenous Mode")

        # shape exogenous time serie for future values
        ts_start_exog = pd.to_datetime([ts_end]) + DateOffset(months=1)
        ts_end_exog = pd.to_datetime([ts_end]) + DateOffset(months=fcst_window)
        ts_exog_future = exog[ts_start_exog[0]:ts_end_exog[0]]
        np_exog = np.array(ts_exog_future)

        # forecast time serie using exogenous factors
        sarimax_rslt = sarimax_mdl.predict(alpha=0.05,
                                           start=0,
                                           end=(len(timeseries) - 1) + fcst_window,
                                           exog=np_exog)
        sarimax_rslt[12] = np.mean([sarimax_rslt[12-1], sarimax_rslt[12+1]])  # PATCH for buggy value
        sarimax_fcst = pd.concat([timeseries, sarimax_rslt], axis=1)
        sarimax_fcst.columns = ['Actual', 'Forecast']

    return sarimax_fcst


def decompose_model(timeseries, fcst_window):
    composed_df = pd.DataFrame()
    res_df = pd.DataFrame()
    res_test = sm.seasonal_decompose(timeseries.dropna(),
                                     two_sided=False)
    composed_df['trend'] = res_test.trend.dropna()
    composed_df['seasonal'] = res_test.seasonal.dropna()
    composed_df['residual'] = res_test.resid.dropna()

    # create date index for the output data frame
    date_rng = pd.date_range(composed_df.index[len(composed_df) - 1] + DateOffset(months=1),
                             periods=fcst_window,
                             freq='MS')
    res_df['Date'] = pd.to_datetime(date_rng,
                                    errors='coerce')
    res_df = res_df.sort_values(by='Date')
    res_df = res_df.set_index('Date')

    # predict the residual component
    resid_mean = composed_df['residual'].mean()
    res_df['Residual'] = resid_mean

    # predict the seasonal component
    last_year = date_rng[0].year - 1
    last_year_rng = pd.date_range(date(last_year, 1, 1),
                                  periods=12,
                                  freq='MS')
    seas_data = composed_df.loc[composed_df.index.isin(last_year_rng)].seasonal
    seas_val = list()
    for i in range(fcst_window):
        seas_val.append(seas_data[res_df.index[i].month - 1])

    res_df['Seasonal'] = seas_val

    # predict the trend component (Gaussian Process)
    x_fit = (composed_df.index - composed_df.index[0]).days.tolist()
    x_test = (res_df.index - composed_df.index[0]).days.tolist()
    x_fit_np = np.asarray(x_fit).reshape((-1, 1))
    x_test_np = np.asarray(x_test).reshape((-1, 1))
    y_fit = composed_df['trend'].values
    y_fit_np = np.asarray(y_fit).reshape((-1, 1))
    gpr = GaussianProcess(corr='cubic', regr='linear', theta0=1e-2, thetaL=1e-4, thetaU=1e-1,
                          random_start=100)
    gpr.fit(x_fit_np, y_fit_np)
    y_gpr = gpr.predict(x_test_np)
    res_df['Trend'] = y_gpr
    res_df['Total'] = res_df.sum(axis=1)
    res_df.loc[res_df['Total'] < 0, 'Total'] = 0
    return res_df


def facebook_model(timeseries, fcst_window):
    df_temp = pd.DataFrame()
    res_df = pd.DataFrame()
    df_temp['ds'] = timeseries.index
    df_temp['y'] = timeseries.values
    m = Prophet()
    m.fit(df_temp)
    future = m.make_future_dataframe(periods=fcst_window, freq='M')
    forecast = m.predict(future)
    res_df['Total'] = forecast['yhat'].tail(fcst_window).values
    res_df['Date'] = pd.date_range(start=timeseries.index.max() + DateOffset(months=1),
                                   periods=fcst_window,
                                   freq='MS')
    res_df = res_df.sort_values(by='Date')
    res_df = res_df.set_index('Date')
    res_df.index.name = 'Date'
    return res_df
