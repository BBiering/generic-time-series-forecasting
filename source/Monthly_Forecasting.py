from sklearn.gaussian_process import GaussianProcess
from tools.forecasting_tools import *
from pandas.tseries.offsets import *
from fbprophet import Prophet

# path parameters
data_path = '/data/hist_data/'
fcst_path = '/data/fcst_Data/'
file_fcst = 'traffic_fcst_sample.xlsx'
file_raw = 'traffic_data_sample.xlsx'
sheet_raw = 'raw_data_monthly'

# forecasting method
sarimax_flag = True
gaussian_flag = False
facebook_flag = False

# seasonality and time frame
seasonality_idx = (12,)  # yearly seasonality assumed
fcst_window = 6  # number of months to forecast

# verbose flag
verbose = True

# import time series as data frame
root_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
x13aspath = root_path + '/bin'
writer = pd.ExcelWriter(root_path + fcst_path + file_fcst,
                        engine='xlsxwriter')
data_raw = import_mltpl_timeserie(root_path + data_path + file_raw,
                                  sheet_name=sheet_raw)
ts_org = data_raw.index.min()
te_org = data_raw.index.max()
res_df = pd.DataFrame()
data_df = pd.DataFrame()

for var in data_raw:
    # linearly interpolate missing values in data set
    data_temp = data_raw[var]
    data_temp.replace(to_replace=0.0, value=np.nan, inplace=True)
    if data_temp.isnull().values.any():
        data_temp.interpolate(method='linear', inplace=True)
        data_temp.fillna(method='bfill', inplace=True)

    # SARIMAX model
    if sarimax_flag is True:
        try:
            mdl_order = select_model_order(data_temp,
                                           x13aspath,
                                           'MS')
            ts_fcst = sarimax_model(data_temp,
                                    seasonality_idx,
                                    mdl_order,
                                    fcst_window,
                                    ts_org,
                                    te_org,
                                    verbose)
            data_df[var] = ts_fcst['Forecast'].tail(fcst_window)
            print("Forecasting {0} with SARIMAX succeeded".format(var))

        except Exception:
            print("Forecasting with SARIMAX failed")
            pass

    # Gaussian Process model
    elif gaussian_flag is True:
        composed_df = pd.DataFrame()
        res_df = pd.DataFrame()
        res_test = sm.seasonal_decompose(data_temp.dropna(), two_sided=False)
        composed_df['trend'] = res_test.trend.dropna()
        composed_df['seasonal'] = res_test.seasonal.dropna()
        composed_df['residual'] = res_test.resid.dropna()
        resid_mean = composed_df['residual'].mean()
        date_rng = pd.date_range(composed_df.index[len(composed_df)-1] + DateOffset(months=1),
                                 periods=fcst_window,
                                 freq='MS')
        res_df['Date'] = pd.to_datetime(date_rng, errors='coerce')
        res_df = res_df.sort_values(by='Date')
        res_df = res_df.set_index('Date')
        res_df['Residual'] = resid_mean
        seas_rng = list()
        for i in range(fcst_window):
            seas_rng.append(res_df.index[i] + DateOffset(months=-(fcst_window + 1)))

        seas_data = composed_df.loc[composed_df.index.isin(seas_rng)]
        res_df['Seasonal'] = seas_data['seasonal'].values

        # Prediction based on a Gaussian Process
        X = (composed_df.index - composed_df.index[0]).days.reshape(-1, 1)
        Y = composed_df['trend'].values
        X_pred = (res_df.index - composed_df.index[0]).days.reshape(-1, 1)
        gpr = GaussianProcess(corr='cubic', regr='linear', theta0=1e-2, thetaL=1e-4, thetaU=1e-1,
                              random_start=100)
        gpr.fit(X, Y)
        y_gpr = gpr.predict(X_pred)
        res_df['Trend'] = y_gpr
        res_df['Total'] = res_df.sum(axis=1)
        print("Forecasting {:s} with seasonal decomposition and Gaussian process".format(var))
        data_df[var] = res_df['Total']

    # Facebook Prophet model
    elif facebook_flag is True:
        df_temp = pd.DataFrame()
        df_temp['ds'] = data_temp.index
        df_temp['y'] = data_temp.values
        m = Prophet()
        m.fit(df_temp)
        future = m.make_future_dataframe(periods=fcst_window, freq='M')
        forecast = m.predict(future)
        data_df[var] = forecast['yhat'].tail(fcst_window).values
        data_df['Date'] = pd.date_range(start=ts_org + DateOffset(months=1),
                                        periods=fcst_window,
                                        freq='MS')
        data_df = data_df.set_index('Date')
    else:
        break

# save forecast to file
data_df.to_excel(writer, 'fcst_data')
writer.save()
