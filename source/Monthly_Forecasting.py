import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from dateutil import relativedelta
from datetime import datetime
from tools.forecasting_tools import *

if __name__ == "__main__":

    # path parameters following the Git project structure
    input_path = '/data/hist_data/'
    output_path = '/data/fcst_data/'

    fcst_method = 'sarima'
    start_date = '2017-07-01'
    end_date = '2018-12-01'
    input_file = 'input_file.xlsx'
    input_file_exog = 'input_file_exog.xlsx'
    output_file = 'output_file.xlsx'
    fcst_window = 18

    # initialize flags and variables
    sarimax_flag = False
    exog_flag = False
    gaussian_flag = False
    facebook_flag = False
    verbose = False
    seasonality_idx = (12,)  # yearly seasonality assumed for seasonal ARIMA

    # retrieve command-line arguments (run script via command line)
    if len(sys.argv) > 1:
        fcst_method = str(sys.argv[1])
        start_date = str(sys.argv[2])
        end_date = str(sys.argv[3])
        input_file = str(sys.argv[4])
        output_file = str(sys.argv[5])  # name of the forecast result file
        fcst_delta = relativedelta.relativedelta(datetime.strptime(end_date, "%Y-%m-%d"),
                                                 datetime.strptime(start_date, "%Y-%m-%d"))
        fcst_window = fcst_delta.years*12 + fcst_delta.months + 1

    # use if multiple sheets in your input file
    input_sheet_name = 'Sheet1'

    # forecasting method
    if fcst_method == 'sarima':
        sarimax_flag = True
        print('Info: Forecasting method is Seasonal ARIMA')
    elif fcst_method == 'sarimax':
        sarimax_flag = True
        exog_flag = True
        print('Info: Forecasting method is Seasonal ARIMA with Exogenous Regressors')
    elif fcst_method == 'gaussian':
        gaussian_flag = True
        print('Info: Forecasting method is Gaussian Process')
    elif fcst_method == 'facebook':
        facebook_flag = True
        print('Info: Forecasting method is Facebook Prophet')
    else:
        print('Error: Forecasting method should be one of sarima, sarimax, gaussian or facebook')

    # import time series as data frame
    root_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    x13aspath = root_path + '/bin'
    writer = pd.ExcelWriter(root_path +
                            output_path +
                            output_file,
                            engine='xlsxwriter')

    # import input data
    data_raw = import_mltpl_timeserie(root_path +
                                      input_path +
                                      input_file,
                                      sheet_name=input_sheet_name)

    # import exogenous factors for SARIMAX method
    if exog_flag is True:
        data_raw_exog = import_mltpl_timeserie(root_path +
                                               input_path +
                                               input_file_exog,
                                               sheet_name=input_sheet_name)
    else:
        data_raw_exog = None

    # initialize data frame variables
    data_df = pd.DataFrame()
    data_temp = pd.DataFrame()
    y_pred = pd.Series()

    # calculate forecast for each data stream
    for var in data_raw:
        # interpolate & backfill missing values
        data_temp = data_raw[var]
        data_temp.replace(to_replace=0.0, value=np.nan, inplace=True)
        if data_temp.isnull().values.any():
            data_temp.interpolate(method='linear', inplace=True)
            data_temp.fillna(method='bfill', inplace=True)

        # Seasonal ARIMA model
        if sarimax_flag is True:
            try:
                mdl_order = select_model_order(data_temp,
                                               x13aspath,
                                               'MS')
                # here you can force the seasonal component to 0
                ts_fcst = sarimax_model(data_temp,
                                        seasonality_idx,
                                        mdl_order,
                                        fcst_window,
                                        data_raw.index.min(),
                                        data_raw.index.max(),
                                        verbose,
                                        data_raw_exog)
                ts_fcst.loc[ts_fcst['Forecast'] < 0, 'Forecast'] = 0
                y_pred = ts_fcst.loc[start_date:end_date].Forecast
                print("{0} forecast with Seasonal ARIMA - Success".format(var))

            except Exception:
                y_pred = pd.Series()
                print("{0} forecast with Seasonal ARIMA - Failure".format(var))
                pass

        # Gaussian Process model
        elif gaussian_flag is True:
            ts_fcst = decompose_model(data_temp,
                                      fcst_window)
            y_pred = ts_fcst.Total
            print("{0} forecast with Gaussian process - Success".format(var))

        # Facebook Prophet model
        elif facebook_flag is True:
            print(data_temp.index)
            ts_fcst = facebook_model(data_temp,
                                     fcst_window)
            y_pred = ts_fcst.Total
            print("{0} forecast with Facebook Prophet - Success".format(var))
        else:
            break

        data_df[var] = y_pred

    data_raw = data_raw.append(data_df)
    data_raw.index.name = 'Date'
    # save to file
    data_raw.to_excel(writer, 'Metric - Forecast')
    writer.save()
