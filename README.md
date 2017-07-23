# Forecasting Wizard

**What's in the box?**

Three Python scripts containing different time series forecasting methods. They take as input set of time series in a tabular format (EXCEL) and output forecast time series.

Current available methods are:
* [Seasonal ARIMA](http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html#statsmodels.tsa.statespace.sarimax.SARIMAX)
* [Facebook Prophet](https://facebookincubator.github.io/prophet/)
* Custom method including a seasonal decomposition and a trend forecast using a [gaussian process](http://scikit-learn.org/stable/modules/gaussian_process.html)

**Limitations**

As these methods work by analyzing the macro trend and the seasonal patterns, there should be enough cycles included in the data. A minimum of 3 years of monthly data is required for the Seasonal ARIMA model.

**Installation Tips**

1. Running the code presumes that you have a distribution of Python (>=3.5) installed on your machine with a certain numbers of libraries.
2. The library containing the seasonal Arima can easily be installed with the command: pip install https://github.com/statsmodels/statsmodels/archive/master.zip
3. The bin folder contains the binary source code required to run the models along with the compiler and instructions (For Mac). For Mac users, here is a [tip](https://github.com/christophsax/seasonal/wiki/Compiling-X-13ARIMA-SEATS-from-Source-for-OS-X) to successfully compile the executable.

**User Guide**

This repository contains a single forecasting scripts. It can be run from within an IDE (ex: Pycharm, Spyder etc.) or from a command line.\
_Note:_
* _All input files must be placed in the folder /data/hist_data/ and all output files will be saved in the folder /data/fcst_data/._
* _The spreadsheet containing the input data should be named 'Sheet1' by default (if excel format)._

**Run from command line**

The script takes as an input an excel file containing the time series to be forecasted and outputs an excel file containing the forecasted time series.

**Script name**: Monthly_Forecasting.py

    $ python Monthly_Forecasting.py fcst_method start_date end_date input_file_name output_file_name\
    $ python Monthly_Forecasting.py sarima 2017-03-01 2017-12-01 input_file_name.xlsx output_file_name.xlsx

Forecasting methods can be one of the following: 'sarima', 'sarimax', 'gaussian', 'facebook'

_Note: It is possible to use the Seasonal ARIMA with exogenous regressors (method 'sarimax'). In that case the name of the exogenous file must be modified in the script._
