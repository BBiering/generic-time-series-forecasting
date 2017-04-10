# eCG Forecasting Project

**What's in the box?**

Python scripts containing different time series forecasting methods. They take as input set of time series in a tabular format (EXCEL) and output forecast time series.

Current available methods are:
* [Seasonal ARIMA](http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html#statsmodels.tsa.statespace.sarimax.SARIMAX)
* [Gaussian Process](http://scikit-learn.org/stable/modules/gaussian_process.html)
* [Facebook Prophet](https://facebookincubator.github.io/prophet/)

**Who should use these tools?**

Quantitative analysts with rudimentary understanding of time series and regression models. 

**Requirements/Limitations**

As these methods work by analyzing the macro trend and the seasonal patterns, there should be enough cyles included in the data. A minimum of 3 years of monthly data is required for the Seasonal ARIMA model.

**Technicalities**

1. Running the code presumes that you have a distribution of Python (>=3.6) installed on your machine with a certain numbers of libraries.
2. The library containing the time serie models can easily be installed with the command: pip install https://github.com/statsmodels/statsmodels/archive/master.zip
3. The bin folder contains the binary source code required to run the models along with the compiler and instructions (For Mac). For Mac users, here is a [tip](https://github.com/christophsax/seasonal/wiki/Compiling-X-13ARIMA-SEATS-from-Source-for-OS-X) to successfully compile the executable. 
