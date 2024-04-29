import pandas as pd
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

def test_stationarity(x, window):
    x_temp = x.copy()
    x_temp.index = x_temp.index.to_timestamp()

    # Determing rolling statistics
    rolmean = x_temp.rolling(window=window, center=False).mean()
    rolstd = x_temp.rolling(window=window, center=False).std()

    # Plot rolling statistics:
    orig = plt.plot(x_temp, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(x_temp, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=[
                         'Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    # Run the ADF test on the time series
    result = adfuller(x_temp)

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    for key, value in result[4].items():
        if result[0] > value:
            print("The data are non-stationary at %s" % (key))
        else:
            print("The data are stationary at %s" % (key))
            break;

    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
        

    
def split_train_test(data, split_ratio = 0.2) :
    
    data_train = data[:int(len(data)*(1-split_ratio))]
    data_test = data[int(len(data)*(1-split_ratio)):]
    return data_train, data_test

def evaluate_performance(data_true,data_pred) :
    y_true = data_true["y"]
    y_pred = data_pred["y"]
    mse = round(mean_squared_error(y_true,y_pred),3)
    mae = round(mean_absolute_error(y_true,y_pred),3)
    rmse = round(mean_squared_error(y_true, y_pred, square_root=True),3)
    mape = round(mean_absolute_percentage_error(y_true,y_pred),3)
    dict_metrics = {"mse" : mse, "mae" : mae, "rmse" : rmse, "mape" : mape}
    return dict_metrics

def postprocess_pred(y_pred):
    data_pred = pd.DataFrame()
    data_pred["y"] = y_pred
    data_pred.index.name = "date"
    return data_pred