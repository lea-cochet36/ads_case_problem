import pandas as pd
import numpy as np
from sktime.forecasting.arima import AutoARIMA
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def preprocess_data(data) :
    data = data.rename(columns={"Unnamed: 0": "date"})
    # Assuming the date column is named 'date' and the y-values are in a column named 'y'
    # Convert date strings from 'DD.MM.YY' to datetime objects
    data['date'] = pd.to_datetime(data['date'], format='%d.%m.%y')
    data = data[data.notna()]
    data.set_index('date', inplace=True)
    data.index = data.index.to_period('M')
    data = data.dropna()
    return data

def train_model_arima(data, stationary, test, sp) : 
    model_arima = AutoARIMA(stationary=stationary,test=test,sp = sp, trace=True,suppress_warnings=True)
    print(len(data))
    fitted_model_arima = model_arima.fit(data["y"]) 
    return fitted_model_arima

def predict_model(model, data) : 
    y_pred = model.predict(fh=np.arange(1, len(data) + 1))
    return y_pred

def postprocess_data(y_pred):
    data_pred = pd.DataFrame()
    data_pred["y"] = y_pred
    data_pred.index.name = "date"
    return data_pred

def evaluate_performance(data_true,data_pred) :
    y_true = data_true["y"]
    y_pred = data_pred["y"]
    mse = round(mean_squared_error(y_true,y_pred),3)
    mae = round(mean_absolute_error(y_true,y_pred),3)
    rmse = round(mean_squared_error(y_true, y_pred, square_root=True),3)
    mape = round(mean_absolute_percentage_error(y_true,y_pred),3)
    dict_metrics = {"mse" : mse, "mae" : mae, "rmse" : rmse, "mape" : mape}
    return dict_metrics

# def plot_results(data_true, data_pred) :
#     plt.plot(y_pred.index.to_timestamp(how='start'), y_pred, marker='o', linestyle='-',color="red",label='Pred')
#     plt.plot(y_true.index.to_timestamp(how='start'), y_true, marker='o', linestyle='-',color="blue",label='True')
#     plt.legend(loc='best')
#     plt.title('True VS Pred')
#     plt.show(block=False)