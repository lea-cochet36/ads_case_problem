import pandas as pd
from sktime.forecasting.arima import AutoARIMA
import numpy as np

def preprocess_data(data) :
    data = data.rename(columns={"Unnamed: 0": "date"})
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

def predict_model(model, horizon) : 
    y_pred = model.predict(np.arange(1,horizon))
    return y_pred

def postprocess_data(y_pred):
    df = pd.DataFrame()
    df["date"] = y_pred.index.strftime('%m.%y')
    df['date'] = df["date"].apply(lambda x: "01." + x)
    df.index = list(df["date"])
    df["y"] = list(y_pred)
    return df[["y"]]
