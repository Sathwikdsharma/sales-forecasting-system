import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

df = pd.read_excel("data/Forecasting Case- Study.xlsx")

df['Date'] = pd.to_datetime(df['Date'])

state_df = df[df['State'] == 'Alabama']

state_df = state_df.sort_values('Date')

sales = state_df['Total']

split_index = int(len(sales) * 0.8)

train = sales[:split_index]
test = sales[split_index:]


model = ARIMA(train, order=(5,1,0))


model_fit = model.fit()


forecast = model_fit.forecast(steps=len(test))


mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))

print("\nARIMA MODEL PERFORMANCE")
print(f"MAE  : {mae}")
print(f"RMSE : {rmse}")