import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

df = pd.read_excel("data/Forecasting Case- Study.xlsx")

df['Date'] = pd.to_datetime(df['Date'])

df.rename(columns={
    'Date': 'ds',
    'Total': 'y'
}, inplace=True)

state_df = df[df['State'] == 'Alabama']

state_df = state_df[['ds', 'y']]

split_index = int(len(state_df) * 0.8)

train = state_df.iloc[:split_index]
test = state_df.iloc[split_index:]

model = Prophet()

model.fit(train)

future = model.make_future_dataframe(periods=len(test))

forecast = model.predict(future)

predictions = forecast['yhat'][-len(test):].values

actual = test['y'].values

mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))

print("\nPROPHET MODEL PERFORMANCE")
print(f"MAE  : {mae}")
print(f"RMSE : {rmse}")