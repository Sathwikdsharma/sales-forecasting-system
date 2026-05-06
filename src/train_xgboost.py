import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import numpy as np
import joblib

df = pd.read_excel("data/Forecasting Case- Study.xlsx")

df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values(['State', 'Date'])

df.rename(columns={'Total': 'Sales'}, inplace=True)

df['lag_1'] = df.groupby('State')['Sales'].shift(1)
df['lag_7'] = df.groupby('State')['Sales'].shift(7)
df['lag_30'] = df.groupby('State')['Sales'].shift(30)

df['rolling_mean_7'] = (
    df.groupby('State')['Sales']
    .transform(lambda x: x.rolling(7).mean())
)

df['rolling_std_7'] = (
    df.groupby('State')['Sales']
    .transform(lambda x: x.rolling(7).std())
)

df['day_of_week'] = df['Date'].dt.dayofweek
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year

df.dropna(inplace=True)

state_encoder = LabelEncoder()
category_encoder = LabelEncoder()

df['State'] = state_encoder.fit_transform(df['State'])
df['Category'] = category_encoder.fit_transform(df['Category'])

X = df.drop(columns=['Sales', 'Date'])
y = df['Sales']

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\nMODEL PERFORMANCE")
print(f"MAE  : {mae}")
print(f"RMSE : {rmse}")

joblib.dump(model, "models/xgboost_model.pkl")

print("\nModel saved successfully!")