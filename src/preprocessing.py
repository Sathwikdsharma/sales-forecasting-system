import pandas as pd
import numpy as np


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


print("FINAL DATASET")
print(df.head())

print("\nFINAL SHAPE")
print(df.shape)

print("\nCOLUMNS")
print(df.columns)