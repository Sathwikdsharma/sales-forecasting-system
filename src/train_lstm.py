import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
df = pd.read_excel("data/Forecasting Case- Study.xlsx")
# Convert date column
df['Date'] = pd.to_datetime(df['Date'])
# Filter one state
state_df = df[df['State'] == 'Alabama']

# Sort date
state_df = state_df.sort_values('Date')

# Sales column
data = state_df['Total'].values.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences
X = []
y = []

sequence_length = 7

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

# Train-test split
split_index = int(len(X) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

# Build LSTM model
model = Sequential()

model.add(
    LSTM(
        50,
        activation='relu',
        input_shape=(X_train.shape[1], 1)
    )
)

model.add(Dense(1))

# Compile
model.compile(
    optimizer='adam',
    loss='mse'
)

# Train
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    verbose=1
)

# Predict
predictions = model.predict(X_test)

# Inverse transform
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)

# Evaluation
mae = mean_absolute_error(y_test_actual, predictions)
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))

print("\nLSTM MODEL PERFORMANCE")
print(f"MAE  : {mae}")
print(f"RMSE : {rmse}")