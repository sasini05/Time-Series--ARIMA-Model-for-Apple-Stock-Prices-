import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Ignore formatting warnings for clean terminal output
warnings.filterwarnings('ignore')

print("--- Starting ARIMA Pre-processing ---")

# 1. Load and Pre-process
df_stock = pd.read_csv('apple_stock.csv')

# Convert 'Date' to datetime and set as index
df_stock['Date'] = pd.to_datetime(df_stock['Date'], format='%d-%m-%y')
df_stock.set_index('Date', inplace=True)

# Sort by date to ensure correct order
df_stock.sort_index(inplace=True)

# Extract the 'Close' price for analysis
df_close = df_stock[['Close']].copy()

# Plot the raw closing price to visualize the overall trend
plt.figure(figsize=(12, 6))
plt.plot(df_close.index, df_close['Close'], color='green')
plt.title('Apple (AAPL) Daily Closing Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.savefig('ARIMA_1_Raw_Data.png', bbox_inches='tight') 
plt.close() 

# 2. Differencing and Stationarity



df_close['Diff_1'] = df_close['Close'].diff()

#  Run ADF test on the RAW, non-stationary data
result_raw = adfuller(df_close['Close'].dropna())
print(f"ADF p-value (Raw Data): {result_raw[1]:.5f}")

#  Apply First-Order Differencing to flatten the trend
df_close['Diff_1'] = df_close['Close'].diff()

# Run ADF test on the DIFFERENCED data
result_diff = adfuller(df_close['Diff_1'].dropna())
print(f"ADF p-value (Differenced Data): {result_diff[1]:.5f}")

#  ACF and PACF Plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plot_acf(df_close['Diff_1'].dropna(), ax=axes[0], lags=40, title='ACF (Differenced)')
plot_pacf(df_close['Diff_1'].dropna(), ax=axes[1], lags=40, title='PACF (Differenced)')
plt.tight_layout()
plt.savefig('ARIMA_2_ACF_PACF.png', bbox_inches='tight') # Saves the image
plt.close()

# 3. Model Fitting and Forecasting

# Split the data (Hold out the last 30 days for testing)
print("--- Fitting ARIMA Model ---")
train = df_close.iloc[:-30]
test = df_close.iloc[-30:]

# Define and Fit the ARIMA Model
# Parameters: (p,d,q) = (1, 1, 1)
model = ARIMA(train['Close'], order=(1, 1, 1))

# Fit the model
fitted_model = model.fit()

# Generate Predictions for the 30-day test period
predictions = fitted_model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

#  Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(test.index, test['Close'], marker='o', label='Actual Stock Price')
plt.plot(test.index, predictions, color='red', marker='x', label='ARIMA Forecast')
plt.legend()
plt.title('ARIMA Forecast vs Actual Apple Stock Price (Last 30 Days)')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.savefig('ARIMA_3_Forecast.png', bbox_inches='tight') # Saves the image
plt.close()

# 4. Calculate Error Metrics
mae = mean_absolute_error(test['Close'], predictions)
rmse = np.sqrt(mean_squared_error(test['Close'], predictions))
mape = np.mean(np.abs((test['Close'].values - predictions.values) / test['Close'].values)) * 100

print("\n--- Final ARIMA Metrics ---")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print("Images successfully saved to your folder!")