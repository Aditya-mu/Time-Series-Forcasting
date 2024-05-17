import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

# Fetch stock data from Yahoo Finance
stock_symbol = 'AAPL'  # You can change this to any stock symbol
start_date = '2020-01-01'
end_date = '2023-01-01'

data = yf.download(stock_symbol, start=start_date, end=end_date)

# Prepare the data
data = data[['Close']].dropna()

# Reset index for plotting
data.reset_index(inplace=True)

# Plot the closing price using Plotly
fig1 = px.line(data, x='Date', y='Close', title=f'{stock_symbol} Stock Closing Prices')
fig1.update_xaxes(title='Date', tickangle=45)
fig1.update_yaxes(title='Price')
fig1.show()

# Split the data into training and test sets
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# Define and fit the ARIMA model
model = ARIMA(train['Close'], order=(5, 1, 0))  # (p, d, q) parameters
model_fit = model.fit()

# Print the model summary
print(model_fit.summary())

# Forecast the future values
forecast = model_fit.forecast(steps=len(test))

# Prepare forecast data for plotting
forecast_df = pd.DataFrame(forecast, index=test.index, columns=['Forecast'])

# Merge forecast with test data
test.set_index('Date', inplace=True)
test = test.join(forecast_df)

# Plot the training and test data along with the forecast using Plotly
fig2 = go.Figure()

# Add training data
fig2.add_trace(go.Scatter(x=train['Date'], y=train['Close'], mode='lines', name='Training'))

# Add test data
fig2.add_trace(go.Scatter(x=test.index, y=test['Close'], mode='lines', name='Actual'))

# Add forecast data
fig2.add_trace(go.Scatter(x=test.index, y=test['Forecast'], mode='lines', name='Forecast'))

fig2.update_layout(title='Train/Test Split and Forecast', xaxis_title='Date', yaxis_title='Price')
fig2.show()

# Calculate MSE and MAE
mse = mean_squared_error(test['Close'], test['Forecast'])
mae = mean_absolute_error(test['Close'], test['Forecast'])

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
