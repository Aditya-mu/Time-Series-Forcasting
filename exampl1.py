import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from plotnine import ggplot, aes, geom_line, labs, theme, element_text

# Fetch stock data from Yahoo Finance
stock_symbol = 'AAPL'  # You can change this to any stock symbol
start_date = '2020-01-01'
end_date = '2023-01-01'

data = yf.download(stock_symbol, start=start_date, end=end_date)

# Prepare the data
data = data[['Close']].dropna()

# Reset index for plotting
data.reset_index(inplace=True)

# Plot the closing price
plot1 = (
    ggplot(data, aes(x='Date', y='Close')) +
    geom_line() +
    labs(title=f'{stock_symbol} Stock Closing Prices', x='Date', y='Price') +
    theme(axis_text_x=element_text(rotation=45, hjust=1))
)
print(plot1)

# Split the data into training and test sets
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# Add a 'Type' column for distinguishing train/test data
train['Type'] = 'Training'
test['Type'] = 'Test'

# Combine train and test data for plotting
combined_data = pd.concat([train, test])

# Plot the training and test data
plot2 = (
    ggplot(combined_data, aes(x='Date', y='Close', color='Type')) +
    geom_line() +
    labs(title='Train/Test Split', x='Date', y='Price') +
    theme(axis_text_x=element_text(rotation=45, hjust=1))
)
print(plot2)

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
test = test.set_index('Date')
test = test.join(forecast_df)

# Plot the forecast along with the actual test data
plot3 = (
    ggplot(test.reset_index(), aes(x='Date')) +
    geom_line(aes(y='Close', color='Actual')) +
    geom_line(aes(y='Forecast', color='Forecast')) +
    labs(title='Stock Price Prediction', x='Date', y='Price') +
    theme(axis_text_x=element_text(rotation=45, hjust=1))
)
print(plot3)

# Calculate MSE and MAE
mse = mean_squared_error(test['Close'], test['Forecast'])
mae = mean_absolute_error(test['Close'], test['Forecast'])

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
