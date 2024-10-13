# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 20:27:22 2024

@author: SURUTHI S
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 20:47:48 2024

@author: SURUTHI S
"""

# %% Import

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,GRU,Dense,Dropout
from sklearn.metrics import mean_squared_error,mean_absolute_error

# %% Data loading
data_ext = pd.read_csv(r"gold_price.csv",
                       parse_dates=['date'], index_col='date')

str_cols = ['price','open','high','low']
for col in str_cols:
  data_ext[col] = pd.to_numeric(data_ext[col].str.replace(',', ''))

data_ext.rename(columns={'index':'date','open':'Open','high':'High','low':'Low','price':'Close','volume':'Volume'},inplace=True)

data_ext.dtypes

data_ext.isna().sum()

data_ext.ffill(inplace=True)

data_ext.isna().sum()

# Check if the date index is consecutive

start_date = data_ext.index.min()
end_date = data_ext.index.max()

def check_date_gaps(df):
  date_range = pd.date_range(start=start_date, end=end_date)

  if len(df.index) == len(date_range) and all(df.index == date_range):
    print("Date index is consecutive\n")
  else:
    print("Date index is NOT consecutive\n")
    missing_dates = date_range.difference(df.index)
    print("Missing dates:", missing_dates)


check_date_gaps(data_ext)


date_range = pd.date_range(start=start_date, end=end_date)

# Reindex the DataFrame to include all dates in the range
data_ext = data_ext.reindex(date_range)

data_ext.ffill(inplace=True)


check_date_gaps(data_ext)


data_ext = data_ext['2023-01-01':]

data_description = data_ext.describe()

data_ext.head()
data_ext.tail()



data = data_ext.reset_index()

data.tail()

data.dtypes

data.index.dtype

data.isna().sum()

data.rename(columns={'index':'date','open':'Open','high':'High','low':'Low','price':'Close','volume':'Volume'},inplace=True)


date_col = pd.to_datetime(data['date']).dt.strftime("%Y-%m-%d")

ohlc_col = data[['Open','High','Low','Close']]


#%% 1. Plotting

# 1.1 Plotting each axis and O vs C and L vs H

fig,axes = plt.subplots(nrows=3,ncols=2, figsize=(10, 8))
color = ['red','green','yellow','blue']

for i in range(4):
    row = i//2
    col = i%2
    
    axes[row,col].plot(data_ext[ohlc_col.columns[i]], color = color[i])
    axes[row,col].set_title(ohlc_col.columns[i])
    axes[row,col].set_xlabel("Year")
    axes[row,col].set_ylabel("Price ")
 

axes[2,0].plot(data_ext['Open'],label='Open', color = 'red')
axes[2,0].plot(data_ext['Close'],label='Close', color = 'green')
axes[2,0].set_title("Open vs Close")

axes[2,1].plot(data_ext['Low'],label='Low', color = 'yellow')
axes[2,1].plot(data_ext['High'],label='High', color = 'blue')
axes[2,1].set_title("Low vs High")

plt.tight_layout()
plt.show()


# 1.2 Distribution of the Data

for i, column in enumerate(ohlc_col.columns, 1):
    plt.subplot(2, 2, i)  # Create a 2x2 grid of subplots
    sns.histplot(ohlc_col[column], kde=True, color='blue', label='Original', stat='density', bins=10)
    # sns.histplot(pd.DataFrame(ohlc_col_scaled),kde=True, color='orange', label='Scaled', stat='density', bins=10)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.legend()

plt.tight_layout()
plt.show()

# plot follows bi-modal distribution


# 1.3 Plotting ACF & PACF plots

fig_ts,axes_ts = plt.subplots(nrows=4,ncols=3, figsize=(10, 10))

for i in range(4):
    
    axes_ts[i,0].plot(data_ext[ohlc_col.columns[i]], color = color[i])
    axes_ts[i,0].set_title(f'{ohlc_col.columns[i]} Original TS')
    axes_ts[i,0].set_xlabel("Years")
    axes_ts[i,0].set_ylabel("Price")
    
    plot_data = data_ext[ohlc_col.columns[i]].diff(1).dropna()
    plot_acf(plot_data, ax=axes_ts[i,1])
    plot_pacf(plot_data, ax=axes_ts[i,2])

plt.tight_layout()
plt.show()

# 1.4 Plotting Seasonal_decompose

for col in range(4):
   
    decomposition = seasonal_decompose(data_ext[ohlc_col.columns[col]], period=4)
    plt.figure(figsize=(10, 10))
    decomposition.plot()
    plt.tight_layout()  
    plt.show()


#%% 2. Scaling

scaler = StandardScaler()
ohlc_col_scaled = scaler.fit_transform(ohlc_col)

scaled_data_description = pd.DataFrame(ohlc_col_scaled).describe()

# %% Train-Test Split

# 1. train_generator / manual split and visualize
# 2. Fit 
# 3. Predict
# 4. Metrics

# 1. Lstm,NBeats, SeqtoSeq

window_size = 7
X,y = [], []

for i in range(window_size, len(ohlc_col)):
    X.append(ohlc_col_scaled[i-window_size: i])
    y.append(ohlc_col_scaled[i])

X = np.array(X)
y = np.array(y)


print("Shape of X", X.shape)
print("Shape of y", y.shape)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.10,shuffle=False)

# %% LSTM

lstm = Sequential()
lstm.add(LSTM(512, activation='relu', input_shape=(x_train.shape[1],x_train.shape[2]),
              return_sequences=True))
lstm.add(LSTM(128, activation='relu',return_sequences=True))
lstm.add(Dropout(0.20))
lstm.add(LSTM(64, activation='relu',return_sequences=True))
lstm.add(Dropout(0.30))
lstm.add(LSTM(32, activation='relu',return_sequences=False))
lstm.add(Dropout(0.20))
lstm.add(Dense(64, activation='relu'))
lstm.add(Dense(32, activation='relu'))
lstm.add(Dense(y.shape[1]))
lstm.compile(optimizer = 'adam', loss = 'mean_squared_error')
lstm.summary()


lstm_history = lstm.fit(x_train,y_train,epochs=50,batch_size=16,validation_split=0.1)


# Make predictions on the test set
lstm_y_pred = lstm.predict(x_test)


train_dates = date_col[:533]
test_dates = date_col[533:-7]


lstm_y_pred_inv = scaler.inverse_transform(lstm_y_pred)  # Inverse transform the predicted values
y_test_inv = scaler.inverse_transform(y_test)  # Inverse transform the actual test values

# test_dates = date_col[956:-7]

lstm_result = pd.DataFrame({
    'Date': test_dates,  
    'Actual_Open': y_test_inv[:, 0],
    'Predicted_Open': lstm_y_pred_inv[:, 0],
    'Actual_High': y_test_inv[:, 1],
    'Predicted_High': lstm_y_pred_inv[:, 1],
    'Actual_Low': y_test_inv[:, 2],
    'Predicted_Low': lstm_y_pred_inv[:, 2],
    'Actual_Close': y_test_inv[:, 3],
    'Predicted_Close': lstm_y_pred_inv[:, 3]
})

lstm_result.head()

lstm_diff = pd.DataFrame()

# Calculate differences

lstm_diff['Open_Diff'] = lstm_result['Actual_Open'] - lstm_result['Predicted_Open']
lstm_diff['High_Diff'] = lstm_result['Actual_High'] - lstm_result['Predicted_Open']
lstm_diff['Low_Diff'] = lstm_result['Actual_Low'] - lstm_result['Predicted_Open']
lstm_diff['Close_Diff'] = lstm_result['Actual_Close'] - lstm_result['Predicted_Open']

lstm_pred_description = lstm_diff.describe()

# Display the DataFrame

lstm_diff.head()

# Evaluate the model using time series evaluation metrics
lstm_mse = mean_squared_error(y_test, lstm_y_pred)
lstm_mae = mean_absolute_error(y_test, lstm_y_pred)
lstm_rmse = np.sqrt(lstm_mse)

print(f'Mean Squared Error (MSE): {lstm_mse}')
print(f'Mean Absolute Error (MAE): {lstm_mae}')
print(f'Root Mean Squared Error (RMSE): {lstm_rmse}')
# lstm_mse = mean_squared_error(y_true, y_pred)

n_future = 10

forecast_dates = pd.date_range(list(date_col)[-1],periods=n_future, freq='1d').tolist()
forecast = lstm.predict(x_train[-n_future:])

y_pred_future = scaler.inverse_transform(forecast)

forecast_df = pd.DataFrame(y_pred_future, columns = ['Open','High','Low','Close'])
forecast_df['Date'] = forecast_dates

forecast_df = forecast_df[['Date','Open','High','Low','Close']]



# Plotting the training and validation loss over epochs

plt.figure(figsize=(10,6))
plt.plot(lstm_history.history['loss'], label='Training Loss')
plt.plot(lstm_history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss (Mean Squared Error)')
plt.legend()
plt.show()




# %% Further Statistical Analysis

# 1.Correlation btwn actual and predicted

lstm_corr = lstm_result.corr()

# 2. comparing the distributions of actual and predicted
# Open Price comparison
plt.subplot(2, 2, 1)
sns.histplot(lstm_result['Actual_Open'], color='blue', label='Actual Open', kde=True)
sns.histplot(lstm_result['Predicted_Open'], color='red', label='Predicted Open', kde=True)
plt.title('Open Price Distribution')
plt.legend()


 # %% GRU

gru = Sequential()
gru.add(GRU(512, activation='relu', input_shape=(X.shape[1],X.shape[2]),
              return_sequences=True))
gru.add(GRU(128, activation='relu',return_sequences=True))
gru.add(Dropout(0.20))
gru.add(GRU(64, activation='relu',return_sequences=True))
gru.add(Dropout(0.30))
gru.add(GRU(32, activation='relu',return_sequences=False))
gru.add(Dropout(0.20))
gru.add(Dense(64, activation='relu'))
gru.add(Dense(32, activation='relu'))
gru.add(Dense(y.shape[1]))
gru.compile(optimizer = 'adam', loss = 'mean_squared_error')
gru.summary()

gru_history = gru.fit(X,y,epochs=20,batch_size=16,validation_split=0.1)

# Make predictions on the test set
gru_y_pred = gru.predict(x_test)

# Evaluate the model using time series evaluation metrics
gru_mse = mean_squared_error(y_test, gru_y_pred)
gru_mae = mean_absolute_error(y_test, gru_y_pred)
gru_rmse = np.sqrt(gru_mse)

print(f'Mean Squared Error (MSE): {gru_mse}')
print(f'Mean Absolute Error (MAE): {gru_mae}')
print(f'Root Mean Squared Error (RMSE): {gru_rmse}')


gru_forecast = gru.predict(X[-n_future:])
gru_y_pred_future = scaler.inverse_transform(gru_forecast)
gru_forecast_df = pd.DataFrame(gru_y_pred_future, columns = ['Open','High','Low','Close'])
gru_forecast_df['Date'] = forecast_dates
gru_forecast_df = gru_forecast_df[['Date','Open','High','Low','Close']]



