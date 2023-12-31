#!/usr/bin/env python
# coding: utf-8

# In[2]:


import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def get_stock_data(symbol):
    try:
        data = yf.download(symbol, start='2020-01-01', end='2023-01-01')
        return data
    except Exception as e:
        print(f"Failed to download data for {symbol}: {e}")
        return None

def predict_stock_price(symbol):
    data = get_stock_data(symbol)
    if data is None or data.empty:
        print(f"No data available for {symbol}. Skipping.")
        return None

    data['Target'] = data['Close'].shift(-1)
    data.dropna(inplace=True)

    if len(data) < 2:  # Ensure there is enough data
        print(f"Not enough data for {symbol}. Skipping.")
        return None

    X = data[['Close']]
    y = data['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    return rmse

symbols = ['7203.T', '9432.T']  # Your list of symbols
for symbol in symbols:
    error = predict_stock_price(symbol)
    if error is not None:
        print(f"RMSE for {symbol}: {error}")


# In[3]:


import matplotlib.pyplot as plt

# RMSEの値
rmse_values = [28.918110273476604, 1.8027915515694954]
symbols = ['7203.T', '9432.T']

# 棒グラフの作成
plt.bar(symbols, rmse_values, color='blue')

# タイトルとラベルの追加
plt.title('RMSE of Stock Predictions')
plt.xlabel('Stock Symbol')
plt.ylabel('RMSE')

# グラフの表示
plt.show()


# In[4]:


import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# 株価と為替レートのデータをダウンロード
stock_data = yf.download('7203.T', start='2020-01-01', end='2023-01-01')
forex_data = yf.download('JPY=X', start='2020-01-01', end='2023-01-01')

# データの前処理：終値を使用
stock_data = stock_data[['Close']].rename(columns={'Close': 'Stock_Close'})
forex_data = forex_data[['Close']].rename(columns={'Close': 'Forex_Close'})

# データを結合
data = pd.concat([stock_data, forex_data], axis=1).dropna()

# 相関を基にした特徴量エンジニアリング
# ここでは、単純に前日の為替レートを特徴量として使用しています
data['Forex_Close_Lag1'] = data['Forex_Close'].shift(1)

# 目的変数と特徴量の設定
X = data[['Stock_Close', 'Forex_Close_Lag1']].dropna()  # NaNがあれば除去
y = data['Stock_Close'].shift(-1).dropna()  # 次の日の株価を予測

# 最後のNaN値を持つ行を削除
X = X.iloc[:-1, :]
y = y.iloc[:-1]

# データセットの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ランダムフォレストモデルの構築
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 予測と評価
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print(f'RMSE: {rmse}')


# In[ ]:




