# ============================================
# Trading Algorithm Project 
# ============================================

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# STEP 1: DATA COLLECTION
# -------------------------------
ticker = "AAPL"   # Change this to any stock symbol you want
today = datetime.today().strftime('%Y-%m-%d')

# FIX: avoid multi-index columns
data = yf.download(ticker, start="2018-01-01", end=today, group_by="column")

print("✅ Data downloaded:", data.shape)

# -------------------------------
# STEP 2: FEATURE ENGINEERING
# -------------------------------
def compute_indicators(df):
    # Simple Moving Average (14-day)
    df["SMA"] = df["Close"].rolling(window=14).mean()
    
    # Exponential Moving Average (14-day)
    df["EMA"] = df["Close"].ewm(span=14, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD (12-day EMA - 26-day EMA)
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    
    return df

data = compute_indicators(data)
data.dropna(inplace=True)

# Save for verification
data.to_csv("apple_stock_with_indicators.csv")
print("✅ Indicators added and saved to apple_stock_with_indicators.csv")

# -------------------------------
# STEP 3: MACHINE LEARNING MODELS
# -------------------------------
# Target: 1 if tomorrow's Close > today's Close, else 0
data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

features = ["SMA", "EMA", "RSI", "MACD", "Volume"]
X = data[features].iloc[:-1]  # drop last row
y = data["Target"].iloc[:-1]

print("✅ Dataset ready for training:", X.shape, "samples")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("📊 Random Forest Accuracy:", accuracy_score(y_test, rf_preds))

# --- XGBoost ---
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
print("📊 XGBoost Accuracy:", accuracy_score(y_test, xgb_preds))



import matplotlib.pyplot as plt

# -------------------------------
# STEP 4: GENERATE TRADING SIGNALS
# -------------------------------

# Align rows: drop last row so X, predictions, and data match
data = data.iloc[:-1]

# Predictions from both models
data["RF_Signal"] = rf.predict(X)
data["XGB_Signal"] = xgb.predict(X)

# Combine signals (majority voting: Buy if either RF or XGB says Buy)
data["Final_Signal"] = ((data["RF_Signal"] + data["XGB_Signal"]) >= 1).astype(int)

print("✅ Signals generated. Preview:")
print(data[["Close", "RF_Signal", "XGB_Signal", "Final_Signal"]].head(15))

# -------------------------------
# STEP 4B: VISUALIZE BUY/SELL SIGNALS
# -------------------------------
plt.figure(figsize=(14,7))
plt.plot(data.index, data["Close"], label="Close Price", alpha=0.5)

# Buy signals (Final_Signal = 1)
plt.scatter(
    data.index[data["Final_Signal"] == 1],
    data["Close"][data["Final_Signal"] == 1],
    label="Buy Signal",
    marker="^",
    color="green",
    alpha=1
)

# Sell signals (Final_Signal = 0)
plt.scatter(
    data.index[data["Final_Signal"] == 0],
    data["Close"][data["Final_Signal"] == 0],
    label="Sell Signal",
    marker="v",
    color="red",
    alpha=1
)

plt.title(f"{ticker} Trading Signals")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()




# ============================================
# STEP 5: BACKTESTING
# ============================================

initial_balance = 10000  # starting with $10,000
balance = initial_balance
position = 0  # 0 = no stock, >0 = holding shares
balance_history = []

for i in range(len(data)):
    price = data["Close"].iloc[i].item()
    signal = int(data["Final_Signal"].iloc[i].item())

    # Buy signal (only buy if not already holding)
    if signal == 1 and position == 0:
        position = balance / price   # buy as many shares as possible
        balance = 0

    # Sell signal (sell if currently holding)
    elif signal == 0 and position > 0:
        balance = position * price   # sell all shares
        position = 0

    # Track daily portfolio value
    portfolio_value = balance + position * price
    balance_history.append(portfolio_value)

# Final portfolio value
final_value = balance_history[-1]
profit = final_value - initial_balance
returns = (final_value / initial_balance - 1) * 100

print("\n💰 Backtest Results")
print(f"Initial Balance: ${initial_balance:,.2f}")
print(f"Final Balance:   ${final_value:,.2f}")
print(f"Net Profit:      ${profit:,.2f}")
print(f"Return:          {returns:.2f}%")

# -------------------------------
# Sharpe Ratio (Risk-adjusted performance)
# -------------------------------
returns_series = pd.Series(balance_history).pct_change().dropna()
sharpe_ratio = (returns_series.mean() / returns_series.std()) * (252**0.5)  # annualized


# ============================================
# 🧠 ADVANCED PERFORMANCE METRICS (Add this below)
# ============================================
import numpy as np

# Convert balance history to a Pandas Series
portfolio = pd.Series(balance_history, index=data.index)

# CAGR - Compound Annual Growth Rate
years = (data.index[-1] - data.index[0]).days / 365
cagr = (final_value / initial_balance) ** (1 / years) - 1

# Maximum Drawdown
rolling_max = portfolio.cummax()
drawdown = (portfolio - rolling_max) / rolling_max
max_drawdown = drawdown.min()

# Volatility (Annualized)
daily_returns = portfolio.pct_change().dropna()
volatility = daily_returns.std() * np.sqrt(252)

# Calmar Ratio (CAGR / |Max Drawdown|)
calmar_ratio = cagr / abs(max_drawdown)

# 🖨️ Display Results
print("\n📊 Advanced Performance Metrics")
print(f"CAGR:            {cagr*100:.2f}%")
print(f"Max Drawdown:    {max_drawdown*100:.2f}%")
print(f"Volatility:      {volatility*100:.2f}%")
print(f"Sharpe Ratio:    {sharpe_ratio:.2f}")
print(f"Calmar Ratio:    {calmar_ratio:.2f}")

# -------------------------------
# PLOT PORTFOLIO GROWTH
# -------------------------------
plt.figure(figsize=(14,6))
plt.plot(data.index, balance_history, label="Portfolio Value")
plt.title(f"{ticker} Backtest Portfolio Growth")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid()
plt.show()


# ============================================
# STEP 6: LSTM Deep Learning Model
# ============================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

print("\n🔮 Training LSTM model...")

# Features and labels
X_lstm = data[["SMA", "EMA", "RSI", "MACD", "Volume"]].values
y_lstm = data["Target"].values

# Define sequence length (how many days of history LSTM sees)
sequence_length = 20
generator = TimeseriesGenerator(X_lstm, y_lstm, length=sequence_length, batch_size=32)

# Build model
lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(sequence_length, X_lstm.shape[1])),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train
history = lstm_model.fit(generator, epochs=5, verbose=1)

# Predictions
y_pred_probs = lstm_model.predict(generator)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Align with data index
data_lstm = data.iloc[sequence_length:].copy()
data_lstm["LSTM_Signal"] = y_pred

print("✅ LSTM signals generated. Preview:")
print(data_lstm[["Close", "LSTM_Signal"]].head(15))

# -------------------------------
# PLOT LSTM SIGNALS
# -------------------------------
plt.figure(figsize=(14,7))
plt.plot(data_lstm.index, data_lstm["Close"], label="Close Price", alpha=0.5)

# Buy
plt.scatter(
    data_lstm.index[data_lstm["LSTM_Signal"] == 1],
    data_lstm["Close"][data_lstm["LSTM_Signal"] == 1],
    label="Buy Signal",
    marker="^",
    color="blue",
    alpha=1
)

# Sell
plt.scatter(
    data_lstm.index[data_lstm["LSTM_Signal"] == 0],
    data_lstm["Close"][data_lstm["LSTM_Signal"] == 0],
    label="Sell Signal",
    marker="v",
    color="orange",
    alpha=1
)

plt.title(f"{ticker} LSTM Trading Signals")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()
