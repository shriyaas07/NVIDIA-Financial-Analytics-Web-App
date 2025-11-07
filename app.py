from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objs as go
import plotly.offline as pyo
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.io as pio

# -------------------------
# Initialize Flask app
# -------------------------
app = Flask(__name__)

# -------------------------
# Load Model and Data
# -------------------------
model = pickle.load(open('nvidia_lr_model.pkl', 'rb'))
data = pd.read_csv('nvidia_actual_vs_predicted.csv')

# Calculate model performance
try:
    r2 = r2_score(data['Actual_Price'], data['Predicted_Price'])
    rmse = np.sqrt(mean_squared_error(data['Actual_Price'], data['Predicted_Price']))
    model_summary = f"Model R² Score: {r2:.3f} | RMSE: {rmse:.2f}"
except Exception as e:
    model_summary = f"Performance metrics unavailable: {e}"

# Ensure Date column exists
if 'Date' not in data.columns:
    data['Date'] = pd.date_range(start='2024-01-01', periods=len(data), freq='D')


# -------------------------
# ROUTE 1: Prediction Dashboard
# -------------------------
@app.route('/')
def home():
    # Chart
    trace_actual = go.Scatter(x=data['Date'], y=data['Actual_Price'], mode='lines',
                              name='Actual Price', line=dict(color='green'))
    trace_pred = go.Scatter(x=data['Date'], y=data['Predicted_Price'], mode='lines',
                            name='Predicted Price', line=dict(color='orange'))
    layout = go.Layout(title='NVIDIA Stock: Actual vs Predicted Prices',
                       xaxis_title='Date', yaxis_title='Price (USD)',
                       template='plotly_white')
    fig = go.Figure(data=[trace_actual, trace_pred], layout=layout)
    graph_html = pyo.plot(fig, output_type='div')

    return render_template('index.html', prediction=None, graph_html=graph_html, model_summary=model_summary)


# -------------------------
# ROUTE 2: Predict API
# -------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        prev_close = float(request.form['prev_close'])
        ma7 = float(request.form['ma7'])
        ma30 = float(request.form['ma30'])
        volatility = float(request.form['volatility'])

        input_data = np.array([[prev_close, ma7, ma30, volatility]])
        predicted_price = float(np.ravel(model.predict(input_data))[0])

        trace_actual = go.Scatter(x=data['Date'], y=data['Actual_Price'], mode='lines',
                                  name='Actual Price', line=dict(color='green'))
        trace_pred = go.Scatter(x=data['Date'], y=data['Predicted_Price'], mode='lines',
                                name='Predicted Price', line=dict(color='orange'))
        layout = go.Layout(title='NVIDIA Stock: Actual vs Predicted Prices',
                           xaxis_title='Date', yaxis_title='Price (USD)',
                           template='plotly_white')
        fig = go.Figure(data=[trace_actual, trace_pred], layout=layout)
        graph_html = pyo.plot(fig, output_type='div')

        return render_template('index.html', prediction=round(predicted_price, 2),
                               graph_html=graph_html, model_summary=model_summary)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {e}", graph_html="", model_summary=model_summary)


# -------------------------
# ROUTE 3: Analytics Dashboard
# -------------------------
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

@app.route('/dashboard')
def dashboard():
    df = pd.read_csv('nvidia_stock_data.csv')

    # Parse date correctly (fixes earlier error)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    # Convert numeric columns to proper numeric types
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(',', '', regex=False)
            .str.replace('₹', '', regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing values
    df = df.dropna(subset=numeric_cols)


    # Sort by date
    df = df.sort_values('date')

    # ---- Candlestick + Moving Average ----
    df['MA20'] = df['Close'].rolling(window=20).mean()

    fig_candle = go.Figure()

    # Candlestick
    fig_candle.add_trace(go.Candlestick(
        x=df['date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick'
    ))

    # Moving Average
    fig_candle.add_trace(go.Scatter(
        x=df['date'],
        y=df['MA20'],
        mode='lines',
        line=dict(color='orange', width=2),
        name='20-Day Moving Avg'
    ))

    fig_candle.update_layout(
        title='NVIDIA Candlestick Chart with 20-Day Moving Average',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        height=500
    )

    graph_candle = fig_candle.to_html(full_html=False)

    # ---- Trade Volume ----
    fig_vol = px.bar(df, x='date', y='Volume', title='Trade Volume Over Time')
    fig_vol.update_layout(template='plotly_white', height=400)
    graph_vol = fig_vol.to_html(full_html=False)

    # ---- Closing Price ----
    fig_close = px.line(df, x='date', y='Close', title='NVIDIA Closing Price Over Time')
    fig_close.update_layout(template='plotly_white', height=400)
    graph_close = fig_close.to_html(full_html=False)

    return render_template(
        'dashboard.html',
        graph_candle=graph_candle,
        graph_vol=graph_vol,
        graph_close=graph_close
    )



# -------------------------
# Run App
# -------------------------
if __name__ == '__main__':
    app.run(debug=True)
