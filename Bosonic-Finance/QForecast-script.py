# Use the Stochastic nature of Quantum Mechanics to predict the future price of a stock simulate the future price of a stock

import numpy as np
import qutip as qt
from pandas_datareader import data as pdr
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

yf.pdr_override()


class BosonicFinance:
    def __init__(self, stock, start_date, end_date):
        self.stock = stock
        self.start_date = start_date
        self.end_date = end_date
        self.data = pdr.get_data_yahoo(self.stock, start=self.start_date, end=self.end_date)
        self.stock_data = self.data['Close'].pct_change().dropna().to_numpy()[::-1]

    @staticmethod
    def create_quantum_state():
        return qt.sigmax() * qt.basis(2, 0)

    def smooth_data(self, window_size=5):
        self.stock_data = pd.Series(self.stock_data).rolling(window=window_size).mean().dropna().to_numpy()

    def plot_stock_data(self):
        plt.figure(figsize=(14, 7))
        plt.plot(self.data.index[:len(self.stock_data)], self.stock_data)
        plt.title('Stock Data Over Time')
        plt.xlabel('Date')
        plt.ylabel('Stock Data')
        plt.grid(True)
        plt.show()

    def measure_quantum_state(self, psi):
        return np.array([np.abs((psi.dag() * qt.Qobj([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]]).dag() * psi)) ** 2
                         for theta in np.linspace(0, 2 * np.pi, len(self.stock_data))])

    def forecast(self):
        probabilities = self.measure_quantum_state(self.create_quantum_state())
        return np.random.choice(self.stock_data, size=8, p=probabilities / np.sum(probabilities))

    def plot_predicted_stock_price(self):
        forecasted_data = self.forecast()
        forecast_dates = self.data.index[-len(forecasted_data):]
        actual_data = self.stock_data[::-1][-len(forecasted_data):]

        fig = go.Figure(data=[go.Scatter(x=self.data.index[:len(self.stock_data)], y=self.stock_data, mode='lines', name='Historical Data', line=dict(color='green')),
                              go.Scatter(x=forecast_dates, y=forecasted_data, mode='lines', name='Forecasted Data', line=dict(color='blue')),
                              go.Scatter(x=forecast_dates, y=actual_data, mode='lines', name='Actual Data', line=dict(color='orange'))],
                        layout=go.Layout(title='Stock Data Over Time', xaxis=dict(title='Date'), yaxis=dict(title='Stock Data'), showlegend=True))
        fig.show()

        print(f'Mean Absolute Error: {np.mean(np.abs(forecasted_data - actual_data))}')


bosonic_finance = BosonicFinance('AAPL', dt.datetime(2020, 1, 1), dt.datetime(2023, 12, 31))
bosonic_finance.smooth_data(window_size=5)
bosonic_finance.plot_stock_data()
bosonic_finance.plot_predicted_stock_price()
