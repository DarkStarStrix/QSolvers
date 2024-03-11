# Use the Stochastic nature of Quantum Mechanics to predict the future price of a stock simulate the future price of a stock

import numpy as np
import qutip as qt
from pandas_datareader import data as pdr
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

yf.pdr_override ()


class BosonicFinance:
    def __init__(self, stock, start_date, end_date):
        self.stock = stock
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.get_data ()
        self.stock_data = self.get_stock_data ()
        self.stock_data = self.stock_data ['Close']
        self.stock_data = self.stock_data.pct_change ()
        self.stock_data = self.stock_data.dropna ()
        self.stock_data = self.stock_data.to_numpy ()
        self.stock_data = self.stock_data [::-1]

    @staticmethod
    def create_quantum_state():
        psi = qt.basis (2, 0)
        psi = qt.sigmax () * psi
        return psi

    def get_data(self):
        data = pdr.get_data_yahoo (self.stock, start=self.start_date, end=self.end_date)
        return data

    def get_stock_data(self):
        return self.data

    def smooth_data(self, window_size=5):
        self.stock_data = pd.Series (self.stock_data).rolling (window=window_size).mean ().dropna ().to_numpy ()

    def plot_stock_data(self):
        plt.figure (figsize=(14, 7))
        plt.plot (self.data.index [:len (self.stock_data)], self.stock_data)
        plt.title ('Stock Data Over Time')
        plt.xlabel ('Date')
        plt.ylabel ('Stock Data')
        plt.grid (True)
        plt.show ()

    def measure_quantum_state(self, psi):
        probabilities = []
        for theta in np.linspace (0, 2 * np.pi, len (self.stock_data)):
            R = qt.Qobj ([[np.cos (theta / 2), -np.sin (theta / 2)], [np.sin (theta / 2), np.cos (theta / 2)]])
            M = R * qt.qeye (2) * R.dag ()
            probabilities.append (np.abs ((psi.dag () * M * psi)) ** 2)

        return np.array (probabilities)

    def forecast(self):
        psi = self.create_quantum_state ()
        probabilities = self.measure_quantum_state (psi)
        probabilities = probabilities / np.sum (probabilities)
        forecasted_data = np.random.choice (self.stock_data, size=8, p=probabilities)
        return forecasted_data

    def plot_predicted_stock_price(self):
        forecasted_data = self.forecast ()
        forecast_dates = self.data.index [-len (forecasted_data):]
        actual_data = self.stock_data [::-1] [-len (forecasted_data):]

        trace1 = go.Scatter (
            x=self.data.index [:len (self.stock_data)],
            y=self.stock_data,
            mode='lines',
            name='Historical Data',
            line=dict (color='green')
        )

        trace2 = go.Scatter (
            x=forecast_dates,
            y=forecasted_data,
            mode='lines',
            name='Forecasted Data',
            line=dict (color='blue')
        )

        trace3 = go.Scatter (
            x=forecast_dates,
            y=actual_data,
            mode='lines',
            name='Actual Data',
            line=dict (color='orange')
        )

        layout = go.Layout (
            title='Stock Data Over Time',
            xaxis=dict (title='Date'),
            yaxis=dict (title='Stock Data'),
            showlegend=True
        )

        fig = go.Figure (data=[trace1, trace2, trace3], layout=layout)
        fig.show ()

        mae = np.mean (np.abs (forecasted_data - actual_data))
        print (f'Mean Absolute Error: {mae}')


bosonic_finance = BosonicFinance ('AAPL', dt.datetime (2020, 1, 1), dt.datetime (2023, 12, 31))
bosonic_finance.smooth_data (window_size=5)
bosonic_finance.plot_stock_data ()
bosonic_finance.plot_predicted_stock_price ()

# Mean Absolute Error: 0.007619796312381751
