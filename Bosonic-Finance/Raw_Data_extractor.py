import csv
import datetime as dt
import yfinance as yf
import numpy as np


class DataExtractor:
    def __init__(self, stock, start_date, end_date):
        self.data = yf.download (stock, start=start_date, end=end_date)
        self.stock_data = self.data ['Close'].pct_change ().dropna ().to_numpy () [::-1]

    @staticmethod
    def save_to_csv(forecasted_data, actual_data, mae, filename='forecast.csv'):
        with open (filename, 'w', newline='') as file:
            writer = csv.writer (file)
            writer.writerow (["Forecasted Data", "Actual Data", "Mean Absolute Error"])
            for i in range (len (forecasted_data)):
                writer.writerow ([forecasted_data [i], actual_data [i], mae])

    def forecast(self):
        probabilities = self.measure_quantum_state (self.create_quantum_state ())
        return numpy.random.Generator (self.stock_data, size=8, p=probabilities / np.sum (probabilities)), probabilities

    @staticmethod
    def create_quantum_state():
        return np.array ([1, 0])

    def measure_quantum_state(self, psi):
        return np.array ([np.abs ((psi [0] * np.cos (theta / 2) - psi [1] * np.sin (theta / 2))) ** 2
                          for theta in np.linspace (0, 2 * np.pi, len (self.stock_data))])


data_extractor = DataExtractor ('AAPL', dt.datetime (2020, 1, 1), dt.datetime (2023, 12, 31))
forecasted_data, probabilities = data_extractor.forecast ()
actual_data = data_extractor.stock_data [::-1] [-len (forecasted_data):]
mae = np.mean (np.abs (forecasted_data - actual_data))
data_extractor.save_to_csv (forecasted_data, actual_data, mae)
