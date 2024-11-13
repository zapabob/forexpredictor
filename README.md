# Forex Rate Prediction Application

![Forex Predictor](https://via.placeholder.com/1200x400.png?text=Forex+Rate+Prediction+Application)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Indicators](#technical-indicators)
- [Screenshots](#screenshots)
- [Troubleshooting](#troubleshooting)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

The **Forex Rate Prediction Application** is a Python-based tool designed to predict foreign exchange rates using machine learning techniques. Featuring a modern and intuitive GUI built with PyQt5, this application allows users to select currency pairs, initiate predictions, and visualize both real-time data and comprehensive technical indicators. Leveraging powerful libraries such as TensorFlow, Optuna, and Matplotlib, the application provides accurate predictions and insightful visualizations to aid in informed trading decisions.

## Features

- **Currency Pair Selection**: Choose from popular currency pairs like USD/JPY, EUR/JPY, and GBP/JPY.
- **Real-Time Price Monitoring**: Displays the latest exchange rates updated every minute.
- **Machine Learning Predictions**: Utilizes a Bidirectional LSTM model optimized with Optuna for accurate rate forecasting.
- **Technical Indicators Visualization**:
  - **SMA (Simple Moving Average)**: 20-period and 50-period moving averages.
  - **MACD (Moving Average Convergence Divergence)**: Includes MACD line, signal line, and histogram.
  - **Stochastic Oscillator**: %K and %D lines with overbought/oversold thresholds.
  - **ATR (Average True Range)**: Measures market volatility.
  - **Volatility**: Standard deviation of closing prices.
- **Performance Metrics**: Displays Mean Squared Error (MSE) and Sharpe Ratio to evaluate prediction performance.
- **Progress Bar**: Visual indicator of ongoing data processing and model training.
- **Comprehensive Backtesting**: Simulates trading strategies based on prediction signals to assess profitability.

## Installation

### Prerequisites

- **Python 3.7 or higher**: Ensure Python is installed on your system. You can download it from [here](https://www.python.org/downloads/).

### Clone the Repository

bash
git clone https://github.com/zapabob/forex-predictor.git
cd forex-predictorCreate a Virtual Environment (Optional but Recommended)
bash

python -m venv venv
Activate the virtual environment:

Windows:
bash

venv\Scripts\activate
macOS/Linux:
bash

source venv/bin/activate
Install Dependencies
bash

pip install -r requirements.txt
If you encounter issues, you can manually install the required packages:

bash

pip install pandas numpy yfinance pandas_ta matplotlib scikit-learn tensorflow optuna PyQt5
Note: Ensure that you have compatible versions of tensorflow and optuna. It's recommended to use the latest stable releases.

Usage
Run the application using the following command:

bash

python forex_predictor.py
Upon launching, the application will automatically start processing data for the default currency pair (USD/JPY). You can select different currency pairs from the dropdown menu and initiate new predictions as needed.

Technical Indicators
The application computes and displays the following technical indicators to provide deeper insights into market trends:

Simple Moving Average (SMA):

SMA 20: Average closing price over the last 20 periods.
SMA 50: Average closing price over the last 50 periods.
Moving Average Convergence Divergence (MACD):

MACD Line: Difference between the 12-period and 26-period EMA.
Signal Line: 9-period EMA of the MACD line.
MACD Histogram: Difference between the MACD line and the signal line.
Stochastic Oscillator:

%K Line: Measures the current closing price relative to the price range over a specified period.
%D Line: 3-period SMA of the %K line.
Overbought/Oversold Levels: Lines at 80 and 20 to indicate potential reversal points.
Average True Range (ATR):

Measures market volatility by decomposing the entire range of an asset price for a given period.
Volatility:

Standard deviation of closing prices over a 20-period window.
Screenshots
Main Interface

Technical Indicators

Troubleshooting
Data Download Errors:

Ensure you have a stable internet connection.
Verify that the selected currency pair is supported by Yahoo Finance.
If the application fails to retrieve data, try selecting a different currency pair or adjusting the date range.
Dependency Issues:

Make sure all required libraries are installed. Use pip list to verify.
For TensorFlow-related issues, ensure that your Python version is compatible and consider installing a version of TensorFlow that matches your system's specifications.
GUI Not Responding:

Long-running tasks may block the main thread. The application uses multithreading to handle data processing; ensure that the thread is not being blocked elsewhere in the code.
Font Display Issues:

The application uses the "Meiryo" font for Japanese characters. Ensure that this font is installed on your system. If not, you can change the font in the code to a different one that supports Japanese characters.
Dependencies
The application relies on the following Python libraries:

pandas: Data manipulation and analysis.
numpy: Numerical computations.
yfinance: Fetching financial data from Yahoo Finance.
pandas_ta: Technical Analysis indicators.
matplotlib: Plotting and visualization.
scikit-learn: Data preprocessing and evaluation metrics.
tensorflow: Building and training the machine learning model.
optuna: Hyperparameter optimization.
PyQt5: Building the graphical user interface.
Contributing
Contributions are welcome! Please follow these steps:

Fork the Repository: Click the "Fork" button at the top-right corner of this page.
Create a New Branch:
bash

git checkout -b feature/YourFeatureName
Commit Your Changes:
bash

git commit -m "Add Your Feature"
Push to Your Fork:


git push origin feature/YourFeatureName
Open a Pull Request: Navigate to the original repository and click "New Pull Request".
Please ensure that your code follows the project's coding standards and includes appropriate documentation.

License
This project is licensed under the MIT License.

Contact
For any inquiries or support, please contact:

Email: your.email@example.com
GitHub: yourusername
This application is developed for educational purposes and should not be used as the sole basis for trading decisions. Always perform your own research and consult with financial advisors before making investment decisions.










