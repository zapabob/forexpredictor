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

```bash
git clone https://github.com/zapabob/forex-predictor.git
cd forex-predictor
