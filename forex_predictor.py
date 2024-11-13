# 必要なライブラリのインポート
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
import optuna
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QComboBox, QVBoxLayout, QWidget,
    QHBoxLayout, QMessageBox, QProgressBar
)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# グラフ描画のバックエンド設定
plt.switch_backend('Qt5Agg')

# Matplotlibで日本語フォントを設定
import matplotlib
matplotlib.rcParams['font.family'] = 'Meiryo'

# データ処理用のスレッドクラス
class DataProcessingThread(QThread):
    # シグナルの定義
    finished = pyqtSignal(pd.DataFrame, dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)  # 進捗表示用シグナル

    def __init__(self, ticker):
        super().__init__()
        self.ticker = ticker

    def run(self):
        try:
            self.progress.emit(0)  # 開始
            print("Data processing started.")

            # データの取得
            end = datetime.datetime.today()
            start = end - datetime.timedelta(days=730)  # 過去2年間（730日以内）
            print(f"Downloading data for {self.ticker} from {start} to {end} with interval='1h'")
            df = yf.download(
                self.ticker,
                start=start,
                end=end,
                interval='1h',
                progress=False
            )

            self.progress.emit(10)
            print("Data downloaded successfully.")

            # データ取得の確認
            if df.empty:
                self.error.emit("データの取得に失敗しました。期間を調整してください。")
                print("Downloaded data is empty.")
                return

            # カラムのフラット化（MultiIndex対策）
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]

            # カラム名にティッカーシンボルが付加されている場合は削除
            suffix = f"_{self.ticker}"
            df.columns = [
                col.replace(suffix, '') if isinstance(col, str) and col.endswith(suffix) else col
                for col in df.columns
            ]

            # 不要なアンダースコアを削除（念のため）
            df.columns = [col.rstrip('_') if isinstance(col, str) else col for col in df.columns]
            print("Cleaned column names.")

            self.progress.emit(20)

            # 必要なカラムが存在するか確認
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.error.emit(f"以下のカラムがデータに存在しません: {missing_columns}")
                print(f"Missing columns: {missing_columns}")
                return

            # 欠損値の処理
            df.dropna(inplace=True)
            print("Dropped missing values.")

            self.progress.emit(30)

            # テクニカル指標の計算
            # 移動平均線 (SMA)
            df['SMA_20'] = ta.sma(df['Close'], length=20)
            df['SMA_50'] = ta.sma(df['Close'], length=50)
            print("Calculated SMA indicators.")

            self.progress.emit(40)

            # MACDの計算と結合
            macd = ta.macd(df['Close'])
            if isinstance(macd.columns, pd.MultiIndex):
                macd.columns = ['_'.join(map(str, col)).strip() for col in macd.columns.values]
            df = pd.concat([df, macd], axis=1)
            print("Calculated and merged MACD indicators.")

            self.progress.emit(50)

            # ストキャスティクスの計算と結合
            stoch = ta.stoch(df['High'], df['Low'], df['Close'])
            if isinstance(stoch.columns, pd.MultiIndex):
                stoch.columns = ['_'.join(map(str, col)).strip() for col in stoch.columns.values]
            df = pd.concat([df, stoch], axis=1)
            print("Calculated and merged Stochastic indicators.")

            self.progress.emit(60)

            # ATR
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])
            print("Calculated ATR.")

            # ボラティリティ（標準偏差）
            df['Volatility'] = df['Close'].rolling(window=20).std()
            print("Calculated Volatility.")

            # 移動平均の差分
            df['SMA_diff'] = df['SMA_20'] - df['SMA_50']
            print("Calculated SMA difference.")

            # ラグ特徴量の作成（過去3時間分）
            for lag in range(1, 4):
                df[f'Lag_{lag}'] = df['Close'].shift(lag)
            print("Created lag features.")

            # 欠損値の再処理
            df.dropna(inplace=True)
            print("Dropped missing values after feature engineering.")

            self.progress.emit(70)

            # MACDとストキャスティクスのカラム名
            expected_macd = ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
            expected_stoch = ['STOCHk_14_3_3', 'STOCHd_14_3_3']

            # MACDとストキャスティクスのカラムが存在するか確認
            missing_macd = [col for col in expected_macd if col not in df.columns]
            missing_stoch = [col for col in expected_stoch if col not in df.columns]

            if missing_macd or missing_stoch:
                missing = missing_macd + missing_stoch
                self.error.emit(f"以下のテクニカル指標がデータフレームに存在しません: {missing}")
                print(f"Missing technical indicators: {missing}")
                return

            # 特徴量とターゲットの設定
            features = [
                'Close', 'SMA_20', 'SMA_50',
                *expected_macd,
                *expected_stoch,
                'ATR', 'Volatility', 'SMA_diff',
                'Lag_1', 'Lag_2', 'Lag_3'
            ]

            # 特徴量がデータフレームに存在するか確認
            missing_features = [feature for feature in features if feature not in df.columns]
            if missing_features:
                self.error.emit(f"以下の特徴量がデータフレームに存在しません:\n{missing_features}")
                print(f"Missing features: {missing_features}")
                return

            data = df[features]
            target = df['Close'].shift(-1)  # 1時間先の価格を予測

            # 最後の行の欠損値を削除
            data = data.iloc[:-1]
            target = target.iloc[:-1]
            print("Prepared features and target.")

            self.progress.emit(80)

            # データの標準化（特徴量とターゲットでスケーラーを分ける）
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()

            scaled_data = scaler_X.fit_transform(data)
            scaled_target = scaler_y.fit_transform(target.values.reshape(-1, 1))
            print("Scaled data and target.")

            self.progress.emit(85)

            # データの分割（訓練用とテスト用）
            train_size = int(len(scaled_data) * 0.8)
            X_train = scaled_data[:train_size]
            y_train = scaled_target[:train_size]
            X_test = scaled_data[train_size:]
            y_test = scaled_target[train_size:]
            print(f"Split data into train ({train_size}) and test ({len(scaled_data) - train_size}) sets.")

            # データを3次元配列に変換（LSTMの入力形式に合わせる）
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            print("Reshaped data for LSTM.")

            self.progress.emit(90)

            # モデル構築関数
            def create_model(trial):
                # ハイパーパラメータの選択
                n_units = trial.suggest_int('n_units', 50, 200)
                dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
                learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

                # 入力層
                inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

                # 双方向LSTM層
                lstm_out = Bidirectional(LSTM(n_units, return_sequences=True))(inputs)
                lstm_out = Dropout(dropout_rate)(lstm_out)

                # GlobalAveragePooling1D
                flatten = GlobalAveragePooling1D()(lstm_out)

                # 出力層
                outputs = Dense(1)(flatten)

                # モデルの定義
                model = Model(inputs=inputs, outputs=outputs)

                # コンパイル
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss='mean_squared_error'
                )

                return model

            # ハイパーパラメータの最適化
            def objective(trial):
                model = create_model(trial)
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
                history = model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping],
                    verbose=0
                )
                # 最良の検証損失を取得
                val_loss = min(history.history['val_loss'])
                return val_loss

            self.progress.emit(95)
            print("Starting hyperparameter optimization with Optuna.")

            # Optunaによる最適化の実行
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=10)  # 試行回数を調整可能
            print("Hyperparameter optimization completed.")

            # 最適なハイパーパラメータでモデルを再構築
            best_trial = study.best_trial
            model = create_model(best_trial)
            print(f"Best trial parameters: {best_trial.params}")

            # モデルの訓練
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=0
            )
            print("Model training completed.")

            self.progress.emit(100)

            # 予測
            y_pred = model.predict(X_test)
            print("Prediction completed.")

            # スケールを元に戻す
            y_test_inv = scaler_y.inverse_transform(y_test)
            y_pred_inv = scaler_y.inverse_transform(y_pred)
            print("Inverse transformed predictions and true values.")

            # 評価指標の計算
            mse = mean_squared_error(y_test_inv, y_pred_inv)
            print(f"Mean Squared Error: {mse}")

            # シグナルの作成（後でバックテストで使用）
            signals = np.where(y_pred_inv > y_test_inv, 1, -1)  # 1:買い、-1:売り
            print("Created trading signals.")

            # リターンの計算（バックテスト用）
            returns = pd.Series(
                signals.flatten(),
                index=df.index[-len(y_test_inv):]
            ).shift(1) * (
                pd.Series(y_test_inv.flatten(), index=df.index[-len(y_test_inv):]).pct_change()
            )
            returns = returns.fillna(0)
            print("Calculated returns.")

            # シャープレシオの計算
            if returns.std() != 0:
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            print(f"Sharpe Ratio: {sharpe_ratio}")

            # 累積リターンの計算
            cumulative_returns = (1 + returns).cumprod()
            print("Calculated cumulative returns.")

            # 結果を辞書にまとめる
            results = {
                'mse': mse,
                'sharpe_ratio': sharpe_ratio,
                'y_test_inv': y_test_inv,
                'y_pred_inv': y_pred_inv,
                'cumulative_returns': cumulative_returns,
                'df': df
            }

            # シグナルの送信
            self.finished.emit(df, results)
            print("Data processing thread finished successfully.")
        except Exception as e:
            error_message = f"エラーが発生しました: {str(e)}"
            self.error.emit(error_message)
            print(error_message)

# PyQt5のメインウィンドウのクラス定義
class ForexPredictor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("為替レート予測アプリケーション")
        self.setGeometry(100, 100, 1400, 900)  # サイズを拡大

        # フォントの設定
        font = QtGui.QFont("Meiryo", 10)
        self.setFont(font)

        # スタイルシートの適用（黒基調）
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
            }
            QPushButton {
                background-color: #3c3f41;
                color: #ffffff;
                border: 1px solid #5c5c5c;
                padding: 5px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #484a4c;
            }
            QComboBox {
                background-color: #3c3f41;
                color: #ffffff;
                border: 1px solid #5c5c5c;
                padding: 5px;
                border-radius: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #3c3f41;
                selection-background-color: #5c5c5c;
            }
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #17becf;
                width: 20px;
            }
        """)

        # メインウィジェットとレイアウトの設定
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # 上部のウィジェット（通貨ペア選択とボタン）
        self.top_widget = QWidget(self)
        self.top_layout = QHBoxLayout(self.top_widget)

        # 通貨ペアの選択
        self.pair_label = QLabel("通貨ペアを選択:", self)
        self.pair_label.setFixedWidth(100)
        self.top_layout.addWidget(self.pair_label)

        self.pair_combo = QComboBox(self)
        self.pair_combo.addItems(["USD/JPY", "EUR/JPY", "GBP/JPY"])
        self.pair_combo.setFixedWidth(100)
        self.top_layout.addWidget(self.pair_combo)

        # 予測ボタン
        self.predict_button = QPushButton("予測を開始", self)
        self.predict_button.clicked.connect(self.start_prediction)
        self.top_layout.addWidget(self.predict_button)

        # リアルタイム価格表示ラベル
        self.price_label = QLabel("リアルタイム価格: 取得中...", self)
        self.price_label.setFixedWidth(200)
        self.top_layout.addWidget(self.price_label)

        # 進捗バーの追加
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)  # 初期は非表示
        self.top_layout.addWidget(self.progress_bar)

        # レイアウトに追加
        self.layout.addWidget(self.top_widget)

        # グラフ表示エリア（予測結果とテクニカル指標）
        self.figure = plt.figure(figsize=(12, 8))  # サイズを拡大
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # グラフ表示エリア（リアルタイムチャート）
        self.rt_figure = plt.figure(figsize=(12, 4))  # サイズを調整
        self.rt_canvas = FigureCanvas(self.rt_figure)
        self.layout.addWidget(self.rt_canvas)

        # タイマーの設定
        self.timer = QTimer()
        self.timer.setInterval(60000)  # 1分ごとに更新
        self.timer.timeout.connect(self.update_realtime_price)
        self.timer.start()

        # 選択されたティッカーシンボルを保存
        self.current_ticker = None

        # リアルタイムデータを保存
        self.rt_data = pd.DataFrame()

        # スレッド初期化
        self.thread = None

        # アプリケーション起動時にデフォルト通貨ペアのデータを表示
        self.pair_combo.setCurrentIndex(0)  # デフォルトをUSD/JPYに設定
        self.start_prediction()

    def update_realtime_price(self):
        if self.current_ticker:
            # 現在の時刻と1か月前の時刻を取得
            end = datetime.datetime.now()
            start = end - datetime.timedelta(days=30)

            # 過去1か月分のデータを5分足で取得
            try:
                print(f"Downloading real-time data for {self.current_ticker} from {start} to {end} with interval='5m'")
                rt_df = yf.download(
                    self.current_ticker,
                    start=start,
                    end=end,
                    interval='5m',
                    progress=False
                )
            except Exception as e:
                print(f"リアルタイムデータの取得中にエラーが発生しました: {e}")
                self.price_label.setText("リアルタイム価格: データ取得エラー")
                return

            # データ取得の確認
            if rt_df.empty:
                self.price_label.setText("リアルタイム価格: データ取得エラー")
                print("リアルタイムデータが空です。")
                return

            # 最新価格の取得
            latest_price = rt_df['Close'].iloc[-1]
            print(f"Latest price retrieved: {latest_price}")
            try:
                # latest_priceをスカラー値に変換
                latest_price = float(latest_price)
                self.price_label.setText(f"リアルタイム価格: {latest_price:.4f}")
            except (ValueError, TypeError) as e:
                print(f"最新価格のフォーマット変換中にエラーが発生しました: {e}")
                self.price_label.setText("リアルタイム価格: フォーマットエラー")
                return

            # リアルタイムデータを更新
            self.rt_data = rt_df

            # リアルタイムチャートの更新
            self.rt_figure.clear()
            ax_rt = self.rt_figure.add_subplot(1, 1, 1)
            ax_rt.plot(
                self.rt_data.index,
                self.rt_data['Close'],
                label='リアルタイム価格',
                color='#17becf'
            )
            ax_rt.set_title('過去1か月のリアルタイム価格（5分足）')
            ax_rt.legend()
            ax_rt.grid(True, which='both', linestyle='--', linewidth=0.5)
            self.rt_canvas.draw()
            print("リアルタイムチャートを更新しました。")

    def start_prediction(self):
        # 通貨ペアの取得
        pair = self.pair_combo.currentText()
        if pair == "USD/JPY":
            ticker = "JPY=X"
        elif pair == "EUR/JPY":
            ticker = "EURJPY=X"
        elif pair == "GBP/JPY":
            ticker = "GBPJPY=X"
        else:
            self.handle_error("選択された通貨ペアが無効です。")
            return

        # 現在のティッカーを更新
        self.current_ticker = ticker
        print(f"Selected ticker: {self.current_ticker}")

        # スレッドが既に動作している場合は終了
        if self.thread and self.thread.isRunning():
            print("既存のスレッドを終了させます。")
            self.thread.terminate()
            self.thread.wait()

        # 進捗バーの表示とリセット
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # スレッドの初期化
        self.thread = DataProcessingThread(ticker)
        self.thread.finished.connect(self.handle_prediction_result)
        self.thread.error.connect(self.handle_error)
        self.thread.progress.connect(self.update_progress)
        self.thread.start()
        print("データ処理スレッドを開始しました。")

    def handle_prediction_result(self, df, results):
        # グラフの描画
        self.figure.clear()

        # 実際の価格と予測された価格のプロット
        ax1 = self.figure.add_subplot(4, 1, 1)  # 4行1列に変更
        ax1.plot(
            df.index[-len(results['y_test_inv']):],
            results['y_test_inv'],
            label='実際の価格',
            color='#1f77b4'
        )
        ax1.plot(
            df.index[-len(results['y_pred_inv']):],
            results['y_pred_inv'],
            label='予測された価格',
            color='#ff7f0e'
        )
        ax1.set_title(f'{self.pair_combo.currentText()} 実際の価格と予測された価格')
        ax1.legend()
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        # テクニカル指標の表示（SMAを追加）
        ax1.plot(
            df.index[-len(results['y_test_inv']):],
            df['SMA_20'].tail(len(results['y_test_inv'])),
            label='SMA 20',
            color='#2ca02c',
            linestyle='--'
        )
        ax1.plot(
            df.index[-len(results['y_test_inv']):],
            df['SMA_50'].tail(len(results['y_test_inv'])),
            label='SMA 50',
            color='#d62728',
            linestyle='--'
        )
        ax1.legend()

        # MACDのプロット
        ax2 = self.figure.add_subplot(4, 1, 2)
        ax2.plot(
            df.index[-len(results['y_test_inv']):],
            df['MACD_12_26_9'].tail(len(results['y_test_inv'])),
            label='MACD',
            color='#ff7f0e'
        )
        ax2.plot(
            df.index[-len(results['y_test_inv']):],
            df['MACDh_12_26_9'].tail(len(results['y_test_inv'])),
            label='MACD Histogram',
            color='#2ca02c'
        )
        ax2.plot(
            df.index[-len(results['y_test_inv']):],
            df['MACDs_12_26_9'].tail(len(results['y_test_inv'])),
            label='MACD Signal',
            color='#d62728'
        )
        ax2.set_title('MACD 指標')
        ax2.legend()
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

        # ストキャスティクスのプロット
        ax3 = self.figure.add_subplot(4, 1, 3)
        ax3.plot(
            df.index[-len(results['y_test_inv']):],
            df['STOCHk_14_3_3'].tail(len(results['y_test_inv'])),
            label='Stochastic %K',
            color='#1f77b4'
        )
        ax3.plot(
            df.index[-len(results['y_test_inv']):],
            df['STOCHd_14_3_3'].tail(len(results['y_test_inv'])),
            label='Stochastic %D',
            color='#ff7f0e'
        )
        ax3.axhline(80, color='grey', linestyle='--', linewidth=0.5)
        ax3.axhline(20, color='grey', linestyle='--', linewidth=0.5)
        ax3.set_title('ストキャスティクス 指標')
        ax3.legend()
        ax3.grid(True, which='both', linestyle='--', linewidth=0.5)

        # ATRとボラティリティのプロット
        ax4 = self.figure.add_subplot(4, 1, 4)
        ax4.plot(
            df.index[-len(results['y_test_inv']):],
            df['ATR'].tail(len(results['y_test_inv'])),
            label='ATR',
            color='#2ca02c'
        )
        ax4.plot(
            df.index[-len(results['y_test_inv']):],
            df['Volatility'].tail(len(results['y_test_inv'])),
            label='Volatility',
            color='#d62728'
        )
        ax4.set_title('ATR と ボラティリティ')
        ax4.legend()
        ax4.grid(True, which='both', linestyle='--', linewidth=0.5)

        # バックテストの実装
        # シンプルな戦略：予測が実際の価格より上がると予測されたら買い、下がると予測されたら売り

        # シグナルの作成
        signals = np.where(results['y_pred_inv'] > results['y_test_inv'], 1, -1)  # 1:買い、-1:売り
        print("Created trading signals.")

        # リターンの計算
        returns = pd.Series(
            signals.flatten(),
            index=df.index[-len(results['y_test_inv']):]
        ).shift(1) * (
            pd.Series(results['y_test_inv'].flatten(), index=df.index[-len(results['y_test_inv']):]).pct_change()
        )
        returns = returns.fillna(0)
        print("Calculated returns.")

        # シャープレシオの計算
        sharpe_ratio = results['sharpe_ratio']
        print(f"Sharpe Ratio: {sharpe_ratio}")

        # 累積リターンの計算
        cumulative_returns = results['cumulative_returns']
        print("Calculated cumulative returns.")

        # バックテスト結果のプロット
        ax4.plot(
            cumulative_returns.index,
            cumulative_returns.values,
            label='戦略のリターン',
            color='#9467bd'
        )
        ax4.set_title('バックテスト結果')
        ax4.legend()
        ax4.grid(True, which='both', linestyle='--', linewidth=0.5)

        self.figure.tight_layout()
        self.canvas.draw()
        print("Prediction results plotted.")

        # リアルタイムチャートの更新
        self.update_realtime_price()

        # 進捗バーの非表示
        self.progress_bar.setVisible(False)

        # MSEとシャープレシオをメッセージボックスで表示
        QtWidgets.QMessageBox.information(
            self,
            "予測結果",
            f"平均二乗誤差 (MSE): {results['mse']:.4f}\nシャープレシオ: {sharpe_ratio:.4f}"
        )
        print("Displayed prediction results in message box.")

    def handle_error(self, message):
        # 進捗バーの非表示
        self.progress_bar.setVisible(False)

        QtWidgets.QMessageBox.warning(
            self,
            "エラー",
            message
        )
        print(f"Error occurred: {message}")

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        print(f"Progress updated to {value}%.")
        
# アプリケーションの実行
if __name__ == '__main__':
    app = QApplication(sys.argv)

    # アプリケーション全体のフォントを設定（黒基調に合わせてフォント色を調整）
    font = QtGui.QFont("Meiryo", 10)
    app.setFont(font)

    window = ForexPredictor()
    window.show()
    sys.exit(app.exec_())
