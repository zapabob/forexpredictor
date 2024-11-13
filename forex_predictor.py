# forex_predictor.py

# 必要なライブラリのインポート
import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
import optuna
import pytz
import logging
import joblib  # スケーラーの保存と読み込み用

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QComboBox, QVBoxLayout, QWidget,
    QHBoxLayout, QMessageBox, QProgressBar
)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# グラフ描画のバックエンド設定
plt.switch_backend('Qt5Agg')

# Matplotlibで日本語フォントを設定（フォールバックを追加）
import matplotlib
import matplotlib.font_manager as fm

def set_matplotlib_font():
    """
    Matplotlibで使用するフォントを設定します。
    'Meiryo'が利用可能な場合はそれを使用し、そうでない場合はデフォルトのフォントを使用します。
    """
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    if 'Meiryo' in available_fonts:
        matplotlib.rcParams['font.family'] = 'Meiryo'
        logging.info("フォントをMeiryoに設定しました。")
    else:
        # 日本語フォントが利用できない場合、デフォルトフォントにフォールバック
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = ['Arial']
        logging.warning("Meiryoフォントが見つかりませんでした。デフォルトフォントにフォールバックしました。")

set_matplotlib_font()

# ログの設定
logging.basicConfig(
    filename='forex_predictor.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)

class DataProcessingThread(QThread):
    """
    データ処理およびモデル訓練用のスレッドクラス。
    """
    # シグナルの定義
    finished = pyqtSignal(pd.DataFrame, dict)  # データフレームと結果辞書を送信
    error = pyqtSignal(str)                     # エラーメッセージを送信
    progress = pyqtSignal(int, str)            # 進捗値とメッセージを送信

    def __init__(self, ticker, total_trials=20, optimize=False):
        """
        コンストラクタ。

        :param ticker: 通貨ペアのティッカーシンボル
        :param total_trials: Optunaの試行数
        :param optimize: ハイパーパラメータ最適化を実行するかどうか
        """
        super().__init__()
        self.ticker = ticker
        self.total_trials = total_trials
        self.optimize = optimize
        self._is_running = True  # スレッドの実行状態を管理
        self.current_trial = 0   # Optuna試行のカウンターを初期化

    def run(self):
        """
        スレッドが開始されたときに実行されるメソッド。
        """
        try:
            self.progress.emit(0, "開始")
            logging.info("DataProcessingThread started.")

            # モデルとスケーラーのファイルパス
            model_path = f"{self.ticker}_model.keras"  # 修正: .h5から.kerasへ
            scaler_x_path = f"{self.ticker}_scaler_x.joblib"
            scaler_y_path = f"{self.ticker}_scaler_y.joblib"

            # モデルとスケーラーの読み込みまたは新規作成
            if self.optimize or not os.path.exists(model_path):
                # データの取得
                df = self.fetch_data()
                self.progress.emit(10, "データ取得完了")

                # データクリーンアップ
                df = self.clean_data(df)
                self.progress.emit(20, "データクリーンアップ完了")

                # テクニカル指標の計算
                df = self.calculate_technical_indicators(df)
                self.progress.emit(40, "テクニカル指標の計算完了")

                # 特徴量エンジニアリング
                df = self.feature_engineering(df)
                self.progress.emit(60, "特徴量エンジニアリング完了")

                # 特徴量とターゲットの準備
                X_train, X_test, y_train, y_test, scaler_X, scaler_y = self.prepare_features(df, scaler_x_path, scaler_y_path)
                self.progress.emit(75, "データの準備完了")

                # モデルの構築および訓練
                if self.optimize:
                    study = optuna.create_study(direction='minimize')
                    study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_test, y_test), 
                                  n_trials=self.total_trials, callbacks=[self.optuna_callback])
                    
                    if not self._is_running:
                        logging.info("スレッド停止要求により最適化を中断しました。")
                        return

                    best_trial = study.best_trial
                    model = self.create_model(best_trial, X_train.shape[1], X_train.shape[2])
                    logging.info(f"最適なハイパーパラメータ: {best_trial.params}")

                    # モデルの再訓練
                    model = self.train_model(model, X_train, y_train, X_test, y_test)
                    model.save(model_path)  # 修正: .h5から.kerasへ
                    logging.info(f"モデルを保存しました: {model_path}")
                else:
                    if os.path.exists(model_path):
                        model = load_model(model_path)
                        logging.info(f"保存されたモデルを読み込みました: {model_path}")
                    else:
                        model = self.create_model(None, X_train.shape[1], X_train.shape[2])
                        model = self.train_model(model, X_train, y_train, X_test, y_test)
                        model.save(model_path)  # 修正: .h5から.kerasへ
                        logging.info(f"モデルを保存しました: {model_path}")

                # 予測と評価
                y_pred = model.predict(X_test)
                y_test_inv, y_pred_inv = self.inverse_scale(y_test, y_pred, scaler_y_path)
                mse, r2 = self.evaluate_performance(y_test_inv, y_pred_inv)
                self.progress.emit(90, "予測と評価完了")

                # バックテスト
                results = self.backtest(df, y_test_inv, y_pred_inv)

                # 結果を辞書にまとめる
                results.update({
                    'mse': mse,
                    'r2': r2,
                })

                # シグナルの送信
                self.finished.emit(df, results)
                logging.info("DataProcessingThread finished successfully.")

            else:
                # モデルが存在し、最適化が不要な場合
                df = self.fetch_data()
                df = self.clean_data(df)
                df = self.calculate_technical_indicators(df)
                df = self.feature_engineering(df)
                X_train, X_test, y_train, y_test, scaler_X, scaler_y = self.prepare_features(df, scaler_x_path, scaler_y_path)
                model = load_model(model_path)
                y_pred = model.predict(X_test)
                y_test_inv, y_pred_inv = self.inverse_scale(y_test, y_pred, scaler_y_path)
                mse, r2 = self.evaluate_performance(y_test_inv, y_pred_inv)
                results = self.backtest(df, y_test_inv, y_pred_inv)
                results.update({
                    'mse': mse,
                    'r2': r2,
                })
                self.finished.emit(df, results)
                logging.info("DataProcessingThread finished successfully (model loaded).")

        except Exception as e:
            if self._is_running:
                error_message = f"エラーが発生しました: {str(e)}"
                self.error.emit(error_message)
                logging.error(error_message)

    def fetch_data(self):
        """
        データを取得するメソッド。

        :return: ダウンロードされたデータフレーム
        """
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=730)  # 過去2年間
        logging.info(f"{self.ticker} のデータを {start} から {end} まで取得します。")
        df = yf.download(
            self.ticker,
            start=start,
            end=end,
            interval='1h',
            progress=False
        )
        if df.empty:
            raise ValueError("データの取得に失敗しました。期間を調整してください。")
        logging.info("データのダウンロードが成功しました。")
        return df

    def clean_data(self, df):
        """
        データのクリーンアップを行うメソッド。

        :param df: 元のデータフレーム
        :return: クリーンアップ後のデータフレーム
        """
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
        logging.info("カラム名をクリーンアップしました。")

        # 必要なカラムが存在するか確認
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"以下のカラムがデータに存在しません: {missing_columns}")

        # 欠損値の処理
        df.dropna(inplace=True)
        logging.info("欠損値を削除しました。")

        return df

    def calculate_technical_indicators(self, df):
        """
        テクニカル指標を計算するメソッド。

        :param df: クリーンアップ済みのデータフレーム
        :return: テクニカル指標を追加したデータフレーム
        """
        logging.info("テクニカル指標の計算を開始します。")

        # テクニカル指標の計算
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['RSI_14'] = ta.rsi(df['Close'], length=14)
        bollinger = ta.bbands(df['Close'], length=20, std=2)
        df = pd.concat([df, bollinger], axis=1)
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        df = pd.concat([df, adx], axis=1)
        logging.info("SMA、RSI、ボリンジャーバンド、ADX指標を計算しました。")

        # MACDの計算と結合
        macd = ta.macd(df['Close'])
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_hist'] = macd['MACDh_12_26_9']
        df['MACD_signal'] = macd['MACDs_12_26_9']
        logging.info("MACD指標を計算し、データフレームに統合しました。")

        # ストキャスティクスの計算と結合
        stoch = ta.stoch(df['High'], df['Low'], df['Close'])
        df['STOCH_k'] = stoch['STOCHk_14_3_3']
        df['STOCH_d'] = stoch['STOCHd_14_3_3']
        logging.info("ストキャスティクス指標を計算し、データフレームに統合しました。")

        # ATRとボラティリティの計算
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])
        df['Volatility'] = df['Close'].rolling(window=20).std()
        logging.info("ATRとボラティリティを計算しました。")

        return df

    def feature_engineering(self, df):
        """
        特徴量エンジニアリングを行うメソッド。

        :param df: テクニカル指標を追加したデータフレーム
        :return: 特徴量エンジニアリング後のデータフレーム
        """
        logging.info("特徴量エンジニアリングを開始します。")

        # 移動平均の差分
        df['SMA_diff'] = df['SMA_20'] - df['SMA_50']
        logging.info("移動平均の差分を計算しました。")

        # ラグ特徴量の作成（過去3時間分）
        for lag in range(1, 4):
            df[f'Lag_{lag}'] = df['Close'].shift(lag)
        logging.info("ラグ特徴量を作成しました。")

        # 欠損値の再処理
        df.dropna(inplace=True)
        logging.info("特徴量エンジニアリング後の欠損値を削除しました。")

        return df

    def prepare_features(self, df, scaler_x_path, scaler_y_path):
        """
        特徴量とターゲットを準備し、スケーリングを行うメソッド。

        :param df: 特徴量エンジニアリング後のデータフレーム
        :param scaler_x_path: スケーラーXの保存パス
        :param scaler_y_path: スケーラーYの保存パス
        :return: 訓練データ、テストデータ、訓練ターゲット、テストターゲット、スケーラーX、スケーラーY
        """
        logging.info("特徴量とターゲットの準備を開始します。")

        # 特徴量とターゲットの設定
        features = [
            'Close', 'SMA_20', 'SMA_50',
            'RSI_14',
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
            'ADX_14',
            'MACD', 'MACD_hist', 'MACD_signal',
            'STOCH_k', 'STOCH_d',
            'ATR', 'Volatility', 'SMA_diff',
            'Lag_1', 'Lag_2', 'Lag_3'
        ]

        # 必要な特徴量がすべて存在するか確認
        missing_features = [feature for feature in features if feature not in df.columns]
        if missing_features:
            raise ValueError(f"以下の特徴量がデータフレームに存在しません:\n{missing_features}")

        # 特徴量とターゲットの抽出
        data = df[features]
        target = df['Close'].shift(-1)  # 1時間先の価格を予測

        # 最後の行の欠損値を削除
        data = data.iloc[:-1]
        target = target.iloc[:-1]
        logging.info("特徴量とターゲットを準備しました。")

        # データの標準化
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        scaled_data = scaler_X.fit_transform(data)
        scaled_target = scaler_y.fit_transform(target.values.reshape(-1, 1))
        logging.info("データとターゲットを標準化しました。")

        # スケーラーの保存
        joblib.dump(scaler_X, scaler_x_path)
        joblib.dump(scaler_y, scaler_y_path)
        logging.info("スケーラーを保存しました。")

        # データの分割
        train_size = int(len(scaled_data) * 0.8)
        X_train = scaled_data[:train_size]
        y_train = scaled_target[:train_size]
        X_test = scaled_data[train_size:]
        y_test = scaled_target[train_size:]
        logging.info(f"データを訓練用 ({train_size}) とテスト用 ({len(scaled_data) - train_size}) に分割しました。")

        # データを3次元配列に変換（LSTMの入力形式に合わせる）
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        logging.info("LSTM用にデータを3次元配列に変換しました。")

        return X_train, X_test, y_train, y_test, scaler_X, scaler_y

    def create_model(self, trial, input_shape_1, input_shape_2):
        """
        モデルを構築するメソッド。

        :param trial: Optunaの試行オブジェクト（Noneの場合、デフォルト値を使用）
        :param input_shape_1: LSTMの入力次元1
        :param input_shape_2: LSTMの入力次元2
        :return: 構築されたKerasモデル
        """
        # デフォルトのハイパーパラメータ
        default_params = {
            'n_units': 100,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'l1_reg': 1e-5,
            'l2_reg': 1e-5
        }

        if trial:
            # ハイパーパラメータの選択
            n_units = trial.suggest_int('n_units', 50, 200)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
            l1_reg = trial.suggest_float('l1_reg', 1e-6, 1e-3, log=True)
            l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
        else:
            # デフォルト値を使用
            n_units = default_params['n_units']
            dropout_rate = default_params['dropout_rate']
            learning_rate = default_params['learning_rate']
            l1_reg = default_params['l1_reg']
            l2_reg = default_params['l2_reg']

        logging.info(f"モデル構築パラメータ: units={n_units}, dropout={dropout_rate}, lr={learning_rate}, l1={l1_reg}, l2={l2_reg}")

        # モデルの構築
        inputs = Input(shape=(input_shape_1, input_shape_2))
        lstm_out = Bidirectional(
            LSTM(
                n_units,
                return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                bias_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
            )
        )(inputs)
        lstm_out = Dropout(dropout_rate)(lstm_out)
        flatten = GlobalAveragePooling1D()(lstm_out)
        outputs = Dense(
            1,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
            bias_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
        )(flatten)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mean_squared_error'
        )
        logging.info("モデルを構築し、コンパイルしました。")
        return model

    def objective(self, trial, X_train, y_train, X_test, y_test):
        """
        Optunaの目的関数。

        :param trial: Optunaの試行オブジェクト
        :param X_train: 訓練データ
        :param y_train: 訓練ターゲット
        :param X_test: テストデータ
        :param y_test: テストターゲット
        :return: 検証損失
        """
        model = self.create_model(trial, X_train.shape[1], X_train.shape[2])
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
        val_loss = min(history.history['val_loss'])
        logging.info(f"試行完了: val_loss={val_loss}")
        return val_loss

    def optuna_callback(self, study, trial):
        """
        Optunaのカスタムコールバック関数。

        :param study: Optunaのスタディオブジェクト
        :param trial: Optunaの試行オブジェクト
        """
        if not self._is_running:
            study.stop()
            return
        self.current_trial += 1
        progress_percentage = int((self.current_trial / self.total_trials) * 100)
        progress_message = f"Optuna試行 {self.current_trial}/{self.total_trials} 完了"
        self.progress.emit(progress_percentage, progress_message)
        logging.info(f"Optuna試行 {self.current_trial}/{self.total_trials} 完了。進捗: {progress_percentage}%")

    def train_model(self, model, X_train, y_train, X_test, y_test):
        """
        モデルを訓練するメソッド。

        :param model: Kerasモデル
        :param X_train: 訓練データ
        :param y_train: 訓練ターゲット
        :param X_test: テストデータ
        :param y_test: テストターゲット
        :return: 訓練済みモデル
        """
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
        logging.info("モデルの訓練が完了しました。")
        return model

    def inverse_scale(self, y_test, y_pred, scaler_y_path):
        """
        スケールを元に戻すメソッド。

        :param y_test: スケーリングされたテストターゲット
        :param y_pred: スケーリングされた予測値
        :param scaler_y_path: スケーラーYのパス
        :return: 元のスケールに戻したテストターゲットと予測値
        """
        scaler_y = joblib.load(scaler_y_path)
        y_test_inv = scaler_y.inverse_transform(y_test)
        y_pred_inv = scaler_y.inverse_transform(y_pred)
        logging.info("予測結果と実際の値を元のスケールに戻しました。")
        return y_test_inv, y_pred_inv

    def evaluate_performance(self, y_test_inv, y_pred_inv):
        """
        モデルの評価指標を計算するメソッド。

        :param y_test_inv: 元のスケールに戻したテストターゲット
        :param y_pred_inv: 元のスケールに戻した予測値
        :return: 平均二乗誤差 (MSE) と決定係数 (R²)
        """
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)
        logging.info(f"平均二乗誤差 (MSE): {mse}")
        logging.info(f"決定係数 (R²): {r2}")
        return mse, r2

    def backtest(self, df, y_test_inv, y_pred_inv):
        """
        バックテストを実行するメソッド。

        :param df: 完全なデータフレーム
        :param y_test_inv: 元のスケールに戻したテストターゲット
        :param y_pred_inv: 元のスケールに戻した予測値
        :return: バックテストの結果辞書
        """
        # シグナルの作成（後でバックテストで使用）
        signals = np.where(y_pred_inv > y_test_inv, 1, -1)  # 1:買い、-1:売り
        logging.info("トレーディングシグナルを作成しました。")

        # リターンの計算（バックテスト用）
        returns = pd.Series(
            signals.flatten(),
            index=df.index[-len(y_test_inv):]
        ).shift(1) * (
            pd.Series(y_test_inv.flatten(), index=df.index[-len(y_test_inv):]).pct_change()
        )
        returns = returns.fillna(0)
        logging.info("リターンを計算しました。")

        # シャープレシオの計算
        if returns.std() != 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        logging.info(f"シャープレシオ: {sharpe_ratio}")

        # カルマーレシオの計算
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        annual_return = returns.mean() * 252  # 252は1年間の取引日数
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        logging.info(f"カルマーレシオ: {calmar_ratio}")

        # 結果を辞書にまとめる
        results = {
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'cumulative_returns': cumulative_returns,
            'y_test_inv': y_test_inv,
            'y_pred_inv': y_pred_inv,
        }

        return results

class ForexPredictor(QMainWindow):
    """
    メインウィジェットのクラス。
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("為替レート予測アプリケーション")
        self.setGeometry(100, 100, 1600, 1200)  # ウィンドウサイズ

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
        self.top_layout = QVBoxLayout(self.top_widget)  # QVBoxLayoutに変更

        # 上部の水平レイアウト（通貨ペア選択、ボタン、価格表示）
        self.top_row_layout = QHBoxLayout()

        # 通貨ペアの選択
        self.pair_label = QLabel("通貨ペアを選択:", self)
        self.pair_label.setFixedWidth(100)
        self.top_row_layout.addWidget(self.pair_label)

        self.pair_combo = QComboBox(self)
        self.pair_combo.addItems(["USD/JPY", "EUR/JPY", "GBP/JPY"])
        self.pair_combo.setFixedWidth(100)
        self.top_row_layout.addWidget(self.pair_combo)

        # 予測ボタン
        self.predict_button = QPushButton("予測を開始", self)
        self.predict_button.clicked.connect(self.start_prediction)
        self.top_row_layout.addWidget(self.predict_button)

        # ハイパーパラメータ最適化ボタン
        self.optimize_button = QPushButton("ハイパーパラメータ最適化", self)
        self.optimize_button.clicked.connect(self.start_optimization)
        self.top_row_layout.addWidget(self.optimize_button)

        # リアルタイム価格表示ラベル
        self.price_label = QLabel("リアルタイム価格: 取得中...", self)
        self.price_label.setFixedWidth(250)
        self.top_row_layout.addWidget(self.price_label)

        # 時刻表示ラベルの追加
        self.time_label = QLabel("現在の時刻 (JST): --:--:--", self)
        self.time_label.setFixedWidth(200)
        self.time_label.setTextFormat(QtCore.Qt.RichText)  # HTML形式を明示
        self.top_row_layout.addWidget(self.time_label)

        # ストレッチを追加して、ウィジェット間のスペースを調整
        self.top_row_layout.addStretch()

        # 上部レイアウトに追加
        self.top_layout.addLayout(self.top_row_layout)

        # 下部の水平レイアウト（プログレスバーとステータス表示）
        self.bottom_row_layout = QHBoxLayout()

        # プログレスバーの追加
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)  # 初期は非表示
        self.progress_bar.setFixedHeight(20)  # 高さを調整
        self.bottom_row_layout.addWidget(self.progress_bar)

        # 進捗ステータス表示ラベルの追加
        self.progress_label = QLabel("", self)
        self.progress_label.setFixedWidth(200)
        self.bottom_row_layout.addWidget(self.progress_label)

        # ストレッチを追加して、ウィジェット間のスペースを調整
        self.bottom_row_layout.addStretch()

        # 下部レイアウトに追加
        self.top_layout.addLayout(self.bottom_row_layout)

        # レイアウトに追加
        self.layout.addWidget(self.top_widget)

        # グラフ表示エリア（予測結果とテクニカル指標）
        self.figure = plt.figure(figsize=(14, 10))  # サイズを調整（4行2列に対応）
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # グラフ表示エリア（リアルタイムチャート）
        self.rt_figure = plt.figure(figsize=(14, 5))  # サイズを調整
        self.rt_canvas = FigureCanvas(self.rt_figure)
        self.layout.addWidget(self.rt_canvas)

        # タイマーの設定（JSTを更新）
        self.timer = QTimer()
        self.timer.setInterval(1000)  # 1秒ごとに更新
        self.timer.timeout.connect(self.update_current_time)  # JST更新関数に接続
        self.timer.start()

        # タイマーの設定（リアルタイムデータの更新は1分ごと）
        self.rt_timer = QTimer()
        self.rt_timer.setInterval(60000)  # 1分ごとに更新
        self.rt_timer.timeout.connect(self.update_realtime_price)
        self.rt_timer.start()

        # 選択されたティッカーシンボルを保存
        self.current_ticker = None

        # リアルタイムデータを保存
        self.rt_data = pd.DataFrame()

        # スレッド初期化
        self.thread = None
        self.optimize_thread = None

        # アプリケーション起動時にデフォルト通貨ペアのデータを表示
        self.pair_combo.setCurrentIndex(0)  # デフォルトをUSD/JPYに設定
        self.start_prediction()

    def update_current_time(self):
        """
        現在の時刻をJSTで更新するメソッド。
        """
        try:
            logging.debug("update_current_time メソッドが呼び出されました。")
            # 現在のUTC時刻をタイムゾーン付きで取得
            utc_now = datetime.datetime.now(pytz.utc)

            # UTCをJSTに変換
            jst = pytz.timezone('Asia/Tokyo')
            jst_now = utc_now.astimezone(jst)

            # 日付と時刻をフォーマット
            formatted_date = jst_now.strftime('%y%m%d')  # YYMMDD形式
            formatted_time = jst_now.strftime('%H:%M:%S')  # HH:MM:SS形式

            # HTMLを使用して時刻部分を青色に設定
            time_html = f"<span style='color: blue;'>{formatted_time}</span>"

            # ラベルに設定
            self.time_label.setText(f"現在の時刻 (JST): {formatted_date} {time_html}")
            logging.debug(f"現在の時刻を更新: {formatted_date} {formatted_time}")
        except Exception as e:
            logging.error(f"時刻更新中にエラーが発生しました: {e}")
            self.time_label.setText("現在の時刻 (JST): エラー")

    def update_realtime_price(self):
        """
        リアルタイム価格を更新するメソッド。
        """
        if self.current_ticker:
            try:
                logging.info(f"{self.current_ticker} のリアルタイムデータを取得中。")
                # 現在の時刻と1日前の時刻を取得
                jst = pytz.timezone('Asia/Tokyo')
                end = datetime.datetime.now(jst)
                start = end - datetime.timedelta(days=1)  # 過去1日分に変更

                # 過去1日分のデータを5分足で取得
                rt_df = yf.download(
                    self.current_ticker,
                    start=start,
                    end=end,
                    interval='5m',
                    progress=False
                )

                # データ取得の確認
                if rt_df.empty:
                    self.price_label.setText("リアルタイム価格: データ取得エラー")
                    logging.warning("リアルタイムデータが空です。")
                    return

                # タイムスタンプをJSTに変換
                if rt_df.index.tz is None:
                    rt_df.index = rt_df.index.tz_localize('UTC').tz_convert('Asia/Tokyo')
                    logging.info("DatetimeIndexにUTCのタイムゾーンをローカライズし、JSTに変換しました。")
                else:
                    rt_df.index = rt_df.index.tz_convert('Asia/Tokyo')
                    logging.info("既存のタイムゾーン情報をJSTに変換しました。")

                # 最新価格の取得
                latest_price = float(rt_df['Close'].iloc[-1])
                self.price_label.setText(f"リアルタイム価格: {latest_price:.4f}")
                logging.info(f"最新価格を取得: {latest_price}")

                # リアルタイムデータを更新
                self.rt_data = rt_df

                # リアルタイムチャートの更新（メインスレッドで行う）
                self.update_realtime_chart()
                logging.info("リアルタイムチャートを更新しました。")
            except Exception as e:
                logging.error(f"リアルタイムデータの更新中にエラーが発生しました: {e}")
                self.price_label.setText("リアルタイム価格: データ取得エラー")

    def update_realtime_chart(self):
        """
        リアルタイムチャートを更新するメソッド。
        """
        try:
            self.rt_figure.clear()
            ax_rt = self.rt_figure.add_subplot(1, 1, 1)
            ax_rt.plot(
                self.rt_data.index,
                self.rt_data['Close'],
                label='リアルタイム価格',
                color='#17becf'
            )
            ax_rt.set_title('過去1日のリアルタイム価格（5分足）')
            ax_rt.legend()
            ax_rt.grid(True, linestyle='--', linewidth=0.5)
            self.rt_canvas.draw()
        except Exception as e:
            logging.error(f"リアルタイムチャートの更新中にエラーが発生しました: {e}")

    def start_prediction(self):
        """
        予測を開始するメソッド。
        """
        try:
            # 通貨ペアの取得
            pair = self.pair_combo.currentText()
            ticker_map = {"USD/JPY": "JPY=X", "EUR/JPY": "EURJPY=X", "GBP/JPY": "GBPJPY=X"}
            ticker = ticker_map.get(pair, None)

            if not ticker:
                self.handle_error("選択された通貨ペアが無効です。")
                return

            self.current_ticker = ticker
            logging.info(f"選択されたティッカー: {self.current_ticker}")

            # 予測ボタンと最適化ボタンを無効化
            self.predict_button.setEnabled(False)
            self.optimize_button.setEnabled(False)

            # 進捗バーとラベルの表示とリセット
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_label.setText("開始")

            # スレッドの初期化（初期トレーニング）
            self.thread = DataProcessingThread(ticker, total_trials=20, optimize=False)
            self.thread.finished.connect(self.handle_prediction_result)
            self.thread.error.connect(self.handle_error)
            self.thread.progress.connect(self.update_progress)
            self.thread.start()
            logging.info("データ処理スレッドを開始しました。")
        except Exception as e:
            logging.error(f"start_predictionでエラーが発生しました: {e}")
            self.handle_error(f"予測開始中にエラーが発生しました: {e}")

    def start_optimization(self):
        """
        ハイパーパラメータ最適化を開始するメソッド。
        """
        try:
            if not self.current_ticker:
                self.handle_error("まず予測を実行してください。")
                return

            # 予測ボタンと最適化ボタンを無効化
            self.predict_button.setEnabled(False)
            self.optimize_button.setEnabled(False)

            # 進捗バーとラベルの表示とリセット
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_label.setText("ハイパーパラメータ最適化開始")

            # 最適化用のスレッドを初期化
            self.optimize_thread = DataProcessingThread(self.current_ticker, total_trials=20, optimize=True)
            self.optimize_thread.finished.connect(self.handle_prediction_result)
            self.optimize_thread.error.connect(self.handle_error)
            self.optimize_thread.progress.connect(self.update_progress)
            self.optimize_thread.start()
            logging.info("ハイパーパラメータ最適化スレッドを開始しました。")
        except Exception as e:
            logging.error(f"start_optimizationでエラーが発生しました: {e}")
            self.handle_error(f"ハイパーパラメータ最適化開始中にエラーが発生しました: {e}")

    def handle_prediction_result(self, df, results):
        """
        予測結果を受け取り、グラフを描画し、評価指標を表示するメソッド。

        :param df: データフレーム
        :param results: 結果辞書
        """
        try:
            # グラフの描画（メインスレッドで行う）
            self.figure.clear()

            # 4行2列のサブプロットグリッドを設定
            ax1 = self.figure.add_subplot(4, 2, 1)
            ax2 = self.figure.add_subplot(4, 2, 2)
            ax3 = self.figure.add_subplot(4, 2, 3)
            ax4 = self.figure.add_subplot(4, 2, 4)
            ax5 = self.figure.add_subplot(4, 2, 5)
            ax6 = self.figure.add_subplot(4, 2, 6)
            ax7 = self.figure.add_subplot(4, 2, 7)
            ax8 = self.figure.add_subplot(4, 2, 8)

            # 実際の価格と予測された価格のプロット
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
            ax1.grid(True, linestyle='--', linewidth=0.5)

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

            # RSIのプロット
            ax2.plot(
                df.index[-len(results['y_test_inv']):],
                df['RSI_14'].tail(len(results['y_test_inv'])),
                label='RSI 14',
                color='#1f77b4'
            )
            ax2.axhline(70, color='grey', linestyle='--', linewidth=0.5)
            ax2.axhline(30, color='grey', linestyle='--', linewidth=0.5)
            ax2.set_title('RSI 指標')
            ax2.legend()
            ax2.grid(True, linestyle='--', linewidth=0.5)

            # ボリンジャーバンドのプロット
            ax3.plot(
                df.index[-len(results['y_test_inv']):],
                df['BBL_20_2.0'].tail(len(results['y_test_inv'])),
                label='Lower Bollinger Band',
                color='#2ca02c',
                linestyle='--'
            )
            ax3.plot(
                df.index[-len(results['y_test_inv']):],
                df['BBM_20_2.0'].tail(len(results['y_test_inv'])),
                label='Middle Bollinger Band',
                color='#1f77b4',
                linestyle='--'
            )
            ax3.plot(
                df.index[-len(results['y_test_inv']):],
                df['BBU_20_2.0'].tail(len(results['y_test_inv'])),
                label='Upper Bollinger Band',
                color='#d62728',
                linestyle='--'
            )
            ax3.plot(
                df.index[-len(results['y_test_inv']):],
                df['Close'].tail(len(results['y_test_inv'])),
                label='Close Price',
                color='#17becf'
            )
            ax3.set_title('ボリンジャーバンド 指標')
            ax3.legend()
            ax3.grid(True, linestyle='--', linewidth=0.5)

            # ADXのプロット
            ax4.plot(
                df.index[-len(results['y_test_inv']):],
                df['ADX_14'].tail(len(results['y_test_inv'])),
                label='ADX 14',
                color='#ff7f0e'
            )
            ax4.axhline(25, color='grey', linestyle='--', linewidth=0.5)
            ax4.set_title('ADX 指標')
            ax4.legend()
            ax4.grid(True, linestyle='--', linewidth=0.5)

            # MACDのプロット
            ax5.plot(
                df.index[-len(results['y_test_inv']):],
                df['MACD'].tail(len(results['y_test_inv'])),
                label='MACD',
                color='#ff7f0e'
            )
            ax5.plot(
                df.index[-len(results['y_test_inv']):],
                df['MACD_hist'].tail(len(results['y_test_inv'])),
                label='MACD Histogram',
                color='#2ca02c'
            )
            ax5.plot(
                df.index[-len(results['y_test_inv']):],
                df['MACD_signal'].tail(len(results['y_test_inv'])),
                label='MACD Signal',
                color='#d62728'
            )
            ax5.set_title('MACD 指標')
            ax5.legend()
            ax5.grid(True, linestyle='--', linewidth=0.5)

            # ストキャスティクスのプロット
            ax6.plot(
                df.index[-len(results['y_test_inv']):],
                df['STOCH_k'].tail(len(results['y_test_inv'])),
                label='Stochastic %K',
                color='#1f77b4'
            )
            ax6.plot(
                df.index[-len(results['y_test_inv']):],
                df['STOCH_d'].tail(len(results['y_test_inv'])),
                label='Stochastic %D',
                color='#ff7f0e'
            )
            ax6.axhline(80, color='grey', linestyle='--', linewidth=0.5)
            ax6.axhline(20, color='grey', linestyle='--', linewidth=0.5)
            ax6.set_title('ストキャスティクス 指標')
            ax6.legend()
            ax6.grid(True, linestyle='--', linewidth=0.5)

            # ATRとボラティリティのプロット
            ax7.plot(
                df.index[-len(results['y_test_inv']):],
                df['ATR'].tail(len(results['y_test_inv'])),
                label='ATR',
                color='#2ca02c'
            )
            ax7.plot(
                df.index[-len(results['y_test_inv']):],
                df['Volatility'].tail(len(results['y_test_inv'])),
                label='Volatility',
                color='#d62728'
            )
            ax7.set_title('ATR と ボラティリティ')
            ax7.legend()
            ax7.grid(True, linestyle='--', linewidth=0.5)

            # バックテスト結果のプロット
            ax8.plot(
                results['cumulative_returns'].index,
                results['cumulative_returns'].values,
                label='戦略のリターン',
                color='#9467bd'
            )
            ax8.set_title('バックテスト結果')
            ax8.legend()
            ax8.grid(True, linestyle='--', linewidth=0.5)

            self.figure.tight_layout()
            self.canvas.draw()
            logging.info("予測結果をプロットしました。")

            # リアルタイムチャートの更新
            self.update_realtime_price()

            # 進捗バーとラベルの非表示
            self.progress_bar.setVisible(False)
            self.progress_label.setText("完了")

            # 評価指標をメッセージボックスで表示
            QtWidgets.QMessageBox.information(
                self,
                "予測結果",
                f"平均二乗誤差 (MSE): {results['mse']:.4f}\n"
                f"決定係数 (R²): {results['r2']:.4f}\n"
                f"シャープレシオ: {results['sharpe_ratio']:.4f}\n"
                f"カルマーレシオ: {results['calmar_ratio']:.4f}"
            )
            logging.info("予測結果をメッセージボックスで表示しました。")

            # 予測ボタンと最適化ボタンを再度有効化
            self.predict_button.setEnabled(True)
            self.optimize_button.setEnabled(True)
        except Exception as e:
            logging.error(f"予測結果のハンドリング中にエラーが発生しました: {e}")
            self.handle_error(f"予測結果の処理中にエラーが発生しました: {e}")

    def handle_error(self, message):
        """
        エラーメッセージを表示し、UIを適切に更新するメソッド。

        :param message: エラーメッセージ
        """
        try:
            # 進捗バーとラベルの非表示
            self.progress_bar.setVisible(False)
            self.progress_label.setText("エラー")

            # メッセージボックスでエラーを表示
            QMessageBox.warning(
                self,
                "エラー",
                message
            )
            logging.warning(f"エラーが発生しました: {message}")

            # 予測ボタンと最適化ボタンを再度有効化
            self.predict_button.setEnabled(True)
            self.optimize_button.setEnabled(True)
        except Exception as e:
            logging.error(f"エラーハンドリング中にさらにエラーが発生しました: {e}")

    def update_progress(self, value, message):
        """
        プログレスバーと進捗ラベルを更新するメソッド。

        :param value: 進捗値（0〜100）
        :param message: 進捗メッセージ
        """
        try:
            self.progress_bar.setValue(value)
            self.progress_label.setText(message)
            logging.debug(f"進捗バーを {value}% に更新しました。メッセージ: {message}")
        except Exception as e:
            logging.error(f"進捗バーの更新中にエラーが発生しました: {e}")

    def closeEvent(self, event):
        """
        アプリケーション終了時に呼び出されるメソッド。
        実行中のスレッドを停止させる。

        :param event: QCloseEvent
        """
        try:
            # スレッドが動作中の場合は停止を試みる
            if self.thread and self.thread.isRunning():
                logging.info("データ処理スレッドを停止します。")
                self.thread._is_running = False
                self.thread.wait()
            if self.optimize_thread and self.optimize_thread.isRunning():
                logging.info("ハイパーパラメータ最適化スレッドを停止します。")
                self.optimize_thread._is_running = False
                self.optimize_thread.wait()
            event.accept()
            logging.info("アプリケーションを終了しました。")
        except Exception as e:
            logging.error(f"closeEvent中にエラーが発生しました: {e}")
            event.accept()

def main():
    """
    アプリケーションのエントリーポイント。
    """
    try:
        app = QApplication(sys.argv)

        # アプリケーション全体のフォントを設定
        font = QtGui.QFont("Meiryo", 10)
        app.setFont(font)

        window = ForexPredictor()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        logging.critical(f"アプリケーション起動中に致命的なエラーが発生しました: {e}")

if __name__ == '__main__':
    main()

