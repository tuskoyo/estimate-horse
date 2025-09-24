import os
import glob
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_processed_data(data_dir):
    """
    加工済みCSVファイルをすべて読み込み、各ファイルを1サンプルの時系列データ（numpy配列）としてリストに格納する。
    入力:
      - data_dir: 加工済みCSVファイルが格納されているディレクトリ
    出力:
      - sequences: 各サンプルの時系列データ（リスト）
      - labels: 各サンプルに対応するラベル（numpy配列）
    """
    file_paths = glob.glob(os.path.join(data_dir, '*.csv'))
    sequences = []
    labels = []
    for fp in file_paths:
        df = pd.read_csv(fp, index_col=0)
        if 'label' not in df.columns:
            print(f"Warning: {fp} に 'label' 列がありません。")
            continue
        label = df['label'].iloc[0]
        df = df.drop(columns=['label'])
        sequences.append(df.values.astype(np.float32))
        labels.append(label)
    return sequences, np.array(labels)

def build_model(input_shape):
    """
    入力形状 (タイムステップ数, 特徴量次元) に対してLSTMモデルを構築する。
    入力:
      - input_shape: タイムステップ数と特徴量次元のタプル (例: (240, 44))
    出力:
      - コンパイル済みのLSTMモデル
    """
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model_loocv(X, y):
    """
    Leave-One-Out Cross-Validation (LOOCV) を用いてモデルの評価を行う。
    入力:
      - X: 各サンプルの時系列データ（リストまたは配列）
      - y: ラベル（numpy配列）
    出力:
      - LOOCVの評価結果（F1スコア、AUCなどをコンソール出力）
    """
    X_padded = pad_sequences(X, padding='post', dtype='float32')
    num_samples, max_timesteps, num_features = X_padded.shape
    print("学習データ形状:", X_padded.shape)

    loo = LeaveOneOut()
    y_true_all = []
    y_pred_all = []
    fold = 1
    for train_index, test_index in loo.split(X_padded):
        X_train, X_test = X_padded[train_index], X_padded[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = build_model((max_timesteps, num_features))
        model.fit(X_train, y_train,
                  epochs=50,
                  batch_size=4,
                  verbose=0,
                  validation_data=(X_test, y_test),
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
        y_pred_prob = model.predict(X_test)[0,0]
        y_pred = 1 if y_pred_prob >= 0.5 else 0
        y_true_all.append(y_test[0])
        y_pred_all.append(y_pred)
        print(f"Fold {fold} - 正解: {y_test[0]}, 予測: {y_pred} (確信度: {y_pred_prob:.3f})")
        fold += 1
    f1 = f1_score(y_true_all, y_pred_all)
    try:
        auc = roc_auc_score(y_true_all, y_pred_all)
    except Exception:
        auc = None
    print("LOOCV 結果: F1スコア =", f1, ", AUC =", auc)

def train_final_model(X, y, save_path='models_horse/trained_model.h5'):
    """
    全データを用いて最終モデルを学習し、保存する。
    入力:
      - X: 各サンプルの時系列データ（リストまたは配列）
      - y: ラベル（numpy配列）
      - save_path: 保存するモデルファイル名（例: 'trained_model.h5'）
    出力:
      - 学習済みの最終モデル（ファイルにも保存）
    """
    X_padded = pad_sequences(X, padding='post', dtype='float32')
    num_samples, max_timesteps, num_features = X_padded.shape
    model = build_model((max_timesteps, num_features))
    model.fit(X_padded, y, epochs=50, batch_size=4, verbose=1,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)])
    model.save(save_path)
    print(f"最終モデルを {save_path} に保存しました。")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="時系列データを用いたLSTMモデルの学習")
    parser.add_argument("data_dir", help="加工済みCSVファイルが格納されているディレクトリ")
    parser.add_argument("--final_model", help="最終モデルを保存する場合のファイル名")
    args = parser.parse_args()
    os.makedirs('models_horse', exist_ok=True)
    X, y = load_processed_data(args.data_dir)
    train_model_loocv(X, y)
    if args.final_model:
        train_final_model(X, y, args.final_model)
