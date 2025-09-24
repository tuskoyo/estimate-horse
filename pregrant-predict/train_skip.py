import os
# GPUメモリ断片化対策
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

import glob
import time
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, roc_auc_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import japanize_matplotlib

# ===============================
# 1. 特徴量加工モジュール
# ===============================
def compute_angle(A, B, C):
    """
    3点 A, B, C (2次元座標) を受け取り、B を頂点とする角度（度）を計算する。
    入力: A, B, C は (x, y) のタプルまたは numpy 配列。
    出力: 角度（度）
    """
    BA = np.array(A) - np.array(B)
    BC = np.array(C) - np.array(B)
    dot_product = np.dot(BA, BC)
    norm_BA = np.linalg.norm(BA)
    norm_BC = np.linalg.norm(BC)
    if norm_BA == 0 or norm_BC == 0:
        return np.nan
    cos_angle = np.clip(dot_product / (norm_BA * norm_BC), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def process_csv_file(input_filepath, output_filepath):
    """
    生データCSVからAnimalPoseのキーポイント座標を読み込み、各フレームごとの
    関節角度、角度差分、withers位置変化などの追加特徴量を計算し、
    加工済みCSVとして保存する。
    入力:
      - input_filepath: 生データCSVファイルのパス
    出力:
      - output_filepath: 加工済みCSVファイルのパス（元のキーポイント＋追加特徴量＋ 'label' 列）
    """
    df = pd.read_csv(input_filepath)
    # 各フレームごとの関節角度計算
    left_front_angles = []
    right_front_angles = []
    left_back_angles = []
    right_back_angles = []
    for idx, row in df.iterrows():
        # 左前肢肘角度: withers, left_front_elbow, left_front_paw
        A = (row['withers_x'], row['withers_y'])
        B = (row['left_front_elbow_x'], row['left_front_elbow_y'])
        C = (row['left_front_paw_x'], row['left_front_paw_y'])
        lf_angle = compute_angle(A, B, C)
        left_front_angles.append(lf_angle)
        # 右前肢肘角度: withers, right_front_elbow, right_front_paw
        A = (row['withers_x'], row['withers_y'])
        B = (row['right_front_elbow_x'], row['right_front_elbow_y'])
        C = (row['right_front_paw_x'], row['right_front_paw_y'])
        rf_angle = compute_angle(A, B, C)
        right_front_angles.append(rf_angle)
        # 左後肢膝角度: left_back_elbow, left_back_knee, left_back_paw
        A = (row['left_back_elbow_x'], row['left_back_elbow_y'])
        B = (row['left_back_knee_x'], row['left_back_knee_y'])
        C = (row['left_back_paw_x'], row['left_back_paw_y'])
        lb_angle = compute_angle(A, B, C)
        left_back_angles.append(lb_angle)
        # 右後肢膝角度: right_back_elbow, right_back_knee, right_back_paw
        A = (row['right_back_elbow_x'], row['right_back_elbow_y'])
        B = (row['right_back_knee_x'], row['right_back_knee_y'])
        C = (row['right_back_paw_x'], row['right_back_paw_y'])
        rb_angle = compute_angle(A, B, C)
        right_back_angles.append(rb_angle)
    df['left_front_elbow_angle'] = left_front_angles
    df['right_front_elbow_angle'] = right_front_angles
    df['left_back_knee_angle'] = left_back_angles
    df['right_back_knee_angle'] = right_back_angles

    # 各角度のフレーム間差分
    df['left_front_elbow_angle_diff'] = df['left_front_elbow_angle'].diff()
    df['right_front_elbow_angle_diff'] = df['right_front_elbow_angle'].diff()
    df['left_back_knee_angle_diff'] = df['left_back_knee_angle'].diff()
    df['right_back_knee_angle_diff'] = df['right_back_knee_angle'].diff()

    # withers の位置変化
    df['withers_dx'] = df['withers_x'].diff()
    df['withers_dy'] = df['withers_y'].diff()
    df['withers_movement'] = np.sqrt(df['withers_dx']**2 + df['withers_dy']**2)

    # 正規化（例: withers_movementのZスコア）
    for col in ['left_front_elbow_angle', 'right_front_elbow_angle',
                'left_back_knee_angle', 'right_back_knee_angle',
                'withers_movement']:
        mean_val = df[col].mean()
        std_val = df[col].std()
        df[col + '_z'] = (df[col] - mean_val) / std_val if std_val != 0 else 0

    df.to_csv(output_filepath, index=False)
    print(f"Processed: {os.path.basename(input_filepath)}")

def process_directory(input_dir, output_dir):
    """
    入力ディレクトリ内のすべての生データCSVファイルを加工済みCSVに変換し、
    指定した出力ディレクトリに保存する。
    入力:
      - input_dir: 生データCSVファイルのディレクトリ
      - output_dir: 加工済みCSVファイルの保存先ディレクトリ
    出力:
      - 各CSVファイルごとに加工済みファイルが出力される。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        output_filepath = os.path.join(output_dir, filename)
        process_csv_file(csv_file, output_filepath)

# ===============================
# 2-1. シーケンス短縮モジュール（ダウンサンプリングとウィンドウ分割）
# ===============================
def downsample_sequence(sequence, factor):
    """
    時系列データ（numpy配列, shape=(タイムステップ数, 特徴量)）を指定した係数でダウンサンプリングする。
    例: factor=2なら元の1/2のフレーム数になる。
    """
    return sequence[::factor]

def split_sequence(sequence, window_size, stride):
    """
    長い時系列データを、ウィンドウサイズとストライドで分割する。
    入力:
      - sequence: numpy配列, shape=(タイムステップ数, 特徴量)
      - window_size: 各ウィンドウのフレーム数
      - stride: ウィンドウをずらすフレーム数
    出力:
      - 各ウィンドウごとのリスト
    """
    windows = []
    for start in range(0, len(sequence) - window_size + 1, stride):
        windows.append(sequence[start:start+window_size])
    return windows

# ===============================
# 2-2. 学習モジュール（PyTorchによるLOOCVおよび最終学習）
# ===============================
class BidirectionalLSTM(nn.Module):
    """
    双方向LSTMモデル
    - 入力: (batch_size, sequence_length, input_dim)
    - 出力: 各時間ステップごとに0〜1の確率を出力（シグモイド適用）
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(BidirectionalLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        output = self.linear(lstm_out)
        output = self.sigmoid(output)
        return output

def load_processed_data_with_temporal_labels(data_dir, downsample_factor=1):
    """
    加工済みCSVファイルを読み込み、各ファイルの時系列データと各時間ステップのラベルを取得する。
    ダウンサンプリングも適用可能。
    入力:
      - data_dir: 加工済みCSVファイルが格納されたディレクトリ
      - downsample_factor: ダウンサンプリング係数（1なら間引きなし）
    出力:
      - X: 特徴量の時系列データ（リスト）
      - y: 各時間ステップのラベル（リスト）
      - file_names: ファイル名のリスト
    """
    file_paths = glob.glob(os.path.join(data_dir, '*.csv'))
    X, y, file_names = [], [], []
    for fp in file_paths:
        try:
            df = pd.read_csv(fp)
            df = df.dropna(how='all')
            if 'label' not in df.columns:
                print(f"警告: {fp} に 'label' 列がありません。スキップします。")
                continue
            time_col = df.columns[0]
            features_cols = [col for col in df.columns if col != 'label' and col != time_col]
            valid_rows = df[features_cols].notna().all(axis=1)
            if valid_rows.sum() == 0:
                print(f"警告: {fp} に有効なデータ行がありません。スキップします。")
                continue
            df = df[valid_rows]
            labels = df['label'].values.astype(np.float32)
            features = df[features_cols].values.astype(np.float32)
            # 0〜1スケーリング
            features_min = np.min(features, axis=0, keepdims=True)
            features_max = np.max(features, axis=0, keepdims=True)
            features_range = np.maximum(features_max - features_min, 1e-5)
            features_normalized = (features - features_min) / features_range
            if downsample_factor > 1:
                features_normalized = downsample_sequence(features_normalized, downsample_factor)
                labels = downsample_sequence(labels, downsample_factor)
            X.append(features_normalized)
            y.append(labels)
            file_names.append(os.path.basename(fp))
            print(f"ファイル {os.path.basename(fp)} 読み込み: 形状 {features_normalized.shape}, ラベル形状 {labels.shape}")
        except Exception as e:
            print(f"ファイル {fp} の読み込みエラー: {e}")
            continue
    if len(X) == 0:
        raise ValueError("有効なデータファイルが見つかりませんでした。")
    return X, y, file_names

def pad_sequences_with_labels(X, y):
    """
    異なる長さの時系列データとラベルを、同じ長さにパディングする。
    入力:
      - X: 特徴量の時系列データ（リスト）
      - y: 各時間ステップのラベル（リスト）
    出力:
      - X_padded: numpy配列、shape=(num_samples, max_length, input_dim)
      - y_padded: numpy配列、shape=(num_samples, max_length)
      - mask: numpy配列、実データは1、パディングは0
      - max_length: 各サンプル中の最大タイムステップ数
    """
    max_len = max(len(seq) for seq in X)
    input_dim = X[0].shape[1]
    X_padded, y_padded, mask = [], [], []
    for i in range(len(X)):
        curr_len = len(X[i])
        pad_len = max_len - curr_len
        curr_mask = np.ones(curr_len, dtype=np.float32)
        if pad_len > 0:
            padded_features = np.vstack([X[i], np.zeros((pad_len, input_dim))])
            padded_labels = np.concatenate([y[i], np.zeros(pad_len)])
            curr_mask = np.concatenate([curr_mask, np.zeros(pad_len)])
        else:
            padded_features = X[i]
            padded_labels = y[i]
        X_padded.append(padded_features)
        y_padded.append(padded_labels)
        mask.append(curr_mask)
    return np.array(X_padded), np.array(y_padded), np.array(mask), max_len

def train_model_loocv(X, y, file_names, processed_dir, output_dir, device='cuda', epochs=50):
    """
    LOOCVを用いて、各サンプルの時系列データに対するモデルの評価を行い、
    各foldの予測結果CSVを output_dir に出力する。
    入力:
      - X: 特徴量の時系列データ（リスト）
      - y: 各時間ステップのラベル（リスト）
      - file_names: 各サンプルのファイル名（リスト）
      - processed_dir: 加工済みCSVファイルのディレクトリ（再読み込み用）
      - output_dir: LOOCVの各fold予測結果CSVの保存先ディレクトリ
      - device: 'cuda' または 'cpu'
      - epochs: 訓練エポック数
    出力:
      - LOOCVの各fold評価結果をコンソールに出力
      - 各foldの予測結果CSVを保存
    """
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDAが利用できません。CPUを使用します。")
        device = 'cpu'
    device = torch.device(device)
    print(f"使用デバイス: {device}")

    X_padded, y_padded, masks, max_timesteps = pad_sequences_with_labels(X, y)
    num_samples, max_timesteps, num_features = X_padded.shape
    print(f"データ形状: {X_padded.shape}, ラベル形状: {y_padded.shape}")

    total_labels = y_padded.sum()
    total_elements = np.sum(masks)
    pos_weight = (total_elements - total_labels) / (total_labels + 1e-6)
    print(f"ポジティブクラスの重み: {pos_weight:.2f}")

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    loo = LeaveOneOut()
    all_y_true, all_y_pred, all_y_prob = [], [], []
    fold = 1
    for train_index, test_index in loo.split(X_padded):
        X_train, X_test = X_padded[train_index], X_padded[test_index]
        y_train, y_test = y_padded[train_index], y_padded[test_index]
        mask_train, mask_test = masks[train_index], masks[test_index]
        test_file = file_names[test_index[0]]
        print(f"\nFold {fold}/{num_samples} - テストファイル: {test_file}")

        # バッチサイズは1に設定
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(-1).to(device)
        mask_train_tensor = torch.FloatTensor(mask_train).unsqueeze(-1).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)

        model = BidirectionalLSTM(input_dim=num_features, hidden_dim=64, output_dim=1).to(device)
        criterion = nn.BCELoss(reduction='none')
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_loss = float('inf')
        patience = 5
        patience_counter = 0

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            weights = torch.ones_like(y_train_tensor)
            weights[y_train_tensor > 0.5] = pos_weight
            masked_loss = loss * mask_train_tensor * weights
            loss_value = masked_loss.sum() / mask_train_tensor.sum()
            loss_value.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"  エポック {epoch+1}/{epochs}, 損失: {loss_value.item():.4f}")
            if loss_value.item() < best_loss:
                best_loss = loss_value.item()
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"  エポック {epoch+1}: 早期停止")
                break

        torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            y_pred_tensor = model(X_test_tensor)
            y_pred_numpy = y_pred_tensor.cpu().numpy()[0, :, 0]
            y_pred = (y_pred_numpy > 0.5).astype(int)

        valid_steps = masks[test_index[0]] > 0
        y_true_valid = y_test[0][valid_steps]
        y_pred_valid = y_pred[valid_steps]
        y_prob_valid = y_pred_numpy[valid_steps]

        all_y_true.extend(y_true_valid)
        all_y_pred.extend(y_pred_valid)
        all_y_prob.extend(y_prob_valid)

        print(f"  真のイベント数: {np.sum(y_true_valid==1)}, 予測イベント数: {np.sum(y_pred_valid==1)}")

        # 各foldのテストサンプルの予測結果CSVを出力（output_dirが指定されていれば）
        if output_dir:
            # 入力ディレクトリ内の同名ファイルを再読み込み（加工済みCSV）
            test_fp = os.path.join(processed_dir, test_file)
            try:
                df_test = pd.read_csv(test_fp)
                # 予測結果の長さとdf_testの行数が一致しているかチェック
                if len(df_test) >= len(y_pred):
                    # 先頭len(y_pred)行に予測結果を追加
                    df_test.loc[:len(y_pred)-1, 'pred_label'] = y_pred
                else:
                    df_test['pred_label'] = y_pred[:len(df_test)]
                out_fp = os.path.join(output_dir, test_file)
                df_test.to_csv(out_fp, index=False)
                print(f"  Fold {fold} の予測結果CSVを保存: {out_fp}")
            except Exception as e:
                print(f"  Fold {fold} の予測結果CSV保存エラー: {e}")
        fold += 1

    print("\n===== LOOCV 全体の評価 =====")
    print(classification_report(all_y_true, all_y_pred))
    f1 = f1_score(all_y_true, all_y_pred)
    print(f"F1スコア: {f1:.4f}")
    try:
        auc = roc_auc_score(all_y_true, all_y_prob)
        print(f"AUC: {auc:.4f}")
    except Exception as e:
        print(f"AUC計算エラー: {e}")

def train_final_model(X, y, save_path='models_horse/pytorch_model.pt', device='cuda', epochs=50):
    """
    全データを用いて最終モデルを学習し、保存する。
    入力:
      - X: 特徴量の時系列データ（リスト）
      - y: 各時間ステップのラベル（リスト）
      - save_path: モデル保存先ファイルパス
      - device: 'cuda' または 'cpu'
      - epochs: 訓練エポック数
    出力:
      - 学習済みモデルを保存し、返却する。
    """
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDAが利用できません。CPUを使用します。")
        device = 'cpu'
    device = torch.device(device)

    X_padded, y_padded, masks, max_timesteps = pad_sequences_with_labels(X, y)
    num_samples, max_timesteps, num_features = X_padded.shape
    total_labels = y_padded.sum()
    total_elements = np.sum(masks)
    pos_weight = (total_elements - total_labels) / (total_labels + 1e-6)
    print(f"ポジティブクラスの重み: {pos_weight:.2f}")

    X_tensor = torch.FloatTensor(X_padded).to(device)
    y_tensor = torch.FloatTensor(y_padded).unsqueeze(-1).to(device)
    mask_tensor = torch.FloatTensor(masks).unsqueeze(-1).to(device)

    model = BidirectionalLSTM(input_dim=num_features, hidden_dim=64, output_dim=1).to(device)
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=0.0001)

    model.train()
    best_loss = float('inf')
    best_model_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        weights = torch.ones_like(y_tensor)
        weights[y_tensor > 0.5] = pos_weight
        masked_loss = loss * mask_tensor * weights
        loss_value = masked_loss.sum() / mask_tensor.sum()
        loss_value.backward()
        optimizer.step()
        scheduler.step(loss_value.item())
        print(f"エポック {epoch+1}/{epochs}, 損失: {loss_value.item():.4f}, 学習率: {optimizer.param_groups[0]['lr']:.6f}")
        if loss_value.item() < best_loss:
            best_loss = loss_value.item()
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  最良モデル更新（損失: {best_loss:.4f}）")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"エポック {epoch+1}: 早期停止")
            break

    model.load_state_dict(best_model_state)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_features': num_features,
        'hidden_dim': 64
    }, save_path)
    print(f"最終モデルを {save_path} に保存しました。")
    return model

# ===============================
# 3. リアルタイム予測モジュール
# ===============================
frame_buffer = []
WINDOW_SIZE = 300  # 例: 1fpsで2分間（300フレーム）
def process_new_frame(frame_features, model):
    """
    新たに1フレーム分の特徴量（numpy配列, shape=(特徴量次元,)）を受け取り、
    グローバルバッファに追加する。バッファ内に十分な実フレーム（WINDOW_SIZE分）
    が揃っていない場合は予測をスキップする。
    入力:
      - frame_features: 1フレーム分の特徴量（numpy配列）
      - model: 学習済みのLSTMモデル
    出力:
      - 十分なフレームがある場合は、ウィンドウ内のデータを用いて予測を行い、
        確信度と結果（警告 or 正常状態）をコンソールに出力する。
      - 十分なフレームがない場合は、予測をスキップする。
    """
    global frame_buffer
    frame_buffer.append(frame_features)

    # 十分な実フレームがなければ予測しない
    if len(frame_buffer) < WINDOW_SIZE:
        print(f"まだ十分なフレームがありません（{len(frame_buffer)}/{WINDOW_SIZE}）。予測をスキップします。")
        return

    # ウィンドウサイズ分の最新フレームを使って予測
    window_data = np.array(frame_buffer[-WINDOW_SIZE:])
    window_data = np.expand_dims(window_data, axis=0)  # (1, WINDOW_SIZE, 特徴量次元)
    input_tensor = torch.FloatTensor(window_data).to(next(model.parameters()).device)
    prediction = model(input_tensor).cpu().detach().numpy()[0, 0]

    print(f"リアルタイム予測 確信度: {prediction:.3f}")
    if prediction >= 0.5:
        print("警告: 出産直前の兆候が検出されました！")
    else:
        print("正常状態")

    # 次回に備え、古いフレームを削除
    frame_buffer.pop(0)

def real_time_prediction(model, simulated_frames):
    """
    シミュレーション用：リスト 'simulated_frames' の各フレーム分特徴量を順次処理し、
    リアルタイム予測を実施する。
    出力:
      - 予測結果をコンソールに出力する。
    """
    import time
    for frame_features in simulated_frames:
        process_new_frame(frame_features, model)
        time.sleep(1)

# ===============================
# 4. 可視化モジュール（オプション）
# ===============================
def visualize_model_predictions(model, X, y, file_names, device='cuda', output_dir='visualization'):
    """
    各サンプルの予測結果をグラフ化して保存する。
    入力:
      - model: 学習済みモデル
      - X: 時系列データ（リスト）
      - y: 各時間ステップのラベル（リスト）
      - file_names: ファイル名リスト
      - device: 'cuda' または 'cpu'
      - output_dir: 保存先ディレクトリ
    出力:
      - 予測結果のグラフを画像として出力先に保存
    """
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)
    X_padded, y_padded, masks, _ = pad_sequences_with_labels(X, y)
    model.eval()
    for i in range(len(X_padded)):
        X_tensor = torch.FloatTensor(X_padded[i:i+1]).to(device)
        with torch.no_grad():
            y_pred_tensor = model(X_tensor)
            y_pred = y_pred_tensor.cpu().numpy()[0, :, 0]
        valid_mask = masks[i] > 0
        valid_y_true = y_padded[i][valid_mask]
        valid_y_pred = y_pred[valid_mask]
        plt.figure(figsize=(12, 6))
        plt.plot(valid_y_true, 'g-', alpha=0.5, label='真のラベル')
        plt.plot(valid_y_pred, 'r-', alpha=0.5, label='予測確率')
        predicted_labels = (valid_y_pred > 0.5).astype(int)
        true_positive_indices = np.where((valid_y_true == 1) & (predicted_labels == 1))[0]
        false_negative_indices = np.where((valid_y_true == 1) & (predicted_labels == 0))[0]
        false_positive_indices = np.where((valid_y_true == 0) & (predicted_labels == 1))[0]
        plt.scatter(true_positive_indices, valid_y_true[true_positive_indices], color='green', marker='o', s=50, label='正検出')
        plt.scatter(false_negative_indices, valid_y_true[false_negative_indices], color='blue', marker='x', s=50, label='見逃し')
        plt.scatter(false_positive_indices, predicted_labels[false_positive_indices], color='red', marker='x', s=50, label='誤検出')
        plt.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
        plt.title(f'ファイル: {file_names[i]} - 陣痛行動予測')
        plt.xlabel('時間ステップ')
        plt.ylabel('陣痛行動確率')
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'prediction_{file_names[i]}.png'), dpi=300)
        plt.close()
    print(f"予測の可視化が {output_dir} に保存されました。")

# ===============================
# 5. 予測結果CSV出力モジュール（各LOOCVでの出力）
# ===============================
def predict_and_save_csv_fold(processed_dir, output_dir, file_name, y_pred):
    """
    指定されたprocessed_dir内のCSVファイルを再読み込みし、各時間ステップごとの予測結果（pred_label）を
    カラムとして追加して、output_dirに保存する。
    入力:
      - processed_dir: 加工済みCSVファイルのディレクトリ
      - output_dir: 予測結果付きCSVファイルの保存先ディレクトリ
      - file_name: 対象CSVのファイル名
      - y_pred: LOOCVで予測されたラベル配列（1次元、各時間ステップの予測）
    出力:
      - 対象ファイルに 'pred_label' カラムを追加してoutput_dirに保存する。
    """
    try:
        in_fp = os.path.join(processed_dir, file_name)
        df = pd.read_csv(in_fp)
        if len(df) >= len(y_pred):
            df.loc[:len(y_pred)-1, 'pred_label'] = y_pred
        else:
            df['pred_label'] = y_pred[:len(df)]
        out_fp = os.path.join(output_dir, file_name)
        df.to_csv(out_fp, index=False)
        print(f"  Fold予測CSVを保存: {out_fp}")
    except Exception as e:
        print(f"  予測CSV保存エラー ({file_name}): {e}")

# ===============================
# 6. 全体実行部
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorchを使用した馬の陣痛行動予測モデル（深層学習＋LOOCV予測結果CSV出力）")
    parser.add_argument("data_dir", help="加工済みCSVファイルが格納されているディレクトリ")
    parser.add_argument("--final_model", help="最終モデル保存ファイル名", default="models_horse/pytorch_model.pt")
    parser.add_argument("--device", choices=['cuda', 'cpu'], default='cuda', help="使用するデバイス")
    parser.add_argument("--epochs", type=int, default=30, help="訓練エポック数")
    parser.add_argument("--downsample", type=int, default=1, help="ダウンサンプリング係数（例: 2なら1/2に間引く）")
    parser.add_argument("--visualize", action="store_true", help="予測の可視化を行う")
    parser.add_argument("--visualize_dir", default="visualization", help="可視化結果の保存ディレクトリ")
    parser.add_argument("--output_dir", default="train_output", help="LOOCV各foldの予測結果付きCSVの保存先ディレクトリ")
    args = parser.parse_args()

    # (A) 加工済みCSVから、時系列データとラベルを読み込み（ダウンサンプリング適用）
    X, y, file_names = load_processed_data_with_temporal_labels(args.data_dir, downsample_factor=args.downsample)
    print(f"読み込まれたファイル数: {len(X)}")

    # (B) LOOCVによるモデル評価と、各foldごとに予測結果CSV出力（output_dirが指定されていれば）
    train_model_loocv(X, y, file_names, processed_dir=args.data_dir, output_dir=args.output_dir, device=args.device, epochs=args.epochs)

    # (C) 全データを用いた最終モデルの訓練・保存
    if args.final_model:
        final_model = train_final_model(X, y, args.final_model, device=args.device, epochs=args.epochs+20)
        # (D) 可視化オプションが有効なら、予測結果をグラフとして保存
        if args.visualize:
            visualize_model_predictions(final_model, X, y, file_names, device=args.device, output_dir=args.visualize_dir)
        # (E) 予測結果CSV出力機能（最終モデルを使って入力ディレクトリの全CSVに対して予測結果を出力）
        # こちらは任意で、output_dirが指定されていれば実行
        if args.output_dir:
            print("最終モデルを用いた全CSVファイルの予測結果出力を開始します。")
            # 各CSVファイルに対して予測結果を出力
            for file_name in file_names:
                try:
                    in_fp = os.path.join(args.data_dir, file_name)
                    df = pd.read_csv(in_fp)
                    time_col = df.columns[0]
                    features_cols = [col for col in df.columns if col != 'label' and col != time_col]
                    valid_rows = df[features_cols].notna().all(axis=1)
                    if valid_rows.sum() == 0:
                        continue
                    df = df[valid_rows]
                    features = df[features_cols].values.astype(np.float32)
                    # 0〜1スケーリング（学習時と同じ）
                    features_min = np.min(features, axis=0, keepdims=True)
                    features_max = np.max(features, axis=0, keepdims=True)
                    features_range = np.maximum(features_max - features_min, 1e-5)
                    features_normalized = (features - features_min) / features_range
                    if args.downsample > 1:
                        features_normalized = features_normalized[::args.downsample]

                    # フレーム数チェック：十分なフレーム数（WINDOW_SIZE以上）がない場合は予測をスキップ
                    if features_normalized.shape[0] < WINDOW_SIZE:
                        print(f"{file_name} のフレーム数が不足しています（{features_normalized.shape[0]}/{WINDOW_SIZE}）。予測をスキップします。")
                        continue

                    input_tensor = torch.FloatTensor(features_normalized).unsqueeze(0).to(args.device)
                    final_model.eval()
                    with torch.no_grad():
                        predictions = final_model(input_tensor).cpu().numpy()[0, :, 0]
                    pred_labels = (predictions > 0.5).astype(int)
                    df['pred_label'] = pred_labels
                    out_fp = os.path.join(args.output_dir, file_name)
                    df.to_csv(out_fp, index=False)
                    print(f"予測結果付きCSVを保存: {out_fp}")
                except Exception as e:
                    print(f"ファイル {file_name} の予測エラー: {e}")
                    continue
            print("全CSVファイルの予測結果出力が完了しました。")
