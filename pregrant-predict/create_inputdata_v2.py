import os
import glob
import numpy as np
import pandas as pd
import argparse

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
    入力CSVファイルからAnimalPoseのキーポイントデータを読み込み、追加特徴量を計算して出力CSVに保存する。
    入力:
      - input_filepath: 生データCSVファイルのパス
    出力:
      - output_filepath: 加工済みCSVファイルとして保存
    計算する追加特徴量:
      ・左前肢肘角度（withers, left_front_elbow, left_front_paw）
      ・右前肢肘角度（withers, right_front_elbow, right_front_paw）
      ・左後肢膝角度（left_back_elbow, left_back_knee, left_back_paw）
      ・右後肢膝角度（right_back_elbow, right_back_knee, right_back_paw）
      ・各角度のフレーム間差分
      ・withers の位置変化（dx, dy, 移動距離）
      ・各特徴量のZスコア正規化（例：withers_movement_z）
    """
    df = pd.read_csv(input_filepath)

    # 各フレームごとに関節角度を計算
    left_front_angles = []
    right_front_angles = []
    left_back_angles = []
    right_back_angles = []
    for idx, row in df.iterrows():
        # 左前肢肘角度
        A = (row['withers_x'], row['withers_y'])
        B = (row['left_front_elbow_x'], row['left_front_elbow_y'])
        C = (row['left_front_paw_x'], row['left_front_paw_y'])
        lf_angle = compute_angle(A, B, C)
        left_front_angles.append(lf_angle)
        # 右前肢肘角度
        A = (row['withers_x'], row['withers_y'])
        B = (row['right_front_elbow_x'], row['right_front_elbow_y'])
        C = (row['right_front_paw_x'], row['right_front_paw_y'])
        rf_angle = compute_angle(A, B, C)
        right_front_angles.append(rf_angle)
        # 左後肢膝角度
        A = (row['left_back_elbow_x'], row['left_back_elbow_y'])
        B = (row['left_back_knee_x'], row['left_back_knee_y'])
        C = (row['left_back_paw_x'], row['left_back_paw_y'])
        lb_angle = compute_angle(A, B, C)
        left_back_angles.append(lb_angle)
        # 右後肢膝角度
        A = (row['right_back_elbow_x'], row['right_back_elbow_y'])
        B = (row['right_back_knee_x'], row['right_back_knee_y'])
        C = (row['right_back_paw_x'], row['right_back_paw_y'])
        rb_angle = compute_angle(A, B, C)
        right_back_angles.append(rb_angle)

    df['left_front_elbow_angle'] = left_front_angles
    df['right_front_elbow_angle'] = right_front_angles
    df['left_back_knee_angle'] = left_back_angles
    df['right_back_knee_angle'] = right_back_angles

    # フレーム間の角度変化
    df['left_front_elbow_angle_diff'] = df['left_front_elbow_angle'].diff()
    df['right_front_elbow_angle_diff'] = df['right_front_elbow_angle'].diff()
    df['left_back_knee_angle_diff'] = df['left_back_knee_angle'].diff()
    df['right_back_knee_angle_diff'] = df['right_back_knee_angle'].diff()

    # withers の位置変化
    df['withers_dx'] = df['withers_x'].diff()
    df['withers_dy'] = df['withers_y'].diff()
    df['withers_movement'] = np.sqrt(df['withers_dx']**2 + df['withers_dy']**2)

    # 正規化（例としてwithers_movementのZスコア）
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
    入力ディレクトリ内のすべてのCSVファイルを処理し、出力ディレクトリに同じファイル名で保存する。
    入力:
      - input_dir: 生データCSVファイルが格納されているディレクトリ
      - output_dir: 加工後CSVを保存するディレクトリ
    出力:
      - 各ファイルごとに加工済みCSVが保存される
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        output_filepath = os.path.join(output_dir, filename)
        process_csv_file(csv_file, output_filepath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AnimalPoseのキーポイントデータから追加特徴量を計算します。")
    parser.add_argument('input_dir', help="生データCSVファイルが格納されているディレクトリ")
    parser.add_argument('output_dir', help="加工後CSVを保存するディレクトリ")
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir)
