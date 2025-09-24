import pandas as pd
import numpy as np
import re
import os

def extract_time_from_text(text):
    """
    ファイル名文字列から「HH_mm_ss.ss」の形式の時間を抽出する。
    例: "bigred-c100/2024-05-10/00_00_00.57.jpg" → "00_00_00.57"
    """
    pattern = r'(\d{2})_(\d{2})_(\d{2}\.\d+)'
    match = re.search(pattern, text)
    if match:
        hh = match.group(1)
        mm = match.group(2)
        ss = match.group(3)
        return f"{hh}_{mm}_{ss}"
    return None

def main(input_filename, output_filename):
    df = pd.read_csv(input_filename)
    # --- 時間情報の抽出 ---
    if 'filename' in df.columns:
        df['time'] = df['filename'].astype(str).apply(extract_time_from_text)
        if df['time'].isnull().any():
            print("一部の行で時間情報が抽出できませんでした。")
        df['time_for_delta'] = df['time'].str.replace("_", ":")
        df['time_sec'] = pd.to_timedelta(df['time_for_delta'], errors='coerce').dt.total_seconds()
        df = df.sort_values('time_sec').reset_index(drop=True)
    else:
        time_interval = 0.1
        df['time_sec'] = df.index * time_interval

    # --- キーポイント列の抽出 ---
    keypoint_x_cols = [col for col in df.columns if col.endswith('_x')]
    keypoints = [col[:-2] for col in keypoint_x_cols]

    # --- 各フレーム間の移動距離、共通の時間差（dt）、速度の計算 ---
    results = []
    for i in range(1, len(df)):
        dt = df.loc[i, 'time_sec'] - df.loc[i-1, 'time_sec']
        row_result = {"time_sec": df.loc[i, 'time_sec'], "dt": dt}
        for kp in keypoints:
            x_prev = df.loc[i-1, f"{kp}_x"]
            y_prev = df.loc[i-1, f"{kp}_y"]
            x_curr = df.loc[i, f"{kp}_x"]
            y_curr = df.loc[i, f"{kp}_y"]
            dist = np.sqrt((x_curr - x_prev)**2 + (y_curr - y_prev)**2)
            row_result[f"{kp}_distance"] = dist
            row_result[f"{kp}_velocity"] = dist / dt if dt != 0 else np.nan
        results.append(row_result)

    result_df = pd.DataFrame(results)

    # --- インデックスの設定 ---
    if 'filename' in df.columns:
        time_index = df.loc[1:, 'time'].values
        result_df.index = time_index
    else:
        result_df.index = result_df['time_sec']

    result_df.to_csv(output_filename)
    print(f"計算結果を {output_filename} として出力しました。")

if __name__ == "__main__":
    input_filename = "/mnt/d/horse/hdd1_unzip/pose-estimation/bigred-c100/2024-05-10/position.csv"
    output_filename = "/mnt/d/horse/hdd1_unzip/pose-estimation/bigred-c100/2024-05-10/position_move.csv"
    main(input_filename, output_filename)
