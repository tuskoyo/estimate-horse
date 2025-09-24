import math
import numpy as np
import pandas as pd

def cal_angle(point1, point2, point3):
    # 各点を numpy array に変換
    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)
    
    # ベクトルを計算
    vector1 = p1 - p2  # point2 から point1 へのベクトル
    vector2 = p3 - p2  # point2 から point3 へのベクトル
    
    # ベクトルの長さを計算
    length1 = np.linalg.norm(vector1)
    length2 = np.linalg.norm(vector2)
    
    # 内積を計算
    dot_product = np.dot(vector1, vector2)
    
    # cosθ を計算
    cos_theta = dot_product / (length1 * length2)
    
    # 数値誤差対策（-1 ≤ cos θ ≤ 1 の範囲に収める）
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # 角度を計算（ラジアンから度数法に変換）
    angle = math.degrees(math.acos(cos_theta))
    
    return angle

def calculate_angles_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    
    angles = []
    for index, row in df.iterrows():
        point1 = [row['x1'], row['y1']]
        point2 = [row['x2'], row['y2']]
        point3 = [row['x3'], row['y3']]
        
        angle = cal_angle(point1, point2, point3)
        angles.append(angle)
    
    return angles

# 使用例
if __name__ == "__main__":
    csv_file = 'keypoints.csv'  # CSVファイルのパスを指定
    angles = calculate_angles_from_csv(csv_file)
    
    for i, angle in enumerate(angles):
        print(f"行 {i+1} の角度: {angle:.2f}度")