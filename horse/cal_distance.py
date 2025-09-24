import math
import numpy as np

def cal_distance(point1, point2):
    """
    2点間の距離を計算する関数

    Parameters:
    point1 (tuple): 最初の点のx, y座標 (x1, y1)
    point2 (tuple): 2番目の点のx, y座標 (x2, y2)

    Returns:
    float: 2点間の距離
    """
    # 各点を numpy array に変換
    p1 = np.array(point1)
    p2 = np.array(point2)
    
    # 距離を計算
    dis = math.sqrt(((p2 - p1)**2).sum())
    
    return dis

# 使用例
if __name__ == "__main__":
    point1 = (1, 2)
    point2 = (4, 6)
    distance = cal_distance(point1, point2)
    print(f"2点間の距離: {distance:.2f}")