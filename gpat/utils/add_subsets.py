import os

import cv2
import json
import pandas as pd
import shutil

from gpat.utils.files import FileName


def add_subsets(front_dir, side_dir):
    front_video_name = os.path.basename(front_dir) if not front_dir.endswith('/') else os.path.basename(front_dir[:-1])
    side_video_name = os.path.basename(side_dir) if not side_dir.endswith('/') else os.path.basename(side_dir[:-1])
    
    json_path = os.path.join(front_dir, FileName.px_subsets)
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    os.remove(os.path.join(front_dir, FileName.position_data))
    os.remove(os.path.join(side_dir, FileName.position_data))

    os.rename(os.path.join(front_dir, FileName.position_data+".old"), os.path.join(front_dir, FileName.position_data))
    os.rename(os.path.join(side_dir, FileName.position_data+".old"), os.path.join(side_dir, FileName.position_data))

    shutil.copy(os.path.join(front_dir, FileName.position_data), os.path.join(front_dir, FileName.position_data+".old"))
    shutil.copy(os.path.join(side_dir, FileName.position_data), os.path.join(side_dir, FileName.position_data+".old"))

    front_df = pd.read_csv(os.path.join(front_dir, FileName.position_data))
    side_df = pd.read_csv(os.path.join(side_dir, FileName.position_data))

    for i, v in enumerate(data["subsets"]):
        front_df[f"POINT{i+1}_x"] = v[front_video_name][0]
        front_df[f"POINT{i+1}_y"] = v[front_video_name][1]
        side_df[f"POINT{i+1}_x"] = v[side_video_name][0]
        side_df[f"POINT{i+1}_y"] = v[side_video_name][1]

    cap = cv2.VideoCapture(os.path.join(front_dir, FileName.output_video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    columns = [c[:-2] for c in front_df.columns if c.endswith('_x')]
    columns = ["frame"] + [f"{c}_{xyz}" for c in columns for xyz in ['x', 'y', 'z']]
    new_front_df = pd.DataFrame(columns=columns)
    new_side_df = pd.DataFrame(columns=columns)

    for col in new_front_df.columns:
        if not col.endswith('z'):
            if col == 'frame':
                new_front_df[col] = front_df[col].copy()
                new_side_df[col] = side_df[col].copy()
            elif col.endswith('x'):
                new_front_df[col] = front_df[col].copy() / width
                new_side_df[col] = side_df[col].copy() / width
            elif col.endswith('y'):
                new_front_df[col] = front_df[col].copy() / height
                new_side_df[col] = side_df[col].copy() / height
        else:
            new_front_df[col] = front_df[col.replace('z', 'x')].copy()
            new_side_df[col] = side_df[col.replace('z', 'x')].copy()
    
    new_front_df.to_csv(os.path.join(front_dir, FileName.position_data), index=False)
    new_side_df.to_csv(os.path.join(side_dir, FileName.position_data), index=False)

if __name__ == '__main__':
    front_dir = "/home/kitano/Soccer/data/front_2364_2808"
    side_dir = "/home/kitano/Soccer/data/side_2364_2808/"
    add_subsets(front_dir, side_dir)