import os

import cv2
import pandas as pd

from gpat.utils.files import FileName
from gpat.utils.skeleton_keypoints import keypoints_list


def gpat2fpat(
    input_path: str,
) -> None:
    data_path = os.path.join(input_path, FileName.position_data)
    video_path = os.path.join(input_path, FileName.output_video)
    old_data_path = data_path + ".old"
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    df = pd.read_csv(data_path)
    columns = ['frame'] + [f"{kpt}_{xyz}" for kpt in keypoints_list for xyz in ['x', 'y', 'z']]
    new_df = pd.DataFrame(columns=columns)
    
    for col in new_df.columns:
        if not col.endswith('z'):
            if col == 'frame':
                new_df[col] = df[col].copy()
            elif col.endswith('x'):
                new_df[col] = df[col].copy() / width
            elif col.endswith('y'):
                new_df[col] = df[col].copy() / height
        else:
            new_df[col] = df[col.replace('z', 'x')].copy()
    
    os.rename(data_path, old_data_path)
    new_df.to_csv(data_path, index=False)