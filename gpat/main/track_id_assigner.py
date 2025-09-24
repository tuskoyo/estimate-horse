import os

import numpy as np
import pandas as pd


class TrackIdAssiner:
    def __init__(self, track_data_path: str):
        self.track_data_path = track_data_path
        self.track_data = pd.read_csv(track_data_path)
        self.unique_track_ids = self.track_data['track_id'].unique()
        self.track_id = 1
    
    def select_max_movement(self):
        max_movement = 0
        
        for track_id in self.unique_track_ids:
            track_data = self.track_data[self.track_data['track_id'] == track_id]
            
            x1_min = track_data['x1'].min()
            y1_min = track_data['y1'].min()
            x2_max = track_data['x2'].max()
            y2_max = track_data['y2'].max()
            
            movement = np.sqrt((x2_max - x1_min) ** 2 + (y2_max - y1_min) ** 2)
            
            if movement > max_movement:
                max_movement = movement
                self.track_id = track_id
        
        return self.track_id


if __name__ == '__main__':
    track_data_path = '/mnt/d/sasaki_2024093/data/center_3/tracking_data.csv'
    track_id_assigner = TrackIdAssiner(track_data_path)
    track_id = track_id_assigner.select_max_movement()
    print(f'Track ID: {track_id}')