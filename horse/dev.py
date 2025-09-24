import os
import sys
import warnings

warnings.filterwarnings('ignore')

import cv2
import mmcv
import mmengine
import numpy as np
import pandas as pd
import torch
import json
from gpat.utils.files import FileName
from gpat.utils.utils import calculate_iou, get_file_name
from mmdet.apis import inference_detector, init_detector

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

from mmpose.evaluation.functional import nms
from mmpose.utils import adapt_mmdet_pipeline

import matplotlib.pyplot as plt

video_path = "/home/osamu/gpat-horse/GPAT/horse.mp4"
frame_dir = "/home/osamu/horse_data/test"

video_name = get_file_name(video_path)
os.makedirs(frame_dir, exist_ok=True)


cap = cv2.VideoCapture(video_path)


det_model = "/home/osamu/gpat-horse/GPAT/models/rtmdet_l_8xb32-300e_coco.py"
det_checkpoint = "/home/osamu/gpat-horse/GPAT/models/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"
pose_model = "/home/osamu/gpat-horse/GPAT/models/td-hm_hrnet-w48_8xb64-210e_animalpose-256x256.py"
pose_checkpoint = "/home/osamu/gpat-horse/GPAT/models/hrnet_w48_animalpose_256x256-34644726_20210426.pth"

detector = init_detector(det_model, det_checkpoint, device="cuda")
detector.cfg = adapt_mmdet_pipeline(detector.cfg)

radius: int = 5
alpha: float = 0.8
thickness: int = 1

det_cat_id = 17
bbox_thr = 0.3
nms_thr = 0.5

pose_estimator = init_pose_estimator(pose_model, pose_checkpoint, device='cuda')
pose_estimator.cfg.visualizer.radius = radius
pose_estimator.cfg.visualizer.alpha = alpha
pose_estimator.cfg.visualizer.line_width = thickness




frame_idx = 0
with open('keypoints.json', 'r') as f:
    keypoints = json.load(f)    

position_df = pd.DataFrame(columns=['frame'] + [f"{kpt}_{xy}" for kpt in keypoints for xy in ["x", "y"]])

while True:
    ret, img = cap.read()
    frame_idx += 1
    print(f'Processing frame: {frame_idx}', end='\r')
    
    if not ret:
        break

    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()

    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == det_cat_id, pred_instance.scores > bbox_thr)]
    bboxes = bboxes[nms(bboxes, nms_thr), :4]

    pose_results = inference_topdown(pose_estimator, img, bboxes)

    #pose_resultの形式を確認
    position_data = pose_results[0].pred_instances.cpu().numpy().keypoints[0, :23].ravel().tolist()

    position_df.loc[frame_idx] = [frame_idx] + position_data

    



    data_samples = merge_data_samples(pose_results)

    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(pose_estimator.dataset_meta)
    # visualize the results
    visualizer.add_datasample(
        'result',
        img,
        data_sample=data_samples,
        draw_gt=False,
        )

    cv2.imwrite(
            os.path.join(frame_dir, f'{video_name}_{frame_idx}.jpg'),
            cv2.cvtColor(visualizer.get_image(), cv2.COLOR_RGB2BGR),
        )

position_df.to_csv(os.path.join(frame_dir,'position.csv'), index=False)



cap.release()
