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
from gpat.utils.files import FileName
from gpat.utils.skeleton_keypoints import exp_keypoints_list
from gpat.utils.utils import calculate_iou, get_file_name
from mmdet.apis import inference_detector, init_detector

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline


def run(
    video_path: str,
    output_path: str,
    pose_model: str,
    pose_checkpoint: str,
    det_model: str,
    det_checkpoint: str,
    det_cat_id: int = 0,
    conf_rank: int = 1,
    bbox_thr: float = 0.3,
    nms_thr: float = 0.3,
    iou_thr: float = 0.3,
    kpt_thr : float = 0.3,
    radius: int = 5,
    alpha: float = 0.8,
    thickness: int = 1,
    skeleton_style: str = 'mmpose',
    draw_heatmap = False,
    show_kpt_idx = False,
    show = False,
    show_interval : int = 0,
    draw_bbox = False,
    save_img = False
) -> None:
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load the pose model
    pose_estimator = init_pose_estimator(pose_model, pose_checkpoint, device=device)
    pose_estimator.cfg.visualizer.radius = radius
    pose_estimator.cfg.visualizer.alpha = alpha
    pose_estimator.cfg.visualizer.line_width = thickness
    
    # Set the visualizer
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style=skeleton_style)
    
    # Load the detection model
    detector = init_detector(det_model, det_checkpoint, device=device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    
    # Make the output directory
    video_name = get_file_name(video_path)
    img_dir = os.path.join(output_path, 'img', video_name)
    os.makedirs(img_dir, exist_ok=True)
    data_dir = os.path.join(output_path, 'data', video_name)
    os.makedirs(data_dir, exist_ok=True)
    
    # Main Loop
    video_writer = None
    tracked_box = None
    frame_idx = 1
    
    # Define the position and visibility dataframes
    columns = ['frame'] + [f"{kpt}_{xy}" for kpt in exp_keypoints_list for xy in ["x", "y"]]
    position_df = pd.DataFrame(columns=columns)
    visibility_df = pd.DataFrame(columns=['frame'] + exp_keypoints_list)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if not cap.isOpened():
        print('Error: Unable to open the video.')
        sys.exit(1)
    
    while True:
        ret, img = cap.read()
        print(f'Processing frame: {frame_idx}/{total_frames}', end='\r')
        
        if not ret:
            break
        
        if video_writer is None:
            h, w, _ = img.shape
            video_writer = cv2.VideoWriter(
                os.path.join(data_dir, FileName.output_video),
                cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        # Detect and track
        det_result = inference_detector(detector, img)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == det_cat_id, pred_instance.scores > bbox_thr)]
        bboxes = bboxes[nms(bboxes, nms_thr), :]

        if len(bboxes) != 0:
            if tracked_box is None:
                sorted_bboxes = sorted(bboxes, key=lambda x: x[-1], reverse=True)
                tracked_box = sorted_bboxes[conf_rank - 1][:4]

            elif tracked_box is not None:
                max_iou = 0
                for bbox in bboxes:
                    box = bbox[:4]
                    iou = calculate_iou(tracked_box, box)
                    if iou > max_iou:
                        max_iou = iou
                        tracked_box = box
                print(f"\nIOU: {max_iou:.2f}")
                if max_iou < iou_thr:
                    # tracked_box = None
                    break
        
        if tracked_box is not None:
            pose_results = inference_topdown(pose_estimator, img, [tracked_box])
            data_samples = merge_data_samples(pose_results)
        
            if isinstance(img, str):
                img = mmcv.imread(img, channel_order='rgb')
            elif isinstance(img, np.ndarray):
                img = mmcv.bgr2rgb(img)

            # visualize the results
            if visualizer is not None:
                visualizer.add_datasample(
                    'result',
                    img,
                    data_sample=data_samples,
                    draw_gt=False,
                    draw_heatmap=draw_heatmap,
                    draw_bbox=draw_bbox,
                    show_kpt_idx=show_kpt_idx,
                    skeleton_style=skeleton_style,
                    show=show,
                    wait_time=show_interval,
                    kpt_thr=kpt_thr)
            
            # Save the results
            bgr_img = cv2.cvtColor(visualizer.get_image(), cv2.COLOR_RGB2BGR)
            cv2.putText(bgr_img, f'Frame: {frame_idx}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            video_writer.write(bgr_img)
            if save_img:
                cv2.imwrite(os.path.join(img_dir, f'{video_name}_{frame_idx}.jpg'), bgr_img)
            
            position_data = [frame_idx] + pose_results[0].pred_instances.cpu().numpy().keypoints[0, :23].ravel().tolist()
            position_df.loc[len(position_df)] = position_data
            visibility_data = [frame_idx] + pose_results[0].pred_instances.cpu().numpy().keypoint_scores[0, :23].tolist()
            visibility_df.loc[len(visibility_df)] = visibility_data
        else:
            cv2.putText(img, f'Frame: {frame_idx}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            video_writer.write(img)
            if save_img:
                cv2.imwrite(os.path.join(img_dir, f'{video_name}_{frame_idx}.jpg'), img)
            
            position_data = [frame_idx] + [np.nan] * 34
            position_df.loc[len(position_df)] = position_data
            visibility_data = [frame_idx] + [np.nan] * 17
            visibility_df.loc[len(visibility_df)] = visibility_data
        
        frame_idx += 1
    
    cap.release()
    video_writer.release()
    
    position_df.to_csv(os.path.join(data_dir, FileName.position_data), index=False, header=True)
    visibility_df.to_csv(os.path.join(data_dir, FileName.visibility_data), index=False, header=True)
    print('\nDone')