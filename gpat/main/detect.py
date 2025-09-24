import os
import sys
import warnings

warnings.filterwarnings('ignore')

import cv2
import mmcv
import mmengine
import numpy as np
import torch
from gpat.utils.files import FileName
from gpat.utils.utils import calculate_iou, get_file_name
from mmdet.apis import inference_detector, init_detector

from mmpose.evaluation.functional import nms
from mmpose.utils import adapt_mmdet_pipeline


def detect_and_track(
    video_path: str,
    model_path: str,
    config_path: str,
    output_path: str,
    det_cat_id: int = 0,
    bbox_thr: float = 0.3,
    nms_thr: float = 0.3,
    iou_thr: float = 0.5,
) -> None:
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load the model
    detector = init_detector(model_path, config_path, device=device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    
    # Make the output directory
    video_name = get_file_name(video_path)
    data_dir = os.path.join(output_path, 'data', video_name)
    os.makedirs(data_dir, exist_ok=True)
    
    # Write the tracking data header
    with open(os.path.join(data_dir, FileName.tracking_data), 'w') as f:
        f.write('frame,track_id,x1,y1,x2,y2\n')
    
    video_writer = None
    tracking_info = {}
    frame_idx = 0
    next_id = 1
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if not cap.isOpened():
        print('Error: Unable to open the video.')
        sys.exit(1)
    
    while True:
        ret, img = cap.read()
        # Update the frame index
        frame_idx += 1
        print(f'Processing frame: {frame_idx}/{total_frames}', end='\r')
        
        if not ret:
            break
        
        if video_writer is None:
            h, w, _ = img.shape
            video_writer = cv2.VideoWriter(
                os.path.join(data_dir, FileName.output_tracking_video),
                cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        # Perform detection
        det_result = inference_detector(detector, img)
        pred_instance = det_result.pred_instances.cpu().numpy()
        scores = pred_instance.scores
        bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == det_cat_id, pred_instance.scores > bbox_thr)]
        bboxes = bboxes[nms(bboxes, nms_thr), :4]
        scores = scores[np.logical_and(pred_instance.labels == det_cat_id, pred_instance.scores > bbox_thr)]

        # Current frame tracking information
        current_frame_tracking = {}

        # Update existing tracking information or assign a new ID
        for box, score in zip(bboxes, scores):
            max_iou = 0
            assigned_id = None
            for obj_id, obj_data in tracking_info.items():
                iou = calculate_iou(obj_data['box'], box)
                if iou > max_iou:
                    max_iou = iou
                    assigned_id = obj_id
            if max_iou > iou_thr:
                current_frame_tracking[assigned_id] = {'box': box, 'score': score}
            else:
                current_frame_tracking[next_id] = {'box': box, 'score': score}
                next_id += 1

        # Update the overall tracking information with the current frame's tracking data
        tracking_info = current_frame_tracking

        # Draw bounding boxes and IDs on the frame
        # Write the tracking data
        with open(os.path.join(data_dir, FileName.tracking_data), 'a') as f:
            for obj_id, obj_data in tracking_info.items():
                box = obj_data['box']
                score = obj_data['score']
                x1, y1, x2, y2 = map(int, box[:4])
                if obj_id == 1:
                    color = (0, 0, 255)
                elif obj_id == 2:
                    color = (0, 255, 0)
                elif obj_id == 3:
                    color = (255, 0, 0)
                else:
                    color = (255, 255, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f'{obj_id} ({score:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                f.write(f'{frame_idx},{obj_id},{x1},{y1},{x2},{y2}\n')
        
        # Write the frame to the output video
        video_writer.write(img)
    
    cap.release()
    video_writer.release()
    print('\nDone')
    
if __name__ == "__main__":
    detect_and_track(
    video_path="/home/ohwada/sasaki_20240930/video/left_34.MP4" ,
    model_path="/home/ohwada/human_pose_estimation/models/rtmdet_l_8xb32-300e_coco.py" ,
    config_path="/home/ohwada/human_pose_estimation/models/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth" ,
    output_path="/home/ohwada/detect_/" ,
    )