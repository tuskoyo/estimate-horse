python mmpose/demo/top_down_video_demo_with_mmdet.py \
    mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/res50_horse10_256x256-split1.py \
    https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split1-3a3dc37e_20210405.pth \
    --video-path https://user-images.githubusercontent.com/15977946/173124855-c626835e-1863-4003-8184-315bc0b7b561.mp4 \
    --out-video-root vis_results \
    --bbox-thr 0.1 \
    --kpt-thr 0.4 \
    --det-cat-id 18