#!/bin/bash
DATASET_ROOT=dataset/BCD
WORK_SPACE=result
GPU=1

# CAM generation and BIS strategy
#CUDA_VISIBLE_DEVICES=${GPU} python3 run_sample.py \
#     --CAM_root ${DATASET_ROOT} --work_space ${WORK_SPACE}\
#     --num_workers 8 \
#     --cam_learning_rate 0.05 --cam_batch_size 16 --cam_num_epoches 10 \
#     --cam_network net.resnet50_cam --cam_weights_name res50_cam \
#     --train_cam_pass False \
#     --make_cam_pass False --eval_cam_pass False \
#     --cam_to_ir_label_pass False --conf_bg_thres 0.45 --conf_fg_thres 0.55 \
#     --cam_to_mask_pass False --eval_mask_pass False \
#     --SAM_label_pass False --eval_sam_pass True --SAMlabel SAMlabelV1  \

# ASPP training
#CUDA_VISIBLE_DEVICES=${GPU} python3 run_sample.py \
#    --CAM_root ${DATASET_ROOT} --work_space ${WORK_SPACE} \
#    --SAMlabel SAMlabelV1 \
#    --train_amn_pass False --amn_network net.RepVGG_amn --amn_weights_name repvgg_amn\
#    --make_amn_cam_pass False \
#    --eval_amn_cam_pass True

# final pseudomask generation
# CUDA_VISIBLE_DEVICES=${GPU} python3 run_sample.py \
#     --CAM_root ${DATASET_ROOT} --work_space ${WORK_SPACE} \
#     --conf_fg_thres 0.6 --conf_bg_thres 0.7 \
#     --amn_cam_to_ir_label_pass False \
#     --eval_mask_pass False --mask amn_mask\
#     --SAM_label_pass False --eval_sam_pass False --SAMlabel amn_SAMlabelV1
