# CUDA_VISIBLE_DEVICES=0 python eval_image.py \
#   --datapath /datanas01/nas01/Student-home/2025D_ShiGuangze/Data/ \
#   --dataset-name ISIC \
#   --img-size 512 \
#   --nshot 1 \
#   --support-indices "[0]" \
#   --bsz 48 \
#   --nworker 4 \
#   --ckpt /datanas01/nas01/Student-home/2025D_ShiGuangze/Code/FSS/DC-SAM-main/output/COCO_90K_Res50_SAM1/_TEST_0423_180507/best_model.pt \
#   --save-dir output/preds/SAM1/ISIC \
#   --sam_version 1 \
#   --sam1-ckpt /datanas01/nas01/Student-home/2025D_ShiGuangze/Code/GPO_1/GPO/segmenter/checkpoint/sam_vit_h_4b8939.pth \
#   --num_query 25 \
#   --backbone resnet50

#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python train_image.py \
#   --datapath /datanas01/nas01/Student-home/2025D_ShiGuangze/Data/ \
#   --dataset-name coco9k_80 \
#   --img-size 512 \
#   --nshot 1 \
#   --support-indices "[0]" \
#   --bsz 48 \
#   --epochs 50 \
#   --nworker 4 \
#   --sam_version 2 \
#   --num_query 25 \
#   --backbone resnet50 \
#   --log-root output/COCO_90K_Res50_SAM2

CUDA_VISIBLE_DEVICES=0

DATAPATH="/datanas01/nas01/Student-home/2025D_ShiGuangze/Data/"
SAM1_CKPT="/datanas01/nas01/Student-home/2025D_ShiGuangze/Code/GPO_1/GPO/segmenter/checkpoint/sam_vit_h_4b8939.pth"
SAM2_CKPT="/datanas01/nas01/Student-home/2025D_ShiGuangze/Code/IFP/Segment_Anything2/checkpoints/sam2.1_hiera_large.pt"

MODEL_CKPT_SAM1="/datanas01/nas01/Student-home/2025D_ShiGuangze/Code/FSS/DC-SAM-main/output/COCO_90K_Res50_SAM1/_TEST_0423_180507/best_model.pt"
MODEL_CKPT_SAM2="/datanas01/nas01/Student-home/2025D_ShiGuangze/Code/FSS/DC-SAM-main/output/COCO_90K_Res50_SAM2/_TEST_0423_222056/best_model.pt"

DATASETS=(ISIC Kvasir TEM COCO VOC)

for DATASET in "${DATASETS[@]}"
do
#   echo "===== Testing ${DATASET} with SAM1 ====="
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python eval_image.py \
#     --datapath "${DATAPATH}" \
#     --dataset-name "${DATASET}" \
#     --img-size 512 \
#     --nshot 1 \
#     --support-indices "[0]" \
#     --bsz 48 \
#     --nworker 4 \
#     --ckpt "${MODEL_CKPT_SAM1}" \
#     --save-dir "output/preds/SAM1/${DATASET}" \
#     --sam_version 1 \
#     --sam1-ckpt "${SAM1_CKPT}" \
#     --num_query 25 \
#     --backbone resnet50

  echo "===== Testing ${DATASET} with SAM2 ====="
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python eval_image.py \
    --datapath "${DATAPATH}" \
    --dataset-name "${DATASET}" \
    --img-size 512 \
    --nshot 1 \
    --support-indices "[0]" \
    --bsz 48 \
    --nworker 4 \
    --ckpt "${MODEL_CKPT_SAM2}" \
    --save-dir "output/preds/SAM2/${DATASET}" \
    --sam_version 2 \
    --sam2-ckpt "${SAM2_CKPT}" \
    --num_query 25 \
    --backbone resnet50
done