#!/bin/bash

# DIR_VID="/mnt/f/old_videos/1996_05.mp4"
# DIR_OUT1="/mnt/f/old_videos/ai_output/rvrt"

DIR_VID="/data/users/firat/other_data/1996_05.mp4"
DIR_OUT1="/data/users/firat/other_data/1996_05/rvrt"

LD_LIBRARY_PATH=/home/firat/miniconda3/envs/rvrt/lib/python3.8/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH CUDA_HOME=/home/firat/miniconda3/envs/rvrt/lib/python3.8/site-packages/nvidia/cuda_runtime python firat_test_rvrt.py --task=001_RVRT_videosr_bi_REDS_30frames --fname_lq=$DIR_VID --folder_out=$DIR_OUT1 --tile 0 0 0 --tile_overlap 2 20 20 --sigma=20 --num_workers=8 