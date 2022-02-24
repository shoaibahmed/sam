#!/bin/bash

# srun -p A100 -K --nodes=1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=5 \
#     --mem=24G --kill-on-bad-exit --job-name sam-test --nice=0 --time=3-00:00:00 \
#     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui \
#     --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
#     --container-workdir=`pwd` --container-mount-home --export='NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5' \
#     /opt/conda/bin/python ./train.py

srun -p A100 -K --nodes=1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=5 \
    --mem=24G --kill-on-bad-exit --job-name sam-noised-test --nice=0 --time=3-00:00:00 \
    --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui \
    --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    --container-workdir=`pwd` --container-mount-home --export='NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5' \
    /opt/conda/bin/python ./train_noised.py
