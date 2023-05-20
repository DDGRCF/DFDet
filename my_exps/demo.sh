#! /bin/bash

env=open-mmlab # change your anacond3 env
source ~/anaconda3/etc/profile.d/conda.sh # change your anaconda3 path

img=$1
save=$2
config=$3
ckpt=$4
device=$5

echo "img source: ${img}"
echo "img dest: ${save}"
echo "config: ${config}"
echo "ckpt: ${ckpt}"
echo "device: ${device}"

CUDA_VISIBLE_DEVICES=$device python demo/image_demo.py ${img} ${save} ${config} ${ckpt} --device cuda --score-thr 0.05

