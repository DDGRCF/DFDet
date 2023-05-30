#! /bin/bash

:<<!
  A Simple Shell Script, that help you start quickly!
!

# activate your python env
env=your_env # note: reset your env
source /path/to/your/anaconda3/etc/profile.d/conda.sh # note: reset your env

if [ $? -ne 0 ]; then
  echo -e "\033[31msource conda path fail! please check your setting\033[0m"
  exit 1
fi
conda activate ${env}
if [ $? -ne 0 ]; then
  echo -e "\033[31msource conda path fail! please check your setting\033[0m"
  exit 1
fi

echo -e "\033[34m*******************************\033[0m"
echo -e "\033[32mactivate env ${env}\033[0m"
echo -e "\033[34m*******************************\033[0m"
echo -e "\033[32mcurrent dir is ${PWD}\033[0m"
echo -e "\033[34m*******************************\033[0m"

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

