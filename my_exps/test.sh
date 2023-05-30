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

config=$1
ckpt=$2
save_dir=$3
cuda=$4
args=${@:5}

echo -e "\033[33mconfig is ${config}\033[0m"
echo -e "\033[33mcheckpoint is ${ckpt}\033[0m"
echo -e "\033[33msave_dir is ${save_dir}\033[0m"
echo -e "\033[33mdevice is ${cuda}\033[0m"
echo -e "\033[33margs is ${args}\033[0m"

if [ -d ${save_dir} ]; then
 cd ${save_dir}/../
 rm -rf ${save_dir}
  cd -
fi

CUDA_VISIBLE_DEVICES=${cuda} python tools/test.py \
${config} ${ckpt} --format-only --options save_dir=${save_dir} nproc=1 ${args}

