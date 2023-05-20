#! /bin/bash

env=open-mmlab # change your anacond3 env
source ~/anaconda3/etc/profile.d/conda.sh # change your anaconda3 path

conda activate ${env}
echo -e "\033[34m*******************************\033[0m"
echo -e "\033[31mactivate env ${env}\033[0m"
echo -e "\033[34m*******************************\033[0m"
echo -e "\033[34mCurrent dir is ${PWD}\033[0m"
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

