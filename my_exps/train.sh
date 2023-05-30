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
work_dir=$2
cuda=${3:0}

if [ -z ${config} ]; then 
  echo -e "\033[31mconfig can't be null\033[0m"
  exit 1
fi

if [ -z ${work_dir} ]; then
  echo -e "\033[31mwork_dir can't be null\033[0m"
  exit 1
fi

if [ -z ${cuda} ]; then
  echo -e "\033[31mcuda can't be null\033[0m"
  exit 1
fi

echo -e "\033[33mconfig is ${config}\033[0m"
echo -e "\033[33mwork_dir is ${work_dir}\033[0m"
echo -e "\033[33mdevice is ${cuda}\033[0m"
sleep 2s
if [ -d ${work_dir} ]; then
    read -n1 -p "find ${work_dir}, do you want to del(y or n):"
    echo 
    if [ ${REPLY}x = yx ]; then  
	rm -rf ${work_dir}
	echo -e "\033[31mAlready del ${work_dir}\033[0m"
    else
	ls -a | grep *log*
	read -n1 -p "do you want to del log(y or n):"
	echo
	if [ ${REPLY}x = yx ]; then
	   rm -rf *log*
	   echo -e "\033]31mAlready del log files\033[0m"
	fi
    
    fi
fi
echo -e "\033[34m*******************************\033[0m"
    
CUDA_VISIBLE_DEVICES=${cuda} python tools/train.py ${config} --work-dir ${work_dir} # --no-validate
