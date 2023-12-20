<div align="center"> 

## ✨𝓓𝓕𝓓𝓮𝓽✨

[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=18&duration=2000&pause=500&center=true&vCenter=true&multiline=true&random=false&width=500&height=90&lines=Oriented+Object+Detection+via+Contextual;Dependence+Mining+and+Penalty-Incentive;Allocation)](https://git.io/typing-svg)

</div>

## Introduction

This is the official implementation🏢 of [DFDet](), which is implemented on [OBBDetection](https://github.com/jbwang1997/OBBDetection)

## Update

- (**2023-05-20**) Release [DFDet](configs/dfdet/dfdet_r50_dota1.py)🔥.

## Installation

Please refer to [install.md](docs/install.md) for installation and dataset preparation.

## Get Started

### How to use OBBDetection

If you want to train or test a oriented model, please refer to [oriented_model_starting.md](docs/oriented_model_starting.md).

### How to Start DFDet

To help you start quickly, I prepare a simple bash script

#### Train


```bash
config=/path/to/config && work_dir=/path/to/work_dir && cuda=(device_id, like 0, 1, 2, 3 ...)
bash my_exps/train.sh ${config} ${work_dir} ${cuda}
```

#### Test

```bash
config=/path/to/config && ckpt=/path/to/checkpoint && save_dir=/path/to/results_save_dir && cuda=(same as above)
bash my_exps/test.sh ${config} ${ckpt} ${save_dir} ${cuda}
```

### How to Deploy DFDet

TODO:

## Cite

TODO:

## License
This project is released under the [Apache 2.0 license](LICENSE).
