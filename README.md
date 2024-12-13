# Feudal Graph Reinforcement Learning (TMLR)

[![TMLR](https://img.shields.io/badge/TMLR-Openreview-blue.svg?style=flat-square)](https://openreview.net/forum?id=wFcyJTik90)
[![PDF](https://img.shields.io/badge/%E2%87%A9-PDF-orange.svg?style=flat-square)](https://openreview.net/pdf?id=wFcyJTik90)
[![arXiv](https://img.shields.io/badge/arXiv-2303.14681-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2304.05099)

This repository contains the code for the paper Feudal Graph Reinforcement Learning (TMLR).

**Authors**: [Tommaso Marzi](mailto:tommaso.marzi@usi.ch), Arshjot Khehra, Andrea Cini, Cesare Alippi

## Requirements
The required packages can be installed using the requirements.txt file (we suggest to use a conda environment). 
The code was implemented with Python 3.10.10. Depending on your operating system, further modifications concerning multiprocessing and the MuJoCo environment may be necessary.

## Usage
### Graph Clustering 
All the hyperparameters as well as the choice of the model can be changed in the `config.json` file.
Notice that the code is designed to run on CPU, and the number of cores used to parallelize the simulations can be specified in the config file.

The code can be run using the following command from the `src_GC` directory (you might need to set it as working directory):
```bash
sh ./run_training.sh
```
It will save the `config.json` and the model parameters in the `.\results\EXP_id\` directory.

### MuJoCo Benchmarks
Follow the instruction on the [github page](https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco) to download mujoco (version 2.1.0).

All the hyperparameters as well as the choice of the model can be changed in the `config.py` file.
Notice that the code is designed to run on CPU, and the number of cores used to parallelize the simulations can be specified in the `config.py` file.

The code can be run using the following command from the `src_MB` directory (you might need to set it as working directory):
```bash
sh ./run_training.sh
```
It will save the `config.py`, agent graphs, and the model parameters in the `.\results\EXP_id\` directory.

## Bibtex reference

If you find this code useful please consider citing our paper:

```
@article{marzi2024feudal,
title={Feudal Graph Reinforcement Learning},
author={Tommaso Marzi and Arshjot Singh Khehra and Andrea Cini and Cesare Alippi},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=wFcyJTik90}
}
```