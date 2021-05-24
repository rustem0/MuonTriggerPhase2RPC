--------------------------------------

MuonTriggerPhase2RPC: python code for toy simulation of ATLAS RPC trigger
=======================================

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mwaskom/seaborn/blob/master/LICENSE)

This package contains python macros for toy simulation of the ATLAS RPC trigger and for study of neural network regression for estimating transverse momentum of muon candidates.


Documentation
-------------

Produce events for NN training without noise:
```bash
python3 macros/runMuonSimulation.py --logy -n 100000 --noise-prob=0.0 -o train_100k_noise-off/train_100k_noise-off.csv
```

Run NN training using these events:
```bash
python3 macros/trainNN.py train_100k_noise-off/train_100k_noise-off.csv
```

Produce events for NN evaluation with noise:
```bash
python3 macros/runMuonSimulation.py --logy -n 200000 --noise-prob=0.001 -o train_200k_noise-on/train_200k_noise-on.csv --out-pickle=train_200k_noise-on/train_200k_noise-on.pickle 
```

Dependencies
------------
 
These macros require Python 3.7+ and PyTorch
