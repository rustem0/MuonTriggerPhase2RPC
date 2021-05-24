--------------------------------------

MuonTriggerPhase2RPC: python code for toy simulation of ATLAS RPC trigger
=======================================

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mwaskom/seaborn/blob/master/LICENSE)

This package contains python macros for toy simulation of the ATLAS RPC trigger and for study of neural network regression for estimating transverse momentum of muon candidates.


Documentation
-------------

Produce events for NN training without noise:

```bash
python3 macros/runMuonSimulation.py --logy -n 100000 --noise-prob=0.0 -o train_100k_noise-off/train_100k_noise-off.csv &> log100k &

```