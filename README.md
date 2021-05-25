--------------------------------------

MuonTriggerPhase2RPC: python code for toy simulation of ATLAS RPC trigger
=======================================

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mwaskom/seaborn/blob/master/LICENSE)

This package contains python macros for toy simulation of the ATLAS RPC trigger and for study of neural network regression for estimating transverse momentum of muon candidates.

Dependencies
------------
 
These macros require Python 3.7+ and PyTorch


Documentation
-------------

Make events for NN training without noise:
```bash
python3 macros/runMuonSimulation.py --logy -n 100000 --noise-prob=0.0 --seed=42
```

Run NN training using these events:
```bash
python3 macros/trainNN.py rpc-sim_100000-events_0000-noise_000042-seed/events.csv
```

Make events for NN evaluation with noise:
```bash
python3 macros/runMuonSimulation.py --logy -n 200000 --noise-prob=0.001 --seed=101042
```

Make plots:
```bash
python3 macros/runMuonSimulation.py --logy --in-pickle=rpc-sim_200000-events_0010-noise_101042-seed/events.pickle --torch-model=rpc-sim_100000-events_0000-noise_000042-seed/events.pt --plot
```

Draw event displays:
```bash
python3 macros/runMuonSimulation.py --logy --in-pickle=rpc-sim_200000-events_0010-noise_101042-seed/events.pickle --torch-model=rpc-sim_100000-events_0000-noise_000042-seed/events.pt --draw
```


