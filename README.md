# CS539-HW-3
### Learning Goal
The goal of this third exercise is to understand and implement three different generic
inference algorithms: importance sampling (IS), Metropolis Hastings within Gibbs
(MH Gibbs), and Hamiltonian Monte Carlo (HMC).


## Setup
**Note:** This code base was developed on Python3.7

Clone Daphne directly into this repo:
```bash
git clone git@github.com:plai-group/daphne.git
```
(To use Daphne you will need to have both a JVM installed and Leiningen installed)

```bash
pip3 install -r requirements.txt
```

## Usage
1. Change the daphne path in `evaluation_based_sampling.py` and run:
```bash
python3 evaluation_based_sampling.py
```

2. Change the daphne path in `graph_based_sampling.py` and run:
```bash
python3 evaluation_based_sampling.py
```

2. Change the daphne path in `MH-gibbs.py` and run:
```bash
python3 MH-gibbs.py
```

2. Change the daphne path in `HMC.py` and run:
```bash
python3 HMC.py
```
