# emergent-entanglement-structures
Code and data for [B. Sokolov, M. A. C. Rossi, G. García-Pérez, S. Maniscalco, arXiv:2007.06989 (2007)](https://arxiv.org/abs/2007.06989)

## Installation

    git clone https://github.com/matteoacrossi/emergent-entanglement-structures.git
    cd emergent-entanglement-structures

In a Python 3.7+ environment install the requirements:

    pip install -r requirements.txt

## Usage

The `data/` folder contains the raw data obtained from simulations of the XX model. 

The files `fig_*.py` generate the figures in the paper, which will be saved in the `figures/` folder. For example,

    python fig_communities.py

will generate the `figures/fig_communities.pdf` file.
