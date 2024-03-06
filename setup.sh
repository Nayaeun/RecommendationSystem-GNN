#!/bin/bash

# Install PyTorch
pip install torch==1.10.0

# Install torch-scatter and torch-sparse
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cpu.html

# Install PyTorch Geometric
pip install git+https://github.com/pyg-team/pytorch_geometric.git

# Clone the required data repository
git clone https://github.com/pmaldonado/cs224w-project-data.git

# Install packages from requirements.txt
pip install -r environments/requirements.txt
