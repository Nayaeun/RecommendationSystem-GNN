#!/bin/bash

# Install PyTorch
pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# Install torch-scatter and torch-sparse
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html

# Install PyTorch Geometric
pip install git+https://github.com/pyg-team/pytorch_geometric.git

# Clone the required data repository
git clone https://github.com/pmaldonado/cs224w-project-data.git

# Install packages from requirements.txt
pip install -r requirements.txt
