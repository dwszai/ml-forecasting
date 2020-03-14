#!/bin/bash
cd mlp
python algorithm_module.py
python feature_engineer_module.py
python missing_module.py
python visualization_module.py
cd ..
jupyter notebook eda.ipynb
