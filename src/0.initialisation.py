#!/usr/bin/env python

from reaching_model_utils.config import load_config

cft = load_config()

print("Creating path system... \nThe following folders have been created :")

for _, path in cft.paths : 
    print("\t", path)
    path.mkdir(parents=True, exist_ok=True)