#!/bin/bash


NAME="Version1"

"""
1. load raw dataset
  --out: out/path.csv (or hdf)
  --sample: load boston housing dataset
"""
poetry run python command.py load -o "${NAME}.csv"


"""
2. Minimum data cleansing

Note:
 MINIMUM is that LightGBM can be used
"""
poetry run python command.py clean -i "${NAME}.csv" -o "${NAME}.csv"


"""
3. Create Features
"""
poetry run python command.py feat -i "${NAME}.csv"


"""
4.  Run analysis
"""
poetry run python command.py feat -i "filename.csv"
