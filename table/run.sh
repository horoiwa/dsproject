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
2. Preprocessing

- poly
- Encoding
"""
poetry run python command.py preprocess -i "${NAME}.csv" -p "${NAME}.csv"
