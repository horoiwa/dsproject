#!/bin/bash


cd `dirname $0`

cd result find . -type "d" | xargs rm -r

cd result
find . -mindepth "1" -maxdepth "1" -type "d"  | xargs rm -r
cd ..

cd data
find . -name "*.csv" -type "f" | xargs rm
find . -name "*.hdf" -type "f" | xargs rm
cd ..
