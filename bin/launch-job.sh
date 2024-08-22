#!/bin/bash 


./bin/docker-run.sh sift-bench-flatnav-unpruned
./bin/docker-run.sh sift-bench-flatnav-pruned

./bin/docker-run.sh glove100-bench-flatnav-unpruned
./bin/docker-run.sh glove100-bench-flatnav-pruned

./bin/docker-run.sh nytimes-bench-flatnav-unpruned
./bin/docker-run.sh nytimes-bench-flatnav-pruned