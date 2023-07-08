#!/bin/bash 

# Default is OFF if no argument is passed
BUILD_TESTS=${1:-OFF}
BUILD_EXAMPLES=${2:-OFF}

mkdir -p build 
cd build && cmake -DBUILD_TESTS=${BUILD_TESTS} -DBUILD_EXAMPLES=${BUILD_EXAMPLES} .. 
make -j 