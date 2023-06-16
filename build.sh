
mkdir -p build 
# The benchmarks are off by default
cd build && cmake -DRUN_BENCHMARKS=OFF .. 
make -j 