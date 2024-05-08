

# Run the yandex-deep benchmark and log everything to a logs.txt file 
yandex-deep-bench: 
	poetry run python run-benchmark.py \
		--dataset /media/scratch/yandex-deep/learn.350.fbin \
		--queries /media/scratch/yandex-deep/queries.fbin \
		--gtruth /media/scratch/yandex-deep/ground_truth.10K.bin \
		--metric l2 > logs.txt 2>&1

<<<<<<< HEAD
# Billion-scale benchmark for the Microsoft SpaceV1B dataset
# Reference: https://github.com/microsoft/SPTAG/tree/main/datasets/SPACEV1B
space-v1b-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset-name msft-spacev-1b \
		--dataset /root/data/SPACEV1B/collection.npy \
		--queries /root/data/SPACEV1B/queries.npy \
		--gtruth /root/data/SPACEV1B/ground_truth.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename /scratch/brc7/space-v1b.mtx \
		--num-node-links 32 \
		--ef-construction 30 50 100 \
		--ef-search 100 200 300 500 1000 \
		--num-build-threads 45 \
		--num-search-threads 1 \
		--metric l2

space-v1b-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset-name msft-spacev-1b \
		--dataset /root/data/SPACEV1B/collection.npy \
		--queries /root/data/SPACEV1B/queries.npy \
		--gtruth /root/data/SPACEV1B/ground_truth.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 50 100 \
		--ef-search 100 200 300 500 1000 \
		--num-build-threads 45 \
		--num-search-threads 1 \
		--metric l2
=======
glove-bench:
	poetry run python run-big-bench.py \
		--dataset /root/data/glove-25-angular/glove-25-angular.train.npy \
		--queries /root/data/glove-25-angular/glove-25-angular.test.npy \
		--gtruth /root/data/glove-25-angular/glove-25-angular.gtruth.npy \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename glove.mtx \
		--num-node-links 32 \
		--ef-construction 100 200 \
		--ef-search 100 200 300 \
		--metric angular
>>>>>>> 7be3dd24a9bcdd4a7b000ebb8d94383498830984

# Other available options include: --index-type (hnsw, flatnav)
# We use dataset-name for the name of the plots.

fashion-mnist-bench-flatnav: 
	poetry run python run-benchmark.py \
		--dataset-name fashion-mnist-784 \
		--dataset /root/data/fashion-mnist-784-euclidean/fashion-mnist-784-euclidean.train.npy \
		--queries /root/data/fashion-mnist-784-euclidean/fashion-mnist-784-euclidean.test.npy \
		--gtruth /root/data/fashion-mnist-784-euclidean/fashion-mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename fashion-mnist.mtx \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 


fashion-mnist-bench-hnsw: 
	poetry run python run-benchmark.py \
		--dataset-name fashion-mnist-784 \
		--dataset /root/data/fashion-mnist-784-euclidean/fashion-mnist-784-euclidean.train.npy \
		--queries /root/data/fashion-mnist-784-euclidean/fashion-mnist-784-euclidean.test.npy \
		--gtruth /root/data/fashion-mnist-784-euclidean/fashion-mnist-784-euclidean.gtruth.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 

gist-bench-flatnav: 
	poetry run python run-benchmark.py \
		--dataset-name gist \
		--dataset /root/data/gist-960-euclidean/gist-960-euclidean.train.npy \
		--queries /root/data/gist-960-euclidean/gist-960-euclidean.test.npy \
		--gtruth /root/data/gist-960-euclidean/gist-960-euclidean.gtruth.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename gist.mtx \
		--num-node-links 32 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 500 1000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 


gist-bench-hnsw: 
	poetry run python run-benchmark.py \
		--dataset-name gist \
		--dataset /root/data/gist-960-euclidean/gist-960-euclidean.train.npy \
		--queries /root/data/gist-960-euclidean/gist-960-euclidean.test.npy \
		--gtruth /root/data/gist-960-euclidean/gist-960-euclidean.gtruth.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 500 1000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 
sift-bench-flatnav: 
	poetry run python run-benchmark.py \
		--dataset-name sift \
		--dataset /root/data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries /root/data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth /root/data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename sift.mtx \
		--num-node-links 32 \
		--ef-construction 100 \
		--ef-search 100 200 300 500 1000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 

hubness-test:
	poetry run python hubness.py \
		--datasets normal-1-angular normal-2-angular normal-4-angular normal-8-angular normal-16-angular normal-32-angular normal-64-angular normal-128-angular normal-256-angular normal-1024-angular normal-1536-angular normal-1-euclidean normal-2-euclidean normal-4-euclidean normal-8-euclidean normal-16-euclidean normal-32-euclidean normal-64-euclidean normal-128-euclidean normal-256-euclidean normal-1024-euclidean normal-1536-euclidean \
		--k 100 \
		--ef-construction 100 \
		--ef-search 200 \
		--num-node-links 32 \
		--hubness-scores hubness-scores.json


generate-datasets:
	poetry run python generate-datasets.py \
		--base-path /home/ubuntu/flatnav-experimental/data \
		--dataset-size 1010000 \
		--num-queries 10000 \
		--dimensions 1 2 4 8 16 32 64 128 1536 \
		--k 100

# Generate hubness scores for each desired dataset
generate-hubness-scores:
	poetry run python generate-hubness-scores.py \
		--dataset-names normal-1-angular normal-2-angular normal-4-angular normal-8-angular normal-16-angular normal-32-angular normal-64-angular normal-128-angular normal-256-angular normal-1024-angular normal-1536-angular normal-2048-angular normal-1-euclidean normal-2-euclidean normal-4-euclidean normal-8-euclidean normal-16-euclidean normal-32-euclidean normal-64-euclidean normal-128-euclidean normal-256-euclidean normal-1024-euclidean normal-1536-euclidean normal-2048-euclidean \
		--base-path /media/scratch/ \
		--k 100 \
		--save-file hubness-scores-full.json


sift-bench-hnsw: 
	poetry run python run-benchmark.py \
		--dataset-name sift \
		--dataset /root/data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries /root/data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth /root/data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1

glove100-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset /root/data/glove-100-angular/glove-100-angular.train.npy \
		--queries /root/data/glove-100-angular/glove-100-angular.test.npy \
		--gtruth /root/data/glove-100-angular/glove-100-angular.gtruth.npy \
		--dataset-name glove-100-angular \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename glove100.mtx \
		--num-node-links 32 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

glove100-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset /root/data/glove-100-angular/glove-100-angular.train.npy \
		--queries /root/data/glove-100-angular/glove-100-angular.test.npy \
		--gtruth /root/data/glove-100-angular/glove-100-angular.gtruth.npy \
		--dataset-name glove-100-angular \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular


glove25-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset /root/data/glove-25-angular/glove-25-angular.train.npy \
		--queries /root/data/glove-25-angular/glove-25-angular.test.npy \
		--gtruth /root/data/glove-25-angular/glove-25-angular.gtruth.npy \
		--dataset-name glove-25-angular \
		--index-type flatnav \
		--use-hnsw-base-layer \
        --hnsw-base-layer-filename glove25.mtx \
		--num-node-links 32 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

glove25-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset /root/data/glove-25-angular/glove-25-angular.train.npy \
		--queries /root/data/glove-25-angular/glove-25-angular.test.npy \
		--gtruth /root/data/glove-25-angular/glove-25-angular.gtruth.npy \
		--dataset-name glove-25-angular \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

nytimes-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset /root/data/nytimes-256-angular/nytimes-256-angular.train.npy \
		--queries /root/data/nytimes-256-angular/nytimes-256-angular.test.npy \
		--gtruth /root/data/nytimes-256-angular/nytimes-256-angular.gtruth.npy \
		--dataset-name nytimes-256-angular \
		--index-type flatnav \
		--use-hnsw-base-layer \
                --hnsw-base-layer-filename nytimes.mtx \
		--num-node-links 32 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 500 1000 2500 5000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

nytimes-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset /root/data/nytimes-256-angular/nytimes-256-angular.train.npy \
		--queries /root/data/nytimes-256-angular/nytimes-256-angular.test.npy \
		--gtruth /root/data/nytimes-256-angular/nytimes-256-angular.gtruth.npy \
		--dataset-name nytimes-256-angular \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 500 1000 2500 5000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

node-access-test:
	poetry run python node-access-patterns.py \
		--datasets mnist-784-euclidean sift-128-euclidean glove-25-angular glove-100-angular \
		--k 100 \
		--metrics l2 l2 angular angular \
		--ef-construction 100 \
		--ef-search 200 \
		--num-node-links 32
		
generate-hubness-scores:
	poetry run python generate-hubness-scores.py \
		--base-path ../data \
		--dataset-names normal-256-angular normal-256-euclidean \
		--k 100 \
		--save-file hubness-scores.json 

# sift-128-euclidean glove-100-angular normal-10-angular normal-10-euclidean normal-256-angular normal-256-euclidean normal-1024-angular normal-1024-euclidean
# l2 angular angular l2 angular l2 angular l2
generate-distributions:
	poetry run python generate-distributions.py \
		--datasets mnist-784-euclidean sift-128-euclidean glove-100-angular normal-10-angular normal-10-euclidean normal-256-euclidean normal-256-angular \
		--k 100 \
		--metrics l2 l2 angular angular l2 l2 angular \
		--ef-construction 100 \
		--ef-search 200 \
		--num-node-links 32 \
		--distribution-type node-access


subgraph-analysis:
	poetry run python subgraph-analysis.py \
		--datasets mnist-784-euclidean sift-128-euclidean glove-100-angular normal-256-euclidean \
		--k 100 \
		--metrics l2 l2 angular l2 \
		--ef-construction 100 \
		--ef-search 100 \
		--num-node-links 32
		
get-index-sizes:
	poetry run python index-sizes.py \
		--datasets sift-128-euclidean \
		--metrics l2 \
		--ef-construction 100 \
		--num-node-links 32 \
		--index-type hnsw


setup: install-flatnav install-hnswlib

# Install all dependencies including flatnav
install-flatnav: generate-wheel 
	poetry add ../flatnav_python/dist/*.whl 
	rm poetry.lock && poetry install --no-root

# This will generate the wheel for flatnav and put it in
# ../flatnav_python/dist
generate-wheel:
	poetry remove flatnav &> /dev/null 
	cd .. && cd flatnav_python && ./install_flatnav.sh 

install-hnswlib:
	if [ ! -d "hnswlib-original" ]; then \
		git clone https://github.com/BlaiseMuhirwa/hnswlib-original.git; \
	fi
	cd hnswlib-original/python_bindings && \
	poetry install --no-root && \
	poetry run python setup.py bdist_wheel 
	poetry add hnswlib-original/python_bindings/dist/*.whl

s3-push:
	poetry run python push-snapshot-to-s3.py

cleanup:
	rm -rf hnswlib-original 

# If passed an invalid argument, print help message
%:
	@echo "Invalid argument: $@"
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  setup: install all dependencies including flatnav"
	@echo "  install-flatnav: install flatnav"
	@echo "  install-hnswlib: install hnswlib"
	@echo "  cleanup: remove hnswlib-original"
	@echo "  generate-wheel: generate wheel for flatnav"
	@echo "  yandex-deep-bench: run yandex-deep benchmark"
	@echo "  sift-bench: run sift benchmark"
	
