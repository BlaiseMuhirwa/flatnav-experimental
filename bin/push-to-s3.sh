#!/bin/bash 


# destination folder 
DESTINATION_LOCAL="/home/ubuntu/flatnav-experimental/node-access-distributions"

echo "Destionation folder is $DESTINATION_LOCAL"


json_files=(
    # "gist-960-euclidean_node_access_counts.json"
    # "glove-100-angular_node_access_counts.json"
    # "nytimes-256-angular_node_access_counts.json"
    # "sift-128-euclidean_node_access_counts.json"
    "normal-1-angular_node_access_counts.json"
    "normal-1-euclidean_node_access_counts.json"
    "normal-2-angular_node_access_counts.json"
    "normal-2-euclidean_node_access_counts.json"
    "normal-4-angular_node_access_counts.json"
    "normal-4-euclidean_node_access_counts.json"
    "normal-8-angular_node_access_counts.json"
    "normal-8-euclidean_node_access_counts.json"
    "normal-16-angular_node_access_counts.json"
    "normal-16-euclidean_node_access_counts.json"
    "normal-32-angular_node_access_counts.json"
    "normal-32-euclidean_node_access_counts.json"
    "normal-64-angular_node_access_counts.json"
    "normal-64-euclidean_node_access_counts.json"
    "normal-128-angular_node_access_counts.json"
    "normal-128-euclidean_node_access_counts.json"
    "normal-256-angular_node_access_counts.json"
    "normal-256-euclidean_node_access_counts.json"
    "normal-1024-angular_node_access_counts.json"
    "normal-1024-euclidean_node_access_counts.json"
    "normal-1536-angular_node_access_counts.json"
    "normal-1536-euclidean_node_access_counts.json"
)


# Now iterate through the list and push to s3. The prefix to push to will be the first
# part of the name.
# Ex. for normal-1-angular_node_access_counts.json, the prefix will be normal-1-angular
for file in "${json_files[@]}"
do
    prefix="${file%%_*}"
    echo "prefix is $prefix"

    # Copy the entire directory from s3 to local excluding any files with the .index and 
    # .json extensions 
    aws s3 cp s3://hnsw-index-snapshots/$prefix $DESTINATION_LOCAL \
                --recursive \
                --exclude "*.index" \
                --exclude "*.npy"
done