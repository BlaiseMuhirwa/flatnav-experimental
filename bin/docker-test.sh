#!/bin/bash 

# Use this script to build and run the docker container.
# Optionally, you can set the environment variable INCLUDE_HNSWLIB to false to build
# the docker image without hnswlib. This is useful for experiments that only require 
# FlatNav. 
# Example usage:
# export INLCUDE_HNSWLIB=false
# ./bin/docker-test.sh sift-bench 


# Print commands and exit on errors
set -ex 

# Make sure we are one level above this directory
cd "$(dirname "$0")/.."

function get_tag_name() {
    # Returns a string to be used as a docker tag revision.
    # If it's in a clean git repo, it returns the commit's short hash with branch name like ce37fd7-main
    # If the working tree is dirty, it returns something like main-ce37fd7-dirty-e52e78f86e575bd
    #     including the branch name, and a consistent hash of the uncommitted changes

    fail() {
        echo $1
        exit 1
    }

    if [[ ! -z "${OVERRIDE_GIT_TAG_NAME}" ]]; then
        echo $OVERRIDE_GIT_TAG_NAME
        exit 0
    fi

    # Figure out which SHA utility exists on this machine.
    HASH_FUNCTION=sha1sum
    which $HASH_FUNCTION > /dev/null || HASH_FUNCTION=shasum
    which $HASH_FUNCTION > /dev/null || fail "Can't find SHA utility"

    # Try to get current branch out of GITHUB_REF for CI
    # The ##*/ deletes everything up to /
    CURRENT_BRANCH=${GITHUB_REF##*/}
    # Now generate the short commit
    CURRENT_COMMIT=$(echo $GITHUB_SHA | cut -c -9)

    # If we're not running in CI, GITHUB_REF and GITHUB_SHA won't be set.
    # In this case, figure them out from our git repository
    # (If we do this during github CI, we get a useless unique commit on the "merge" branch.)
    # When infering CURRENT_BRANCH, convert '/'s to '-'s, since '/' is not allowed in docker tags but
    # is part of common git branch naming formats e.g. "feature/branch-name" or "user/branch-name"
    CURRENT_BRANCH=${CURRENT_BRANCH:-$(git rev-parse --abbrev-ref HEAD | sed -e 's/\//-/g')}
    CURRENT_COMMIT=${CURRENT_COMMIT:-$(git rev-parse --short=9 HEAD)}

    if [[ -z "$(git status --porcelain)" ]] || [[ "${CI}" = true ]]; then
        # Working tree is clean
        echo "${CURRENT_COMMIT}-${CURRENT_BRANCH}"
    else
        # Working tree is dirty.
        HASH=$(echo $(git diff && git status) | ${HASH_FUNCTION} | cut -c -15)
        echo "${CURRENT_BRANCH}-${CURRENT_COMMIT}-dirty-${HASH}"
    fi
}

# Get the tag name
TAG_NAME=$(get_tag_name)

INCLUDE_HNSWLIB=${INCLUDE_HNSWLIB:-true}
DATA_DIR=${DATA_DIR:-$(pwd)/data}


echo "Building docker image with tag name: $TAG_NAME"

# Make sure the data/ directory exists
mkdir -p data

# Clean up existing docker images matching "flatnav" if any 
docker rmi -f $(docker images --filter=reference="flatnav" -q) &> /dev/null || true

docker build --build-arg INCLUDE_HNSWLIB=$INCLUDE_HNSWLIB \
             --tag flatnav:$TAG_NAME -f Dockerfile .

# Check if the first argument is set. If it is, then run docker container with the 
# first argument as the make target. If not, then run the container with the default
# make target
if [ -z "$1" ]
then
    # This will build the image and run the container with the default make target
    # (i.e., print help message)
    docker run -it --volume ${DATA_DIR}:/root/data --rm flatnav:$TAG_NAME make help
    exit 0
fi


# Run the container and mount the data/ directory as volume to /root/data
# Pass the make target as argument to the container. 
ARG1=$1
docker run -it --volume ${DATA_DIR}:/root/data flatnav:$TAG_NAME /bin/bash \
                -c "make ${ARG1}; tail -f /dev/null"
