#!/bin/bash

IMAGE_NAME=samurai
CONTAINER_NAME=samurai_container

echo "🔧 Building Docker image: $IMAGE_NAME..."
docker build -t $IMAGE_NAME .

echo "🚀 Running Docker container: $CONTAINER_NAME..."


docker run --gpus all --rm \
  --shm-size=16g \
  --name $CONTAINER_NAME \
  -v $(pwd)/..:/workspace/samurai_deploy \
  $IMAGE_NAME bash
