#!/bin/bash


echo "Hello"

# Name for the container
CONTAINER_NAME="ollama"

# Build image (optional — comment out if you use official ollama/ollama directly)
# docker build -t my-ollama .

# Stop and remove any existing container with the same name
if [ $(docker ps -aq -f name=$CONTAINER_NAME) ]; then
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

# Run Ollama with a 2GB RAM limit
docker run -d \
  --name $CONTAINER_NAME \
  -p 11434:11434 \
  --memory=2g \
  --memory-swap=2g \
  ollama/ollama:latest

curl http://localhost:11434/

curl -s http://localhost:11434/api/generate -d '{
  "model": "gemma2:2b",
  "prompt": "Welcome to bad file systems!!"
}'

echo "Ollama is buzzin"
