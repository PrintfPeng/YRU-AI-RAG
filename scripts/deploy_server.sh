#!/bin/bash
# deploy.sh - Pull latest code and restart docker container

echo "=== Finding project directory ==="
# Find the project directory
PROJECT_DIR=""
for dir in \
    "/home/yruadmin/Data_Ingestion_Hybrid_RAG" \
    "/root/Data_Ingestion_Hybrid_RAG" \
    "/opt/Data_Ingestion_Hybrid_RAG" \
    "/home/yruadmin/YRU-AI-SPACE-NEW/Data_Ingestion_Hybrid_RAG" \
    "/root/YRU-AI-SPACE-NEW/Data_Ingestion_Hybrid_RAG"; do
    if [ -d "$dir" ] && [ -f "$dir/docker-compose.yml" ]; then
        PROJECT_DIR="$dir"
        break
    fi
done

if [ -z "$PROJECT_DIR" ]; then
    echo "❌ Project directory not found! Searching..."
    PROJECT_DIR=$(find / -name "docker-compose.yml" -path "*/Data_Ingestion*" 2>/dev/null | head -1 | xargs dirname)
fi

if [ -z "$PROJECT_DIR" ]; then
    echo "❌ Cannot find project. Please check manually."
    exit 1
fi

echo "✅ Found project at: $PROJECT_DIR"
cd "$PROJECT_DIR"

echo "=== Current git status ==="
git log --oneline -3

echo "=== Pulling latest code ==="
git pull origin main

echo "=== Restarting Docker container ==="
if docker-compose ps | grep -q "Up"; then
    docker-compose restart hybrid_rag_backend 2>/dev/null || docker-compose restart
    echo "✅ Container restarted"
else
    docker-compose up -d
    echo "✅ Container started"
fi

echo "=== Container status ==="
docker-compose ps

echo "=== Last 20 lines of log ==="
docker-compose logs --tail=20 hybrid_rag_backend 2>/dev/null || docker-compose logs --tail=20

echo "=== DONE ==="
