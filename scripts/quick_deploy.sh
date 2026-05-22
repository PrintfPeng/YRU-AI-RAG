#!/bin/bash
# =================================================================
# YRU-AI-RAG: Quick Deploy Script
# วิธีใช้:
#   1. SSH เข้า server: ssh yruadmin@10.20.41.108
#   2. รันคำสั่ง: bash <(curl -s https://raw.githubusercontent.com/PrintfPeng/YRU-AI-RAG/main/scripts/quick_deploy.sh)
#   หรือถ้า copy ไฟล์ขึ้นมาแล้ว: bash quick_deploy.sh
# =================================================================
set -e

echo "🚀 Starting YRU-AI-RAG Deploy..."
echo ""

# ===== หา Project Directory =====
PROJECT_DIR=""
SEARCH_PATHS=(
    "/home/yruadmin/Data_Ingestion_Hybrid_RAG"
    "/home/yruadmin/YRU-AI-SPACE-NEW/Data_Ingestion_Hybrid_RAG"
    "/root/Data_Ingestion_Hybrid_RAG"
    "/opt/hybrid_rag"
)

for dir in "${SEARCH_PATHS[@]}"; do
    if [ -d "$dir" ] && [ -f "$dir/docker-compose.yml" ]; then
        PROJECT_DIR="$dir"
        break
    fi
done

if [ -z "$PROJECT_DIR" ]; then
    echo "🔍 Searching for docker-compose.yml..."
    COMPOSE_FILE=$(find / -name "docker-compose.yml" -path "*Data_Ingestion*" 2>/dev/null | head -1)
    if [ -n "$COMPOSE_FILE" ]; then
        PROJECT_DIR=$(dirname "$COMPOSE_FILE")
    fi
fi

if [ -z "$PROJECT_DIR" ]; then
    echo "❌ Project directory not found!"
    echo "Please specify manually: cd /path/to/project && git pull && docker-compose restart"
    exit 1
fi

echo "✅ Project: $PROJECT_DIR"
cd "$PROJECT_DIR"

# ===== Git Pull =====
echo ""
echo "📥 Pulling latest code from GitHub..."
git fetch origin
git reset --hard origin/main
echo "✅ Code updated to: $(git log --oneline -1)"

# ===== Docker Restart =====
echo ""
echo "🐳 Restarting Docker containers..."
CONTAINER_NAME=$(docker-compose ps --services 2>/dev/null | grep -i "backend\|rag" | head -1)

if [ -z "$CONTAINER_NAME" ]; then
    # ลอง restart ทั้งหมด
    docker-compose restart
else
    echo "  Restarting: $CONTAINER_NAME"
    docker-compose restart "$CONTAINER_NAME"
fi

# รอ container พร้อม
echo "  Waiting for container to be ready..."
sleep 5

# ===== Status Check =====
echo ""
echo "📊 Container Status:"
docker-compose ps

echo ""
echo "📋 Last 30 lines of logs:"
docker-compose logs --tail=30 2>/dev/null | tail -30

echo ""
echo "✅ Deploy complete!"
echo "   Test chat at: http://$(hostname -I | awk '{print $1}'):8000/app/index.html"
