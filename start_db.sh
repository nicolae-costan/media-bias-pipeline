#!/bin/bash

# start_db.sh
# Standard Unix/Bash script to ensure Docker is running and launch the database via Docker Compose

# Color definitions
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}Checking Docker status...${NC}"

# Check if Docker command exists
if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker CLI is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker daemon is running
if ! docker ps &> /dev/null; then
    echo -e "${YELLOW}Docker daemon is not running. Attempting to start service...${NC}"
    
    # Try starting docker service (for Linux systemd environments)
    if command -v systemctl &> /dev/null; then
        sudo systemctl start docker
    elif [ "$(uname)" == "Darwin" ]; then
        # For macOS
        open --background -a Docker
    else
        echo -e "${RED}ERROR: Could not auto-start Docker. Please start Docker Desktop manually.${NC}"
        exit 1
    fi
    
    # Wait for Docker to be ready
    echo -e "${YELLOW}Waiting for Docker to be ready (up to 30 seconds)...${NC}"
    for i in {1..10}; do
        if docker ps &> /dev/null; then
            break
        fi
        sleep 3
    done
fi

# Verify again
if ! docker ps &> /dev/null; then
    echo -e "${RED}ERROR: Docker daemon is still not responsive. Please verify Docker Desktop is running.${NC}"
    exit 1
fi

# Determine if docker-compose or "docker compose" should be used
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    COMPOSE_CMD="docker compose"
fi

echo -e "${GREEN}Starting PostgreSQL + pgvector via Docker Compose...${NC}"
$COMPOSE_CMD up -d

echo -e "\n${GREEN}Database is ready on localhost:5433${NC}"
echo -e "User: ${CYAN}postgres${NC} | Password: ${CYAN}mediabias123${NC} | DB: ${CYAN}media_bias${NC}"
