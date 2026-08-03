# Linux (Ubuntu-like) Setup Guide

This guide provides step-by-step instructions to set up **Chatboton** on Ubuntu or similar Debian-based Linux distributions.

## Prerequisites

Before starting, ensure your system is up to date:
```bash
sudo apt update && sudo apt upgrade -y
```

## Step 1: Install System Dependencies

We need Git, Python 3, pip, and Docker.

### 1.1 Git
Git is required to clone the repository and manage versions.
```bash
sudo apt install -y git
```
*   **Role**: Versions control and repository management.
*   **Dependencies**: `libc6`, `libssl3`, `zlib1g`.

### 1.2 Python and Venv
Python is the core language of the project. We use `venv` to isolate dependencies.
```bash
sudo apt install -y python3 python3-pip python3-venv
```
*   **Role**: Executes the agent logic, FastAPI server, and seeding scripts.
*   **Dependencies**: `python3-venv` is required to create isolated environments.

### 1.3 Docker and Docker Compose
Docker runs our five databases (Postgres, Neo4j, Chroma, Qdrant, OpenSearch) in isolated containers.
```bash
# Install Docker and Docker Compose plugin
sudo apt install -y docker.io docker-compose-v2
# Add your user to the docker group (optional, requires logout/login)
sudo usermod -aG docker $USER
```
*   **Role**: Orchestrates the multi-database environment without manual installation of each DB.
*   **Dependencies**: `containerd`, `runc`, `iptables`.

### 1.4 Ollama
Ollama runs the LLM locally.
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
*   **Role**: Serves the `R4C3R/qwen2.5-3b-heretic` model for reasoning and `nomic-embed-text` for vector search.
*   **Dependencies**: Requires a modern Linux kernel (v4.18+ recommended) and optional NVIDIA/AMD GPU drivers for acceleration.

---

## Step 2: Prepare the LLM

Start Ollama and pull the required models.
```bash
# Start Ollama service (if not already running)
sudo systemctl start ollama

# Pull the base model and embedding model
ollama pull R4C3R/qwen2.5-3b-heretic
ollama pull nomic-embed-text
```

Then, create the specialized "Chatboton" model with the custom tool-aware template:
```bash
ollama create chatboton-heretic -f ollama/Modelfile
```
*   **Role**: The custom Modelfile optimizes the model for tool-calling with our specific database tools.

---

## Step 3: Set Up Python Environment

Clone the repository (if you haven't) and enter the directory, then:
```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```
*   **Role**: `requirements.txt` contains libraries like `langchain` (agent logic), `fastapi` (UI), and specific DB drivers (`psycopg`, `neo4j`, `chromadb`, `qdrant-client`, `opensearch-py`).
*   **Note**: `psutil` (system memory/disk/CPU introspection for the `reality_check` tool) is included in `requirements.txt`. If installing manually or updating an existing environment, run:

```bash
pip install psutil
```

---

## Step 4: Start Databases

Use Docker Compose to launch all services.
```bash
docker-compose up -d
```
*   **What gets installed/started**:
    *   **Postgres (Port 55433)**: Stores relational product data (ID, name, price, stock).
    *   **Neo4j (Port 57475/57688)**: Stores the purchase graph (`Customer -> BOUGHT -> Product`).
    *   **Chroma (Port 58001)**: Stores product reviews as vectors for semantic search.
    *   **Qdrant (Port 58333)**: Advanced vector store for product similarity search.
    *   **OpenSearch (Port 59200)**: Keyword/Full-text search engine.

---

## Step 5: Seed Data

Populate the databases with demo "gadget store" data.
```bash
python scripts/seed.py
```
*   **Role**: Ensures all five databases have consistent data so the agent can perform cross-database queries.

---

## Step 6: Launch Chatboton

Run the FastAPI application.
```bash
uvicorn app.main:app --reload --port 8000
```
*   **Access**: Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## Troubleshooting (Linux)
- **Docker Permission Denied**: If you didn't add your user to the `docker` group, use `sudo docker-compose up -d`.
- **Port Conflicts**: Ensure ports 8000, 55433, 57475, 58001, 58333, and 59200 are free.
