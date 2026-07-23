# macOS Setup Guide

This guide provides step-by-step instructions to set up **Chatboton** on macOS using Homebrew.

## Prerequisites

Ensure you have [Homebrew](https://brew.sh/) installed. If not, run:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

## Step 1: Install System Dependencies

We need Git, Python 3, and a Docker desktop environment.

### 1.1 Git
macOS often comes with Git via Xcode Command Line Tools, but we recommend the Homebrew version.
```bash
brew install git
```
*   **Role**: Versions control and repository management.
*   **Dependencies**: `openssl`, `pcre2`.

### 1.2 Python
macOS usually comes with Python, but we recommend the Homebrew version for consistency.
```bash
brew install python
```
*   **Role**: Executes the agent logic, FastAPI server, and seeding scripts.
*   **Dependencies**: Includes `pip` and `venv`, depends on `openssl`, `sqlite`.

### 1.3 Docker Desktop / OrbStack
On macOS, Docker requires a virtualized environment.
```bash
# Install Docker Desktop (or use OrbStack for better performance)
brew install --cask docker
```
*   **Action**: After installation, open **Docker.app** from your Applications folder and wait for it to start.
*   **Role**: Orchestrates the five database containers (Postgres, Neo4j, Chroma, Qdrant, OpenSearch). Docker Desktop includes **Docker Compose**.
*   **Dependencies**: Requires macOS 11+ and at least 4GB of RAM.

### 1.4 Ollama
Ollama runs the LLM locally on your Mac (leveraging Apple Silicon GPU if available).
```bash
brew install ollama
```
*   **Role**: Serves the `R4C3R/qwen2.5-3b-heretic` model and `nomic-embed-text` embeddings.
*   **Dependencies**: Requires macOS 11+ (Big Sur) or later.

---

## Step 2: Prepare the LLM

Start the Ollama app, then pull the required models:
```bash
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

In your terminal, navigate to the project directory:
```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```
*   **Role**: Installs libraries for LangChain, FastAPI, and database connectors (`psycopg`, `neo4j`, `chromadb`, etc.).

---

## Step 4: Start Databases

Ensure Docker Desktop is running, then launch the services:
```bash
docker-compose up -d
```
*   **Services Started**:
    *   **Postgres**: Relational data (Port 55433).
    *   **Neo4j**: Graph data (Port 57475/57688).
    *   **Chroma**: Review vectors (Port 58001).
    *   **Qdrant**: Product similarity (Port 58333).
    *   **OpenSearch**: Full-text search (Port 59200).

---

## Step 5: Seed Data

Populate the databases with the demo gadget-store data:
```bash
python scripts/seed.py
```
*   **Role**: Synchronizes the state across all five databases for consistent agent testing.

---

## Step 6: Launch Chatboton

Run the FastAPI application.
```bash
uvicorn app.main:app --reload --port 8000
```
*   **Access**: Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## Troubleshooting (macOS)
- **Docker Not Found**: Ensure the Docker app is actually running in your menu bar.
- **Port Conflicts**: macOS sometimes uses port 5000 or 7000 for AirPlay/Control Center. Our ports (8000, 55433, etc.) are usually safe, but check if `uvicorn` fails to bind.
- **Architecture Issues**: If you are on Intel Mac, everything works the same. On Apple Silicon (M1/M2/M3), Docker and Ollama will automatically use ARM64 for better performance.
