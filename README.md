# SMART-RAG-GYM-AGENT
An end-to-end AI system that integrates image captioning, document ingestion, vector databases, and LLMs to deliver grounded, personalized fitness recommendations using Retrieval-Augmented Generation (RAG).

## Live Demo
https://huggingface.co/spaces/YUKTA2000/SMART-RAG-GYM-AGENT

## 0) Prerequisites (install first)

### Required
- **Python 3.10+** (recommended: 3.10 or 3.11)
- **Git** (optional but recommended)
- **Docker Desktop** (optional, only if you want Docker)
  - Make sure Docker Desktop is **running**
  - WSL2 enabled (Windows)


<<<<<<< HEAD
=======
## Clone repository

>>>>>>> c1847f9fc2e1fc7ed10b4db556a30cdc50a130d5
git clone <YOUR_REPO_URL>
cd "D:\UIC\Personal Projects\Smart RAG Gym Agent"

## 2) Create an env file

Create a .env file in the project root
OPENAI_API_KEY=your_key_here

## 3) Create & Activate Python Environment

python -m venv env_rag
env_rag\Scripts\activate
pip install -r requirements.txt
python app.py

For Conda:
conda create -n env_rag python=3.10
conda activate env_rag
pip install -r requirements.txt
python app.py

## 4) Run with Docker

docker compose up --build
