# 🩺 Medical RAG Chatbot — Complete Project Guide

> An end-to-end **Retrieval-Augmented Generation (RAG)** chatbot that answers medical questions by retrieving context from *The Gale Encyclopedia of Medicine* (PDF) and generating answers using the **LLaMA 3.1-8B** model via the **Groq** API. The entire pipeline — from PDF ingestion to a live Flask web interface — is containerised with Docker and automated through a Jenkins CI/CD pipeline that pushes to AWS ECR.

---

## Table of Contents

1. [What This Project Actually Does (The Big Picture)](#1-what-this-project-actually-does-the-big-picture)
2. [Tech Stack — Why Each Tool Was Chosen](#2-tech-stack--why-each-tool-was-chosen)
3. [Project Structure — Every File Explained](#3-project-structure--every-file-explained)
4. [The RAG Pipeline — Step-by-Step Deep Dive](#4-the-rag-pipeline--step-by-step-deep-dive)
   - [Step 1: Loading the PDF](#step-1-loading-the-pdf-pdf_loaderpy)
   - [Step 2: Chunking the Text](#step-2-chunking-the-text-pdf_loaderpy)
   - [Step 3: Generating Embeddings](#step-3-generating-embeddings-embeddingspy)
   - [Step 4: Building & Saving the FAISS Vector Store](#step-4-building--saving-the-faiss-vector-store-vector_storepy)
   - [Step 5: The Orchestrator Script](#step-5-the-orchestrator-script-data_loaderpy)
   - [Step 6: Loading the LLM](#step-6-loading-the-llm-llmpy)
   - [Step 7: Creating the QA Chain (Retrieval + LLM)](#step-7-creating-the-qa-chain-retrieval--llm-retrieverpy)
   - [Step 8: The Flask Web App](#step-8-the-flask-web-app-applicationpy)
5. [Configuration & Environment Variables](#5-configuration--environment-variables)
6. [Logging & Custom Exception Handling](#6-logging--custom-exception-handling)
7. [The Frontend — Chat UI](#7-the-frontend--chat-ui)
8. [How to Run the Project Locally (Step by Step)](#8-how-to-run-the-project-locally-step-by-step)
9. [Docker — Containerising the App](#9-docker--containerising-the-app)
10. [CI/CD — Jenkins Pipeline + AWS ECR](#10-cicd--jenkins-pipeline--aws-ecr)
11. [End-to-End Data Flow (Request Lifecycle)](#11-end-to-end-data-flow-request-lifecycle)
12. [Common Errors & Troubleshooting](#12-common-errors--troubleshooting)
13. [Key Design Decisions & Notes](#13-key-design-decisions--notes)

---

## 1. What This Project Actually Does (The Big Picture)

Traditional chatbots either generate hallucinated answers (pure LLMs without context) or can only return pre-written responses. This project solves that by combining **retrieval** and **generation**:

```
User asks: "What is Achalasia?"
          │
          ▼
┌─────────────────────────────┐
│  1. Question gets embedded  │  (sentence-transformers/all-MiniLM-L6-v2)
│  2. FAISS searches the      │  (finds the top-1 most relevant chunk)
│     vector store for the    │
│     most similar chunk      │
│  3. That chunk + question   │  (stuffed into a prompt template)
│     go to LLaMA 3.1-8B     │
│  4. LLM generates answer    │  (via Groq API — ultra-fast inference)
│     using ONLY the context  │
└─────────────────────────────┘
          │
          ▼
Bot says: "Achalasia is a disorder of the esophagus where..."
```

**In plain English:** The PDF is pre-processed and stored as vectors. When you ask a question, the system finds the most relevant paragraph from the PDF and asks the LLM to answer *only* from that paragraph. This prevents hallucination — the LLM can't make things up because it's constrained to the provided context.

---

## 2. Tech Stack — Why Each Tool Was Chosen

| Layer | Technology | Why This One? |
|---|---|---|
| **LLM** | LLaMA 3.1-8B Instant (via Groq) | Fast, free-tier friendly, open-source model. Groq provides near real-time inference. |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) | Lightweight (80MB), fast on CPU, produces 384-dim vectors. Great for semantic search without a GPU. |
| **Vector Database** | FAISS (Facebook AI Similarity Search) | Local, no server needed, blazing fast for similarity search. Stores vectors on disk as `.faiss` + `.pkl` files. |
| **Orchestration** | LangChain | Glues together the PDF loader, text splitter, embeddings, vector store, prompt template, and LLM into a single `RetrievalQA` chain. |
| **Web Framework** | Flask | Minimal, lightweight. Perfect for a single-page chatbot UI. Uses Jinja2 templates and server-side sessions. |
| **PDF Parsing** | PyPDF (via LangChain's `PyPDFLoader`) | Reliable, pure-Python PDF reader. No system-level dependencies. |
| **Containerisation** | Docker | Ensures the app runs identically in dev, CI, and production. |
| **CI/CD** | Jenkins | Automates build → scan → push. Uses Trivy for container security scanning. |
| **Cloud** | AWS ECR (Elastic Container Registry) | Stores Docker images. Optionally deploys to AWS App Runner. |
| **Environment** | python-dotenv | Loads API keys from `.env` so secrets never get hardcoded. |

---

## 3. Project Structure — Every File Explained

```
medical-chatbot/
│
├── .env                          # API keys (HF_TOKEN, GROQ_API_KEY) — NEVER commit this
├── .gitignore                    # Ignores: medvenv/, logs/, .egg-info/, .env
├── Dockerfile                    # Builds the production Docker image for the Flask app
├── Jenkinsfile                   # CI/CD pipeline: clone → build → scan → push to ECR
├── requirements.txt              # All Python dependencies with pinned versions
├── setup.py                      # Makes the project pip-installable as a package ("pip install -e .")
├── README.md                     # ← You are reading this file
│
├── data/
│   └── The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf   # 12MB medical PDF (759 pages)
│
├── vectorstore/
│   └── db_faiss/
│       ├── index.faiss           # The FAISS index file (~10.9MB) — the actual vectors
│       └── index.pkl             # Metadata + document chunks (~4.3MB) — maps vectors back to text
│
├── app/
│   ├── __init__.py               # Makes `app/` a Python package (empty file)
│   ├── application.py            # Flask web server — routes, session management, chat logic
│   │
│   ├── components/               # Core RAG pipeline modules
│   │   ├── pdf_loader.py         # Loads PDFs from data/ and splits them into text chunks
│   │   ├── embeddings.py         # Initialises the HuggingFace embedding model
│   │   ├── vector_store.py       # Saves/loads the FAISS vector store to/from disk
│   │   ├── data_loader.py        # Orchestrator: chains pdf_loader → embeddings → vector_store
│   │   ├── llm.py                # Loads the LLaMA 3.1-8B model via Groq API
│   │   └── retriever.py          # Builds the RetrievalQA chain: vector store + LLM + prompt
│   │
│   ├── common/                   # Shared utilities
│   │   ├── __init__.py           # Makes `common/` a Python package
│   │   ├── logger.py             # Configures file-based logging (logs/log_YYYY-MM-DD.log)
│   │   └── custom_exception.py   # Custom exception class with file/line info for debugging
│   │
│   ├── config/                   # Centralised configuration
│   │   ├── __init__.py           # Makes `config/` a Python package
│   │   └── config.py             # All settings: paths, API keys, chunk sizes, model names
│   │
│   └── templates/
│       └── index.html            # Jinja2 template — the chat UI (HTML + inline CSS)
│
├── custom_jenkins/
│   └── Dockerfile                # Custom Jenkins image with Docker-in-Docker support
│
├── logs/
│   └── log_2026-03-25.log        # Auto-generated daily log files
│
├── medvenv/                      # Python virtual environment (git-ignored)
│
└── RAG_Medcal_Chatbot.egg-info/  # Auto-generated by "pip install -e ." (git-ignored)
```

---

## 4. The RAG Pipeline — Step-by-Step Deep Dive

This is the core of the project. The pipeline has **two phases**:

- **Phase A — Ingestion (Offline/One-time):** PDF → Chunks → Embeddings → FAISS Vector Store on disk
- **Phase B — Inference (Every user query):** Question → Embed → FAISS Search → Top-1 Chunk → Prompt + LLM → Answer

Let's walk through every file involved.

---

### Step 1: Loading the PDF (`pdf_loader.py`)

**File:** `app/components/pdf_loader.py` — Function: `load_pdf_files()`

```python
loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
```

**What it does:**
- Uses LangChain's `DirectoryLoader` to scan the `data/` folder for all `*.pdf` files.
- Under the hood, each PDF is opened by `PyPDFLoader`, which extracts text page-by-page.
- The result is a list of `Document` objects. Each `Document` has:
  - `.page_content` — the raw text of one PDF page
  - `.metadata` — source file path and page number

**In this project:** The Gale Encyclopedia of Medicine (12MB, 759 pages) → produces **759 Document objects** (one per page).

**Error handling:** If the `data/` folder doesn't exist, it raises a `CustomException`. If no PDFs are found, it logs a warning and returns an empty list.

---

### Step 2: Chunking the Text (`pdf_loader.py`)

**File:** `app/components/pdf_loader.py` — Function: `create_text_chunks(documents)`

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)
```

**What it does:**
- Takes the 759 page-level documents and breaks them into smaller chunks.
- **`chunk_size=500`** — Each chunk is at most 500 characters long.
- **`chunk_overlap=50`** — Each chunk overlaps the previous by 50 characters. This ensures context isn't lost at chunk boundaries (e.g., a sentence split across two chunks will still appear fully in at least one).
- `RecursiveCharacterTextSplitter` first tries to split on `\n\n` (paragraphs), then `\n` (lines), then ` ` (words), then individual characters. This preserves natural text structure.

**Result:** 759 documents → **7,093 text chunks**.

**Why chunk?** Embedding models have a token limit, and smaller chunks give more precise retrieval (a 500-char chunk about "Achalasia" is more relevant than an entire 5,000-char page that also mentions 10 other diseases).

---

### Step 3: Generating Embeddings (`embeddings.py`)

**File:** `app/components/embeddings.py` — Function: `get_embedding_model()`

```python
model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

**What it does:**
- Downloads and loads the `all-MiniLM-L6-v2` model from HuggingFace.
- This model converts any text into a **384-dimensional vector** (a list of 384 numbers).
- Texts with similar *meanings* get vectors that are close together in vector space, even if the exact words are different ("heart attack" ≈ "myocardial infarction").

**Why this model?**
- Only ~80MB — runs fine on CPU.
- Produces high-quality embeddings for semantic search.
- No GPU required.

**Note:** This function is called in two places — once during ingestion (to embed chunks) and once during query time (to embed the user's question). The *same model* must be used in both places for the vectors to be comparable.

---

### Step 4: Building & Saving the FAISS Vector Store (`vector_store.py`)

**File:** `app/components/vector_store.py`

This file has **two functions**, used at different times:

#### `save_vector_store(text_chunks)` — Used during ingestion

```python
embedding_model = get_embedding_model()
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
```

**What it does:**
1. Gets the embedding model.
2. Takes all 7,093 text chunks and embeds each one into a 384-dim vector.
3. Builds a FAISS index from those vectors (an optimised data structure for fast nearest-neighbour search).
4. Saves two files to `vectorstore/db_faiss/`:
   - `index.faiss` (~10.9MB) — The actual FAISS index with all 7,093 vectors.
   - `index.pkl` (~4.3MB) — A pickle file mapping each vector back to its original text chunk + metadata.

#### `load_vector_store()` — Used during query time

```python
embedding_model = get_embedding_model()
return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
```

**What it does:**
- Loads the pre-built FAISS index from disk.
- `allow_dangerous_deserialization=True` is required because the `.pkl` file uses Python pickle (LangChain requires this flag for safety awareness).
- Returns a FAISS object that can be queried with `.similarity_search()` or used as a LangChain retriever.

---

### Step 5: The Orchestrator Script (`data_loader.py`)

**File:** `app/components/data_loader.py` — Function: `process_and_store_pdfs()`

```python
documents = load_pdf_files()        # Step 1: Load PDFs
text_chunks = create_text_chunks(documents)  # Step 2: Chunk text
save_vector_store(text_chunks)       # Steps 3+4: Embed + Save to FAISS
```

**What it does:**
- This is the **one-time ingestion script**. You run it once to process the PDF and build the vector store.
- It chains Steps 1 → 2 → 3 → 4 together.
- Run it as: `python -m app.components.data_loader`
- After this completes, the `vectorstore/db_faiss/` folder is populated and ready for queries.

**You only run this once** (or whenever you add/change PDFs in the `data/` folder).

---

### Step 6: Loading the LLM (`llm.py`)

**File:** `app/components/llm.py` — Function: `load_llm()`

```python
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=256,
)
```

**What it does:**
- Connects to **Groq's inference API** to use the `LLaMA 3.1-8B Instant` model.
- Groq is an LPU-based provider that runs LLMs extremely fast (hundreds of tokens/sec).
- **`temperature=0.3`** — Low randomness. For medical answers, we want factual, consistent responses, not creative ones.
- **`max_tokens=256`** — Limits response length. The prompt (in Step 7) asks for 2-3 line answers, so 256 tokens is plenty.
- Returns a LangChain-compatible LLM object that can be used in chains.

**Why Groq and not OpenAI?** Groq offers free-tier API access and runs inference remarkably fast. No GPU/local compute needed.

---

### Step 7: Creating the QA Chain — Retrieval + LLM (`retriever.py`)

**File:** `app/components/retriever.py`

This is where **retrieval** and **generation** come together.

#### The Prompt Template

```python
CUSTOM_PROMPT_TEMPLATE = """
Answer the following medical question in 2-3 lines maximum using only the information provided in the context.

Context:
{context}

Question:
{question}

Answer:
"""
```

**Why is this important?** This prompt *constrains* the LLM to:
1. Only use the provided context (prevents hallucination).
2. Keep answers concise (2-3 lines).
3. Answer in a medical Q&A format.

#### Building the Chain — `create_qa_chain()`

```python
db = load_vector_store()              # Load FAISS from disk
llm = load_llm()                      # Connect to Groq API

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",                 # "Stuff" = put all retrieved docs into the prompt
    retriever=db.as_retriever(search_kwargs={'k': 1}),  # Retrieve top-1 chunk
    return_source_documents=False,
    chain_type_kwargs={'prompt': set_custom_prompt()}
)
```

**Breaking this down:**

| Parameter | Value | What It Means |
|---|---|---|
| `chain_type="stuff"` | Stuff strategy | All retrieved chunks are *stuffed* (concatenated) directly into the prompt. Simple and effective when `k` is small. |
| `search_kwargs={'k': 1}` | Top-1 retrieval | Only the single most relevant chunk is retrieved. This keeps the prompt small and focused. |
| `return_source_documents=False` | No sources in output | The response only contains the answer, not the raw chunk that was used. |
| `chain_type_kwargs={'prompt': ...}` | Custom prompt | Uses the template above instead of LangChain's default. |

**The full flow when `qa_chain.invoke({"query": "What is Achalasia?"})` is called:**
1. The user's question is embedded using the same `all-MiniLM-L6-v2` model.
2. FAISS finds the chunk whose embedding is closest (cosine similarity) to the question embedding.
3. That chunk's text is inserted into `{context}` in the prompt template.
4. The complete prompt (context + question) is sent to LLaMA 3.1-8B via Groq.
5. The LLM generates an answer constrained to the provided context.
6. The answer is returned as `response["result"]`.

---

### Step 8: The Flask Web App (`application.py`)

**File:** `app/application.py`

This is the web server that serves the chat interface and handles user interactions.

#### App Setup

```python
load_dotenv()                          # Load .env file into environment variables
app = Flask(__name__)
app.secret_key = os.urandom(24)        # Required for Flask sessions (stores chat history)
```

#### Custom Jinja2 Filter

```python
def nl2br(value):
    return Markup(value.replace("\n", "<br>\n"))

app.jinja_env.filters['nl2br'] = nl2br
```

This filter converts newline characters (`\n`) in the LLM's response to HTML `<br>` tags so they render properly in the browser.

#### The Main Route — `GET /` and `POST /`

```python
@app.route("/", methods=["GET", "POST"])
def index():
```

**On GET request** (page load):
- Renders the chat template with any existing messages from the session.

**On POST request** (user submits a question):
1. Gets the user's input from the form.
2. Appends the user message to the session's message list.
3. Calls `create_qa_chain()` to build the retrieval chain.
4. Invokes the chain with the user's query.
5. Appends the assistant's response to the session.
6. Redirects to GET (PRG pattern — prevents duplicate submissions on refresh).
7. If anything fails, renders the page with an error message.

#### The Clear Route — `GET /clear`

```python
@app.route("/clear")
def clear():
    session.pop("messages", None)
    return redirect(url_for("index"))
```

Clears the chat history from the session and redirects back to the chat page.

#### Running the Server

```python
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
```

- `host="0.0.0.0"` — Listens on all network interfaces (required for Docker).
- `port=5000` — The Flask server runs on port 5000.
- `debug=False` — No debug mode in production.
- `use_reloader=False` — Prevents the server from starting twice (important because the embedding model takes time to load).

---

## 5. Configuration & Environment Variables

**File:** `app/config/config.py`

```python
import os

HF_TOKEN = os.environ.get("HF_TOKEN")                     # HuggingFace token
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")              # Groq API key

HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3" # (Legacy — not actively used)
DB_FAISS_PATH = "vectorstore/db_faiss"                      # Where FAISS files are stored
DATA_PATH = "data/"                                         # Where PDFs live
CHUNK_SIZE = 500                                            # Characters per chunk
CHUNK_OVERLAP = 50                                          # Characters of overlap between chunks
```

**The `.env` file** (at project root) must contain:
```env
HF_TOKEN="your_huggingface_token_here"
GROQ_API_KEY="your_groq_api_key_here"
```

| Variable | Where to Get It | What It's For |
|---|---|---|
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Downloading the embedding model from HuggingFace |
| `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) | Authenticating with Groq's LLM inference API |

---

## 6. Logging & Custom Exception Handling

### Logger (`app/common/logger.py`)

```python
LOG_FILE = os.path.join("logs", f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

logging.basicConfig(
    filename=LOG_FILE,
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
```

**What it does:**
- Creates a new log file every day (e.g., `logs/log_2026-03-25.log`).
- Every module calls `get_logger(__name__)` to get a logger named after the module.
- Logs are written to the file, not the console.
- Every key action is logged: model loading, chunking, FAISS operations, API calls, errors.

**Sample log output:**
```
2026-03-25 22:41:32,934 - INFO - Sucesfully fetched 759 documents
2026-03-25 22:41:33,446 - INFO - Generated 7093 text chunks
2026-03-25 22:41:47,096 - INFO - Use pytorch device_name: cpu
2026-03-25 22:47:37,317 - INFO - Saving vectorstoree
```

### Custom Exception (`app/common/custom_exception.py`)

```python
class CustomException(Exception):
    def __init__(self, message: str, error_detail: Exception = None):
        ...
    
    @staticmethod
    def get_detailed_error_message(message, error_detail):
        _, _, exc_tb = sys.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "Unknown File"
        line_number = exc_tb.tb_lineno if exc_tb else "Unknown Line"
        return f"{message} | Error: {error_detail} | File: {file_name} | Line: {line_number}"
```

**What it does:**
- Wraps any exception with extra context: the original error message, the file name, and the exact line number where it occurred.
- Makes debugging significantly easier — instead of a generic Python traceback, you get a single-line error like:
  ```
  Failed to load an LLM from Groq | Error: api_key not set | File: llm.py | Line: 12
  ```

---

## 7. The Frontend — Chat UI

**File:** `app/templates/index.html`

A minimal, functional chat interface built with plain HTML + inline CSS + Jinja2 templating:

- **Chat messages** are stored in Flask's server-side session and rendered in a loop.
- **User messages** have a blue background (`#d1e7ff`), aligned right.
- **Assistant messages** have a grey background (`#f1f1f1`), aligned left.
- **Error messages** appear in red.
- **Two forms:**
  - A textarea + "Send" button for submitting questions.
  - A "Clear Chat" button that hits the `/clear` route.

---

## 8. How to Run the Project Locally (Step by Step)

### Prerequisites
- Python 3.10+
- Git
- A HuggingFace account (free) → get your token
- A Groq account (free) → get your API key

### Step 1: Clone & Enter the Project

```bash
git clone https://github.com/codeWithOwaisAhmad/medical-chatbot.git
cd medical-chatbot
```

### Step 2: Create & Activate a Virtual Environment

```bash
# Windows
python -m venv medvenv
medvenv\Scripts\activate

# Linux/Mac
python3 -m venv medvenv
source medvenv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -e .
```

This reads `setup.py`, which reads `requirements.txt`, and installs everything:
- `langchain`, `langchain-community`, `langchain-huggingface`, `langchain-groq` — the RAG orchestration framework
- `faiss-cpu` — vector similarity search
- `pypdf` — PDF text extraction
- `sentence-transformers` — the embedding model
- `flask` — the web server
- `python-dotenv` — loads `.env` files

### Step 4: Set Up Environment Variables

Create a `.env` file at the project root:

```env
HF_TOKEN="hf_your_token_here"
GROQ_API_KEY="gsk_your_key_here"
```

### Step 5: Build the Vector Store (One-Time)

> **Skip this if `vectorstore/db_faiss/index.faiss` already exists.**

```bash
python -m app.components.data_loader
```

This takes 5-10 minutes on a CPU. It will:
1. Load the 759-page PDF.
2. Split it into 7,093 chunks.
3. Embed all chunks using `all-MiniLM-L6-v2`.
4. Save the FAISS index to `vectorstore/db_faiss/`.

### Step 6: Run the Flask Server

```bash
python app/application.py
```

### Step 7: Open the Chatbot

Go to **http://127.0.0.1:5000** in your browser and start asking medical questions!

---

## 9. Docker — Containerising the App

### Application Dockerfile (`Dockerfile`)

```dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*
COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 5000
CMD ["python", "app/application.py"]
```

**What each line does:**
1. **Base image:** Python 3.10 slim (minimal Debian variant).
2. **`PYTHONDONTWRITEBYTECODE=1`** — Don't create `.pyc` files (saves space).
3. **`PYTHONUNBUFFERED=1`** — Print logs immediately (don't buffer stdout/stderr).
4. **`WORKDIR /app`** — All subsequent commands run inside `/app`.
5. **System deps:** `build-essential` (for compiling some Python packages), `curl`.
6. **`COPY . .`** — Copies the entire project (code, data, vectorstore) into the container.
7. **`pip install -e .`** — Installs the package and all dependencies.
8. **`EXPOSE 5000`** — Documents that the container listens on port 5000.
9. **`CMD`** — Starts the Flask server when the container runs.

### Build & Run with Docker

```bash
# Build the image
docker build -t medical-chatbot .

# Run the container
docker run -p 5000:5000 --env-file .env medical-chatbot
```

Then visit **http://localhost:5000**.

---

## 10. CI/CD — Jenkins Pipeline + AWS ECR

### Custom Jenkins Image (`custom_jenkins/Dockerfile`)

A custom Jenkins image that includes Docker installed inside it (Docker-in-Docker). This lets Jenkins build Docker images as part of the pipeline.

Key additions over the base `jenkins/jenkins:lts`:
- Installs Docker CE, Docker CLI, and containerd.
- Adds the `jenkins` user to the `docker` group.
- Creates a volume for Docker data.

### The Pipeline (`Jenkinsfile`)

The Jenkins pipeline has **two stages**:

#### Stage 1: Clone GitHub Repo

```groovy
checkout scmGit(branches: [[name: '*/main']], ...)
```

Clones the `main` branch from GitHub using stored credentials.

#### Stage 2: Build, Scan, and Push Docker Image to ECR

```groovy
aws ecr get-login-password ... | docker login ...    # Authenticate with AWS ECR
docker build -t medical-rag:latest .                   # Build the Docker image
trivy image --severity HIGH,CRITICAL ... || true       # Security scan with Trivy
docker tag medical-rag:latest <account>.dkr.ecr...     # Tag for ECR
docker push <account>.dkr.ecr...                       # Push to ECR
```

1. **Authenticates** with AWS ECR using credentials stored in Jenkins.
2. **Builds** the Docker image.
3. **Scans** the image with **Trivy** (a container vulnerability scanner) for HIGH and CRITICAL severity issues. The `|| true` ensures the pipeline continues even if vulnerabilities are found (the report is archived as an artifact).
4. **Tags** the image with the ECR URL.
5. **Pushes** the image to AWS ECR.

#### Stage 3 (Commented Out): Deploy to AWS App Runner

There's a commented-out stage that would trigger a deployment to **AWS App Runner** — an AWS service that runs containers without managing infrastructure. It finds the App Runner service ARN and triggers a new deployment.

**Environment Variables in Jenkinsfile:**
| Variable | Value | Purpose |
|---|---|---|
| `AWS_REGION` | `eu-north-1` | AWS region for ECR/App Runner |
| `ECR_REPO` | `medical-rag` | ECR repository name |
| `IMAGE_TAG` | `latest` | Docker image tag |
| `SERVICE_NAME` | `llmops-medical-service` | App Runner service name |

---

## 11. End-to-End Data Flow (Request Lifecycle)

Here's what happens from the moment a user types a question to the answer appearing on screen:

```
┌──────────────────────────────────────────────────────────────────────┐
│ 1. User types "What is Achalasia?" and clicks Send                  │
│    └─→ Browser sends POST / with form data {prompt: "What is..."}  │
│                                                                      │
│ 2. Flask receives the POST request (application.py)                 │
│    └─→ Extracts user_input from request.form                        │
│    └─→ Appends {"role": "user", "content": "..."} to session        │
│                                                                      │
│ 3. create_qa_chain() is called (retriever.py)                       │
│    ├─→ load_vector_store() (vector_store.py)                        │
│    │   ├─→ get_embedding_model() loads all-MiniLM-L6-v2             │
│    │   └─→ FAISS.load_local() reads index.faiss + index.pkl         │
│    ├─→ load_llm() connects to Groq API (llm.py)                    │
│    └─→ RetrievalQA.from_chain_type() assembles the chain            │
│                                                                      │
│ 4. qa_chain.invoke({"query": "What is Achalasia?"})                 │
│    ├─→ Embed the question using all-MiniLM-L6-v2 → 384-dim vector  │
│    ├─→ FAISS searches for the top-1 nearest chunk                   │
│    ├─→ Chunk text inserted into the prompt template as {context}    │
│    ├─→ Full prompt sent to LLaMA 3.1-8B via Groq API               │
│    └─→ LLM returns answer: "Achalasia is a disorder of the..."     │
│                                                                      │
│ 5. Response appended to session messages as "assistant"              │
│    └─→ Flask redirects to GET / (PRG pattern)                       │
│                                                                      │
│ 6. GET / renders index.html with all messages from session          │
│    └─→ User sees the answer in the chat interface                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 12. Common Errors & Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `GROQ_API_KEY environment variable not set` | `.env` file missing or `load_dotenv()` not called | Create `.env` with `GROQ_API_KEY="gsk_..."` at the project root |
| `Vector store not present or empty` | The FAISS index hasn't been built yet | Run `python -m app.components.data_loader` first |
| `Failed to load PDF \| Data path doesnt exist` | The `data/` folder is missing or empty | Ensure the PDF is placed in `data/` |
| `ModuleNotFoundError: No module named 'app'` | Project not installed as a package | Run `pip install -e .` from the project root |
| `allow_dangerous_deserialization` error | FAISS is refusing to load the pickle file | Already handled in the code with `allow_dangerous_deserialization=True` |
| Slow first query (~10 seconds) | The embedding model is being downloaded/loaded for the first time | Normal — subsequent queries will be faster |
| `502 Bad Gateway` in Docker | Container crashed or port not mapped | Check logs with `docker logs <container_id>` and ensure `-p 5000:5000` |

---

## 13. Key Design Decisions & Notes

1. **Top-1 Retrieval (`k=1`):** Only one chunk is retrieved per query. This keeps the prompt small and focused, but may miss answers that span multiple chunks. For a production system, consider `k=3` with the "map_reduce" chain type.

2. **Stuff Chain Type:** All retrieved chunks are concatenated directly into the prompt. This works well with `k=1` but would hit token limits if `k` were increased significantly.

3. **No Chat Memory:** Each query is independent — the QA chain doesn't receive previous conversation history. The session stores messages for UI display only, not for the LLM.

4. **QA Chain Rebuilt Every Request:** `create_qa_chain()` is called on every POST request, which reloads the embedding model and FAISS index each time. For production, consider caching the chain as a global variable.

5. **CPU-Only:** The entire project runs on CPU. The embedding model (`all-MiniLM-L6-v2`) is small enough that GPU acceleration isn't necessary. The LLM runs on Groq's servers.

6. **Security Note:** API keys are loaded from `.env` and are git-ignored. The `.env` file should **never** be committed to version control. For Docker, pass secrets via `--env-file` or Docker secrets.

7. **The `setup.py` Makes the Project a Package:** Running `pip install -e .` installs the project in editable mode, which means `from app.components.retriever import ...` works from anywhere in the project. This is why all the imports use `app.` prefix.

---

> **That's it!** You now have a complete understanding of every part of this project — from how PDFs get chunked and embedded, to how FAISS finds the right context, to how the LLM generates constrained answers, to how Docker and Jenkins automate the entire deployment. 🚀