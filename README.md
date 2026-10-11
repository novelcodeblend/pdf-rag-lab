# PDF RAG Lab (Free, Local Model)

A beginner-friendly Retrieval-Augmented Generation (RAG) application for AI students.

## What students will learn

1. Extract text from PDF documents.
2. Split text into overlapping chunks.
3. Build a searchable TF-IDF index.
4. Retrieve relevant chunks for a question.
5. Pass retrieved context to a local language model.
6. Inspect retrieved source passages and page numbers.

**Important:** This is indexing/retrieval, not model fine-tuning. The LLM's weights are not trained on the uploaded PDF.

## Files

- `rag_core.py` — PDF parsing, chunking, retrieval, and Ollama generation.
- `streamlit_app.py` — browser-based interface.
- `requirements.txt` — Python dependencies.

## Free local setup (recommended for the lab)

### 1. Install prerequisites

- Install Python 3.10 or newer.
- Install Ollama from https://ollama.com/download.
- Install Git from https://git-scm.com/downloads (optional if uploading through GitHub's website).

### 2. Download a model

Open a terminal/Command Prompt and run:

```bash
ollama pull llama3.2:3b
```

Ollama should be running. If needed, start it in a separate terminal:

```bash
ollama serve
```

The model is downloaded once. Local inference has no per-token API fee, but requires disk space, memory, and compute on the machine running Ollama.

### 3. Install Python packages

Open a terminal in this project folder:

```bash
python -m venv .venv
```

Activate it:

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Then install packages:

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run streamlit_app.py
```

A browser tab will open. Upload a text-based PDF, click **Index PDF**, then ask a question.

## Upload to GitHub

1. Sign in to https://github.com.
2. Click **New repository**.
3. Name it `pdf-rag-lab` and create it.
4. Upload `rag_core.py`, `streamlit_app.py`, `requirements.txt`, and this `README.md` to the repository root.
5. Commit the files.

Do not upload private documents, API keys, passwords, or student records.

## Deploy the Streamlit frontend

Official guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

1. Push the files above to a GitHub repository.
2. Sign in at https://share.streamlit.io using your GitHub account.
3. Choose **Create app**.
4. Select the repository, branch, and `streamlit_app.py` as the main file.
5. Click **Deploy**.

### Important deployment limitation

Streamlit Community Cloud hosts the frontend, but it does **not** automatically run Ollama or download/run the LLM for you. The default `http://localhost:11434` on a cloud deployment points to the cloud app container, not your personal computer.

Therefore, the app will work immediately when run locally with Ollama. For a public Streamlit deployment, you need a reachable model endpoint that you control or a hosted inference service. Free hosted endpoints have usage limits and are not guaranteed to be unlimited. Do not expose an unauthenticated Ollama server directly to the public internet.

For a fully free classroom demo with no hosted inference service, run Streamlit and Ollama on the same teacher computer and share the local network URL with students, subject to your network/firewall settings.

## Troubleshooting

- **Cannot reach Ollama:** Start Ollama and check `http://localhost:11434` is reachable.
- **Model missing:** Run `ollama pull llama3.2:3b`.
- **No readable text:** The PDF may be a scanned image. This starter does not include OCR; try a text-based PDF or extend it with OCR.
- **Answers are not perfect:** TF-IDF is a simple lexical retriever. Try clearer questions, inspect source passages, and compare with embedding-based retrieval in a later lesson.

## Suggested lab exercises

1. Change chunk size and compare retrieved passages.
2. Change `top_k` and observe answer quality.
3. Ask a question whose answer is not in the PDF; check whether the model admits it.
4. Compare TF-IDF with a sentence-transformer embedding model.
5. Add OCR for scanned PDFs.
6. Add conversation memory and evaluate whether it helps or hurts factual grounding.
