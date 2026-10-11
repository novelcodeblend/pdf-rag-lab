"""
Simple Retrieval-Augmented Generation (RAG) core for a classroom lab.

Pipeline:
PDF -> text extraction -> chunks -> TF-IDF retrieval -> Ollama local LLM answer

This does NOT fine-tune/train model weights. It builds a searchable index over
the uploaded PDF and retrieves relevant passages at question time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
import requests
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    text: str
    page: int


class SimplePDFRAG:
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        chunk_size: int = 900,
        overlap: int = 150,
    ):
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.model = model
        self.chunk_size = max(200, int(chunk_size))
        self.overlap = max(0, min(int(overlap), self.chunk_size // 2))
        self.chunks: List[Chunk] = []
        self.vectorizer = None
        self.matrix = None

    @staticmethod
    def _extract_pages(pdf_bytes: bytes) -> List[tuple[int, str]]:
        from io import BytesIO
        reader = PdfReader(BytesIO(pdf_bytes))
        pages = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = " ".join(text.split())
            if text:
                pages.append((page_num, text))
        return pages

    def index_pdf(self, pdf_bytes: bytes) -> Dict[str, int]:
        """Extract PDF text and build a TF-IDF index. Returns simple stats."""
        pages = self._extract_pages(pdf_bytes)
        chunks: List[Chunk] = []
        step = self.chunk_size - self.overlap

        for page_num, text in pages:
            start = 0
            while start < len(text):
                part = text[start:start + self.chunk_size].strip()
                if part:
                    chunks.append(Chunk(text=part, page=page_num))
                if start + self.chunk_size >= len(text):
                    break
                start += step

        if not chunks:
            raise ValueError(
                "No readable text was found in this PDF. It may be scanned; "
                "try a text-based PDF or add OCR as an extension."
            )

        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])

        return {
            "pages_with_text": len(pages),
            "chunks": len(chunks),
            "characters": sum(len(text) for _, text in pages),
        }

    def retrieve(self, question: str, top_k: int = 4) -> List[dict]:
        if self.vectorizer is None or self.matrix is None or not self.chunks:
            raise RuntimeError("Upload and index a PDF before asking questions.")
        query_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = scores.argsort()[::-1][:max(1, min(top_k, len(self.chunks)))]
        results = []
        for idx in top_indices:
            results.append({
                "text": self.chunks[idx].text,
                "page": self.chunks[idx].page,
                "score": float(scores[idx]),
            })
        return results

    def ask(self, question: str, top_k: int = 4, temperature: float = 0.2) -> dict:
        retrieved = self.retrieve(question, top_k=top_k)
        context = "\n\n".join(
            f"[Source: page {item['page']}]\n{item['text']}" for item in retrieved
        )
        prompt = f"""You are a helpful assistant answering questions about an uploaded PDF.
Use ONLY the context below to answer. If the answer is not supported by the
context, say: "I couldn't find that information in the uploaded PDF."
Do not invent facts. Mention page numbers when useful.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

        response = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": float(temperature)},
            },
            timeout=180,
        )
        if response.status_code == 404:
            raise RuntimeError(
                f"Ollama/model not found. Check Ollama is running and run: "
                f"ollama pull {self.model}"
            )
        response.raise_for_status()
        data = response.json()
        return {
            "answer": data.get("response", "").strip(),
            "sources": retrieved,
            "model": self.model,
        }

    def check_ollama(self) -> tuple[bool, str]:
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            response.raise_for_status()
            names = [m.get("name", "") for m in response.json().get("models", [])]
            if self.model not in names and not any(n.startswith(self.model.split(":")[0]) for n in names):
                return False, f"Ollama is reachable, but model '{self.model}' is not listed. Pull it with: ollama pull {self.model}"
            return True, "Ollama is running."
        except Exception as exc:
            return False, f"Cannot reach Ollama at {self.ollama_base_url}. Is Ollama installed and running? Details: {exc}"
