"""Dense text encoders. All return L2-normalised float32 matrices (dot product == cosine)."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol

import numpy as np
import requests


class Encoder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class WordLlamaEncoder:
    """Local static embeddings (WordLlama, MIT licence). Tiny, CPU-only, no API key: good for
    development, CI and as a cheap first-stage retriever. Swap for OpenAI/Ollama in production."""

    def __init__(self, dim: int = 256):
        from wordllama import WordLlama

        self._use_bundled_tokenizer()
        self.model = WordLlama.load(dim=dim)
        self.dim = dim
        self.name = f"wordllama-{dim}"

    @staticmethod
    def _use_bundled_tokenizer() -> None:
        # wordllama ships its tokenizer config inside the wheel but looks for it in the cache first and
        # otherwise downloads it; copying it into the cache makes loading work fully offline.
        import wordllama

        src = Path(wordllama.__file__).parent / "tokenizers"
        dst = Path.home() / ".cache" / "wordllama" / "tokenizers"
        if src.exists():
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.glob("*.json"):
                if not (dst / f.name).exists():
                    shutil.copy(f, dst / f.name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return _normalise(self.model.embed(texts))


class OpenAIEncoder:
    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None, batch: int = 128):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model, self.batch = model, batch
        self.name = f"openai:{model}"
        self.dim = 1536 if "small" in model else 3072

    def encode(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            resp = self.client.embeddings.create(model=self.model, input=texts[i:i + self.batch])
            out.extend(d.embedding for d in resp.data)
        return _normalise(np.array(out))


class OllamaEncoder:
    def __init__(self, model: str = "nomic-embed-text", url: str = "http://localhost:11434"):
        self.url, self.model = url.rstrip("/"), model
        self.name = f"ollama:{model}"
        self.dim = 0  # known after the first call

    def encode(self, texts: list[str]) -> np.ndarray:
        r = requests.post(f"{self.url}/api/embed", json={"model": self.model, "input": texts}, timeout=300)
        r.raise_for_status()
        m = _normalise(np.array(r.json()["embeddings"]))
        self.dim = m.shape[1]
        return m


def get_encoder(name: str = "wordllama") -> Encoder:
    if name.startswith("openai"):
        return OpenAIEncoder(name.split(":", 1)[1]) if ":" in name else OpenAIEncoder()
    if name.startswith("ollama"):
        return OllamaEncoder(name.split(":", 1)[1]) if ":" in name else OllamaEncoder()
    return WordLlamaEncoder()
