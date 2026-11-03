import hashlib
import io
import os
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import BinaryIO, TypeVar

import numpy as np
from google import genai
from google.genai import errors, types
from pypdf import PdfReader

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
GENERATION_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
BATCH_SIZE = 80
MAX_BATCH_CHARS = 50_000
DIVERSITY_RATIO = 0.85

T = TypeVar("T")
StatusCallback = Callable[[str], None]

SYSTEM_INSTRUCTION = (
    "Você é um auditor técnico de documentos. Responda à pergunta do usuário "
    "utilizando EXCLUSIVAMENTE os trechos fornecidos no contexto. "
    "Detecte o idioma da pergunta do usuário e responda OBRIGATORIAMENTE "
    "nesse mesmo idioma, mesmo que os documentos estejam escritos em outro idioma. "
    "Se a informação não constar explicitamente nos trechos, informe (no idioma da pergunta) "
    "que não há dados suficientes nos documentos enviados para responder com segurança. "
    "Sempre fundamente cada afirmação citando a página exata (ex: [Página X] ou [Page X], "
    "incluindo o nome do documento quando houver mais de um arquivo)."
)


@dataclass(frozen=True)
class Chunk:
    source: str
    page: int
    text: str


class RAGEngine:
    def __init__(self, api_key: str, chunk_size: int = 2000, overlap: int = 250) -> None:
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self._file_cache: dict[tuple[str, str], tuple[list[Chunk], np.ndarray]] = {}

    @staticmethod
    def _call_with_retry(
        fn: Callable[[], T],
        max_retries: int = 5,
        on_status: StatusCallback | None = None,
    ) -> T:
        for attempt in range(max_retries):
            try:
                return fn()
            except errors.APIError as exc:
                is_last = attempt == max_retries - 1
                if exc.code == 429 and not is_last:
                    match = re.search(
                        r"(?:retry in |'retryDelay': ')([0-9.]+)s",
                        str(exc),
                        re.IGNORECASE,
                    )
                    wait_seconds = min(float(match.group(1)) + 4.0 if match else 62.0, 95.0)
                    if on_status:
                        on_status(
                            f"Limite gratuito por minuto da API atingido. Retomando em {int(wait_seconds)}s..."
                        )
                    time.sleep(wait_seconds)
                elif exc.code in {500, 502, 503, 504} and not is_last:
                    delay = 2.0 * (2**attempt)
                    if on_status:
                        on_status(f"Servidor ocupado (HTTP {exc.code}). Nova tentativa em {int(delay)}s...")
                    time.sleep(delay)
                else:
                    raise
        raise RuntimeError("Limite de tentativas da API excedido.")

    @staticmethod
    def _make_batches(texts: list[str]) -> list[list[str]]:
        batches: list[list[str]] = []
        current: list[str] = []
        char_count = 0
        for text in texts:
            if current and (
                len(current) >= BATCH_SIZE or char_count + len(text) > MAX_BATCH_CHARS
            ):
                batches.append(current)
                current, char_count = [], 0
            current.append(text)
            char_count += len(text)
        if current:
            batches.append(current)
        return batches

    @staticmethod
    def _split_text(text: str, size: int, overlap: int) -> list[str]:
        cleaned = " ".join(text.split())
        if len(cleaned) <= size:
            return [cleaned] if cleaned else []

        pieces: list[str] = []
        start, n = 0, len(cleaned)
        while start < n:
            end = min(start + size, n)
            if end < n:
                split_pos = cleaned.rfind(" ", start + size // 2, end)
                if split_pos != -1:
                    end = split_pos
            piece = cleaned[start:end].strip()
            if piece:
                pieces.append(piece)
            if end >= n:
                break
            next_start = max(start + 1, end - overlap)
            space_after = cleaned.find(" ", next_start, end)
            start = space_after + 1 if space_after != -1 else next_start
        return pieces

    def _embed_texts(
        self,
        texts: list[str],
        task_type: str,
        on_status: StatusCallback | None = None,
    ) -> np.ndarray:
        vectors: list[list[float]] = []
        cfg = types.EmbedContentConfig(task_type=task_type)
        batches = self._make_batches(texts)
        for idx, batch in enumerate(batches, start=1):
            if on_status and len(batches) > 1:
                on_status(f"Gerando embeddings: lote {idx}/{len(batches)}...")
            res = self._call_with_retry(
                lambda b=batch: self.client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=b,
                    config=cfg,
                ),
                on_status=on_status,
            )
            vectors.extend(item.values for item in res.embeddings)
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-12, None)

    def index_pdfs(
        self,
        files: list[tuple[str, BinaryIO]],
        on_status: StatusCallback | None = None,
    ) -> int:
        all_chunks: list[Chunk] = []
        all_matrices: list[np.ndarray] = []

        for filename, stream in files:
            stream.seek(0)
            raw_bytes = stream.read()
            cache_key = (filename, hashlib.sha256(raw_bytes).hexdigest())

            if cache_key in self._file_cache:
                cached_chunks, cached_matrix = self._file_cache[cache_key]
                all_chunks.extend(cached_chunks)
                all_matrices.append(cached_matrix)
                continue

            if on_status:
                on_status(f"Extraindo texto de {filename}...")
            reader = PdfReader(io.BytesIO(raw_bytes))
            file_chunks: list[Chunk] = []
            for page_num, page in enumerate(reader.pages, start=1):
                raw_text = page.extract_text() or ""
                for piece in self._split_text(raw_text, self.chunk_size, self.overlap):
                    file_chunks.append(Chunk(source=filename, page=page_num, text=piece))

            if not file_chunks:
                continue

            file_matrix = self._embed_texts(
                [c.text for c in file_chunks],
                task_type="RETRIEVAL_DOCUMENT",
                on_status=lambda msg, fn=filename: on_status(f"[{fn}] {msg}")
                if on_status
                else None,
            )
            self._file_cache[cache_key] = (file_chunks, file_matrix)
            all_chunks.extend(file_chunks)
            all_matrices.append(file_matrix)

        if not all_chunks:
            raise ValueError("Nenhum texto selecionável foi encontrado nos PDFs enviados.")

        self.chunks = all_chunks
        self.embeddings = np.vstack(all_matrices)
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = 6) -> list[tuple[Chunk, float]]:
        if self.embeddings is None or not self.chunks:
            return []

        query_vec = self._embed_texts([query], task_type="RETRIEVAL_QUERY")[0]
        scores = self.embeddings @ query_vec
        k = min(top_k, len(self.chunks))
        ranked_idx = np.argsort(scores)[::-1]

        max_score = float(scores[ranked_idx[0]])
        threshold = max_score * DIVERSITY_RATIO if max_score > 0 else max_score - 0.10

        by_source: dict[str, deque[int]] = {}
        for idx in ranked_idx:
            if scores[idx] < threshold:
                break
            pos = int(idx)
            by_source.setdefault(self.chunks[pos].source, deque()).append(pos)

        selected: list[int] = []
        seen: set[int] = set()
        queues = list(by_source.values())
        while queues and len(selected) < k:
            next_queues: list[deque[int]] = []
            for q in queues:
                pos = q.popleft()
                selected.append(pos)
                seen.add(pos)
                if len(selected) == k:
                    break
                if q:
                    next_queues.append(q)
            queues = next_queues

        if len(selected) < k:
            for idx in ranked_idx:
                pos = int(idx)
                if pos not in seen:
                    selected.append(pos)
                    if len(selected) == k:
                        break

        return [(self.chunks[i], float(scores[i])) for i in selected]

    def answer(self, query: str, top_k: int = 6) -> tuple[str, list[tuple[Chunk, float]]]:
        retrieved = self.retrieve(query, top_k=top_k)
        if not retrieved:
            return "Nenhum documento indexado.", []

        blocks = [
            f"[Documento: {c.source} | Página {c.page}]\n{c.text}" for c, _ in retrieved
        ]
        joined_context = "\n\n---\n\n".join(blocks)
        prompt = (
            f"CONTEXTO EXTRAÍDO DOS DOCUMENTOS:\n\n{joined_context}\n\n"
            f"PERGUNTA DO USUÁRIO:\n{query}\n\n"
            "Responda estritamente no mesmo idioma utilizado em PERGUNTA DO USUÁRIO."
        )
        gen_cfg = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.1,
        )
        try:
            res = self._call_with_retry(
                lambda: self.client.models.generate_content(
                    model=GENERATION_MODEL,
                    contents=prompt,
                    config=gen_cfg,
                ),
                max_retries=3,
            )
        except errors.APIError as exc:
            if exc.code not in {429, 500, 502, 503, 504}:
                raise
            res = self._call_with_retry(
                lambda: self.client.models.generate_content(
                    model=FALLBACK_MODEL,
                    contents=prompt,
                    config=gen_cfg,
                ),
                max_retries=3,
            )
        return res.text or "Não foi possível gerar uma resposta.", retrieved
