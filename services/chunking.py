"""Document parsing and recursive character chunking service."""

from typing import List, Dict, Any, Optional
import io
import re

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


class DocumentChunk:
    """Represents a chunk extracted from a source document."""
    def __init__(
        self,
        chunk_index: int,
        text: str,
        token_count: int,
        page_number: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.chunk_index = chunk_index
        self.text = text
        self.token_count = token_count
        self.page_number = page_number
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "text": self.text,
            "token_count": self.token_count,
            "page_number": self.page_number,
            "metadata": self.metadata,
        }


class RecursiveChunker:
    """Recursively splits text into chunks respecting semantic boundaries."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count based on whitespace & word boundaries (~4 chars/token)."""
        words = len(text.split())
        return max(1, int(words * 1.3))

    def split_text(self, text: str) -> List[str]:
        """Splits raw text into chunk strings of at most chunk_size tokens with overlap."""
        if not text or not text.strip():
            return []

        clean_text = text.replace("\r\n", "\n")
        raw_chunks = self._recursive_split(clean_text, self.separators)

        # Merge small slices into chunks up to chunk_size tokens, with chunk_overlap
        chunks = []
        current_chunk_pieces = []
        current_tokens = 0

        for piece in raw_chunks:
            piece_tokens = self._estimate_tokens(piece)
            if not piece.strip():
                continue

            if current_tokens + piece_tokens > self.chunk_size and current_chunk_pieces:
                joined = "".join(current_chunk_pieces).strip()
                if joined:
                    chunks.append(joined)

                # Keep overlap pieces
                overlap_pieces = []
                overlap_tokens = 0
                for p in reversed(current_chunk_pieces):
                    pt = self._estimate_tokens(p)
                    if overlap_tokens + pt <= self.chunk_overlap:
                        overlap_pieces.insert(0, p)
                        overlap_tokens += pt
                    else:
                        break
                current_chunk_pieces = overlap_pieces
                current_tokens = overlap_tokens

            current_chunk_pieces.append(piece)
            current_tokens += piece_tokens

        if current_chunk_pieces:
            joined = "".join(current_chunk_pieces).strip()
            if joined:
                chunks.append(joined)

        return chunks if chunks else [text.strip()]

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Split text recursively using decreasing hierarchy of separators."""
        if not separators:
            return [text]

        sep = separators[0]
        rest = separators[1:]

        if sep == "":
            return list(text)

        splits = text.split(sep)
        result = []
        for i, s in enumerate(splits):
            piece = s if i == len(splits) - 1 else s + sep
            if self._estimate_tokens(piece) > self.chunk_size and rest:
                result.extend(self._recursive_split(piece, rest))
            else:
                result.append(piece)
        return result

    def chunk_document(
        self,
        content: str,
        filename: str = "",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Splits raw document text into structured DocumentChunk objects."""
        text_chunks = self.split_text(content)
        extra = extra_metadata or {}
        chunks = []
        for idx, text in enumerate(text_chunks):
            token_est = self._estimate_tokens(text)
            chunks.append(
                DocumentChunk(
                    chunk_index=idx,
                    text=text,
                    token_count=token_est,
                    page_number=extra.get("page_number"),
                    metadata={
                        "filename": filename,
                        **extra,
                    },
                )
            )
        return chunks

    def extract_and_chunk(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str = "",
    ) -> List[DocumentChunk]:
        """Extracts text from PDF, Markdown, or TXT file bytes and splits into chunks."""
        chunks: List[DocumentChunk] = []

        if filename.lower().endswith(".pdf") or "pdf" in content_type.lower():
            if PdfReader is None:
                raise RuntimeError("pypdf is required to parse PDF files.")
            reader = PdfReader(io.BytesIO(file_bytes))
            chunk_global_idx = 0
            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    continue
                page_chunks = self.split_text(page_text)
                for c_text in page_chunks:
                    tokens = self._estimate_tokens(c_text)
                    chunks.append(
                        DocumentChunk(
                            chunk_index=chunk_global_idx,
                            text=c_text,
                            token_count=tokens,
                            page_number=page_idx + 1,
                            metadata={"filename": filename, "page": page_idx + 1},
                        )
                    )
                    chunk_global_idx += 1
        else:
            # UTF-8 text / Markdown fallback
            try:
                text_content = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text_content = file_bytes.decode("latin-1", errors="replace")
            chunks = self.chunk_document(text_content, filename=filename)

        return chunks
