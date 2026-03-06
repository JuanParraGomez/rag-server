from __future__ import annotations

import io
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader


class UnsupportedFileTypeError(ValueError):
    pass


class TextExtractor:
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

    def extract(self, filename: str, content: bytes) -> str:
        extension = Path(filename).suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(f"Unsupported file type: {extension}")

        if extension == ".txt":
            return self._from_txt(content)
        if extension == ".pdf":
            return self._from_pdf(content)
        if extension == ".docx":
            return self._from_docx(content)

        raise UnsupportedFileTypeError(f"Unsupported file type: {extension}")

    @staticmethod
    def _from_txt(content: bytes) -> str:
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("latin-1")

    @staticmethod
    def _from_pdf(content: bytes) -> str:
        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)

    @staticmethod
    def _from_docx(content: bytes) -> str:
        document = DocxDocument(io.BytesIO(content))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
