from pathlib import Path

from docx import Document
from pypdf import PdfReader


class DocumentLoader:
    SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}

    def load(self, file_path: str | Path) -> str:
        file_path = Path(file_path)

        if not file_path.is_file():
            raise FileNotFoundError(f"Document file not found: {file_path}")

        extension = file_path.suffix.lower()

        if extension == ".txt":
            text = self._load_txt(file_path)
        elif extension == ".pdf":
            text = self._load_pdf(file_path)
        elif extension == ".docx":
            text = self._load_docx(file_path)
        else:
            raise ValueError(f"Unsupported document type: {extension}")

        text = text.strip()

        if not text:
            raise ValueError(f"No text was extracted from: {file_path}")

        return text

    def _load_txt(self, file_path: Path) -> str:
        return file_path.read_text(encoding="utf-8")

    def _load_pdf(self, file_path: Path) -> str:
        reader = PdfReader(file_path)

        pages = []

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                pages.append(page_text)

        return "\n".join(pages)

    def _load_docx(self, file_path: Path) -> str:
        document = Document(file_path)

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        return "\n".join(paragraphs)