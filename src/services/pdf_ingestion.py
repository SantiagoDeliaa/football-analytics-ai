from __future__ import annotations

from io import BytesIO
from typing import Any


def _extract_with_pypdf(payload: bytes) -> tuple[list[str], int]:
    try:
        from pypdf import PdfReader
    except Exception:
        return [], 0
    reader = PdfReader(BytesIO(payload))
    pages: list[str] = []
    for page in reader.pages:
        pages.append((page.extract_text() or "").strip())
    return pages, len(reader.pages)


def _extract_with_pypdf2(payload: bytes) -> tuple[list[str], int]:
    try:
        from PyPDF2 import PdfReader
    except Exception:
        return [], 0
    reader = PdfReader(BytesIO(payload))
    pages: list[str] = []
    for page in reader.pages:
        pages.append((page.extract_text() or "").strip())
    return pages, len(reader.pages)


def _extract_with_pdfplumber(payload: bytes) -> tuple[list[str], int]:
    try:
        import pdfplumber
    except Exception:
        return [], 0
    pages: list[str] = []
    with pdfplumber.open(BytesIO(payload)) as pdf:
        for page in pdf.pages:
            pages.append((page.extract_text() or "").strip())
        return pages, len(pdf.pages)


def ingest_pdf(uploaded_file: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "error",
        "file_name": getattr(uploaded_file, "name", "unknown.pdf"),
        "bytes_size": 0,
        "page_count": 0,
        "pages": [],
        "raw_text": "",
        "messages": [],
    }
    if uploaded_file is None:
        result["messages"] = ["No se recibió ningún archivo."]
        return result

    try:
        payload = uploaded_file.read()
    except Exception:
        result["messages"] = ["No fue posible leer el archivo cargado."]
        return result

    if not payload:
        result["messages"] = ["El archivo está vacío o no contiene datos."]
        return result

    result["bytes_size"] = len(payload)

    extractors = (
        ("pypdf", _extract_with_pypdf),
        ("PyPDF2", _extract_with_pypdf2),
        ("pdfplumber", _extract_with_pdfplumber),
    )
    pages: list[str] = []
    page_count = 0
    parser_used = ""

    for parser_name, extractor in extractors:
        try:
            parser_pages, parser_page_count = extractor(payload)
        except Exception:
            parser_pages, parser_page_count = [], 0
        normalized_pages = [text for text in parser_pages if text]
        if normalized_pages:
            pages = normalized_pages
            page_count = parser_page_count or len(normalized_pages)
            parser_used = parser_name
            break

    if not pages:
        fallback_text = payload.decode("utf-8", errors="ignore").strip()
        if fallback_text:
            pages = [fallback_text]
            page_count = 1
            parser_used = "utf8_fallback"

    if not pages:
        result["status"] = "warning"
        result["messages"] = [
            "No se pudo extraer texto utilizable del PDF.",
            "Revisa que el archivo no sea una imagen escaneada sin capa de texto.",
        ]
        return result

    result["page_count"] = page_count
    result["pages"] = [
        {"page_number": idx + 1, "char_count": len(text), "text": text}
        for idx, text in enumerate(pages)
    ]
    result["raw_text"] = "\n\n".join(pages)
    result["status"] = "ok"
    result["messages"] = [f"Extracción completada con {parser_used}."]
    return result
