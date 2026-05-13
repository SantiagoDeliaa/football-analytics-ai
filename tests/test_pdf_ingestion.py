from __future__ import annotations

from io import BytesIO

import pytest

from src.services.pdf_ingestion import ingest_pdf


reportlab_canvas = pytest.importorskip("reportlab.pdfgen.canvas")
pytest.importorskip("pypdf")


class UploadStub:
    def __init__(self, name: str, payload: bytes) -> None:
        self.name = name
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


def _build_pdf_bytes(text: str) -> bytes:
    buffer = BytesIO()
    canvas = reportlab_canvas.Canvas(buffer)
    canvas.drawString(72, 720, text)
    canvas.save()
    return buffer.getvalue()


def test_ingest_pdf_extracts_text_with_real_parser():
    payload = _build_pdf_bytes("Attack shots 10 xg 1.5 final third 8")

    result = ingest_pdf(UploadStub("report.pdf", payload))

    assert result["status"] == "ok"
    assert result["parser_used"] in {"pypdf", "pdfplumber"}
    assert "Attack shots 10" in result["raw_text"]
    assert result["page_count"] == 1
