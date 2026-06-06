# tests/test_document_loader.py
import os
import tempfile

import pytest

from atlas.document_loader import (
    DocumentLoadError,
    chunk_text,
    detect_format_from_content_type,
    detect_format_from_extension,
    extract_text_from_txt,
    load_from_local,
    sanitize_text,
)


def test_sanitize_text_normalizes_whitespace():
    raw = "  Hello   world \r\n\r\n  Second   line  "
    cleaned = sanitize_text(raw)
    assert cleaned == "Hello world\n\nSecond line"


def test_chunk_text_respects_word_boundaries():
    text = "one two three four five six seven eight nine ten"
    chunks = chunk_text(text, max_chunk_size=15)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 20
        assert ' ' in chunk or len(chunk.split()) == 1


def test_detect_format_from_extension():
    assert detect_format_from_extension('report.pdf') == '.pdf'
    assert detect_format_from_extension('notes.DOCX') == '.docx'
    assert detect_format_from_extension('readme.md') == '.md'


def test_detect_format_from_extension_unsupported():
    with pytest.raises(DocumentLoadError):
        detect_format_from_extension('archive.zip')


def test_detect_format_from_content_type():
    assert detect_format_from_content_type('application/pdf') == '.pdf'
    assert detect_format_from_content_type('text/plain; charset=utf-8') == '.txt'
    assert detect_format_from_content_type('application/octet-stream') is None


def test_extract_text_from_txt(tmp_path):
    file_path = tmp_path / 'sample.txt'
    file_path.write_text('Line one\n\nLine two\n', encoding='utf-8')
    text = extract_text_from_txt(str(file_path))
    assert text == "Line one\n\nLine two"


def test_load_from_local_txt(tmp_path):
    file_path = tmp_path / 'notes.txt'
    file_path.write_text('Atlas learns from documents quickly and reliably.', encoding='utf-8')
    text = load_from_local(str(file_path))
    assert 'Atlas learns from documents' in text


def test_load_from_local_missing_file():
    with pytest.raises(DocumentLoadError, match='File not found'):
        load_from_local('/tmp/does-not-exist-atlas-doc.txt')


def test_load_from_local_empty_txt(tmp_path):
    file_path = tmp_path / 'empty.txt'
    file_path.write_text('   \n  ', encoding='utf-8')
    with pytest.raises(DocumentLoadError, match='empty'):
        load_from_local(str(file_path))
