import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from pypdf import PdfReader
from docx import Document
from app.db import get_db_connection

DATA_DIR = Path(__file__).parent.parent / "data"

def extract_text_from_bytes(content: bytes, suffix: str) -> str:
    """Extract raw text from raw byte content based on extension."""
    suffix = suffix.lower()
    
    if suffix in [".txt", ".md"]:
        return content.decode("utf-8", errors="ignore")
    
    if suffix == ".pdf":
        import io
        reader = PdfReader(io.BytesIO(content))
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    
    if suffix == ".docx":
        import io
        doc = Document(io.BytesIO(content))
        return "\n".join([p.text for p in doc.paragraphs if p.text])
        
    return ""

def extract_text(file_path: Path) -> str:
    """Extract raw text from local files."""
    return extract_text_from_bytes(file_path.read_bytes(), file_path.suffix)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into character chunks with word-aware boundaries."""
    words = text.split()
    if not words:
        return []

    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1

        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))

            overlap_words = []
            overlap_len = 0
            for w in reversed(current_chunk):
                if overlap_len + len(w) > overlap:
                    break
                overlap_words.insert(0, w)
                overlap_len += len(w) + 1

            current_chunk = overlap_words
            current_length = overlap_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def save_chunks_to_db(file_name: str, chunks: list[str]) -> int:
    """Atomically store extracted chunks inside SQLite and FTS5 tables."""
    if not chunks:
        return 0

    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.cursor()
            
            
            doc_data = [(file_name, idx, chunk) for idx, chunk in enumerate(chunks)]
            
            
            for idx, chunk in enumerate(chunks):
                cursor.execute(
                    "INSERT INTO document_chunks (file_name, chunk_index, content) VALUES (?, ?, ?)",
                    (file_name, idx, chunk)
                )
                chunk_id = cursor.lastrowid
                
                cursor.execute(
                    "INSERT INTO fts_chunks (content, file_name, chunk_id) VALUES (?, ?, ?)",
                    (chunk, file_name, str(chunk_id))
                )
        return len(chunks)
    finally:
        conn.close()

def ingest_data_folder() -> int:
    """Parse files, generate chunks, and insert into SQLite and FTS5 tables."""
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)
        print(f"Created {DATA_DIR}. Drop files here to ingest.")
        return 0

    total_chunks = 0
    valid_extensions = {".txt", ".md", ".pdf", ".docx"}

    for file_path in DATA_DIR.glob("*.*"):
        if file_path.suffix.lower() not in valid_extensions:
            continue

        raw_text = extract_text(file_path)
        if not raw_text.strip():
            continue

        chunks = chunk_text(raw_text)
        inserted = save_chunks_to_db(file_path.name, chunks)
        total_chunks += inserted

    print(f"Ingestion complete. Processed {total_chunks} chunks.")
    return total_chunks


app = FastAPI()

@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in [".txt", ".md", ".pdf", ".docx"]:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    content = await file.read()
    raw_text = extract_text_from_bytes(content, ext)

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file contains no readable text")

    chunks = chunk_text(raw_text)
    total_inserted = save_chunks_to_db(file.filename, chunks)

    return {
        "status": "success",
        "filename": file.filename,
        "chunks_ingested": total_inserted
    }

if __name__ == "__main__":
    ingest_data_folder()