import os
import sqlite3
from pathlib import Path

# Vercel serverless containers only allow writing inside the /tmp directory
if os.environ.get("VERCEL"):
    DB_PATH = Path("/tmp/microrag.db")
else:
    DB_PATH = Path(__file__).parent.parent / "microrag.db"


def get_db_connection():
    # Ensure directory exists if path is modified locally
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # WAL mode requires write access to sidecar files (-wal, -shm).
    # Only enable WAL outside of Vercel to avoid disk permission errors.
    if not os.environ.get("VERCEL"):
        conn.execute("PRAGMA journal_mode = WAL;")
        
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS document_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Vercel's AWS Lambda Python binaries often lack FTS5 support.
    # Catching OperationalError prevents the entire app initialization from crashing.
    try:
        cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5 ( 
            content,
            file_name UNINDEXED,
            chunk_id UNINDEXED,
            tokenize = 'porter unicode61' 
        );
        """)
    except sqlite3.OperationalError as e:
        print(f"Warning: FTS5 virtual table initialization skipped: {e}")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS query_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT NOT NULL,
        retrieval_time_ms REAL NOT NULL,
        total_latency_ms REAL NOT NULL,
        chunks_retrieved INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cursor.execute("PRAGMA table_info(query_logs);")
    columns = [column[1] for column in cursor.fetchall()]
    
    if "chunk_retrieved" in columns and "chunks_retrieved" not in columns:
        cursor.execute("ALTER TABLE query_logs RENAME COLUMN chunk_retrieved TO chunks_retrieved;")

    conn.commit()
    conn.close()