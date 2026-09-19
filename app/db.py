import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "microrag.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
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

    cursor.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5 ( 
        content,
        file_name UNINDEXED,
        chunk_id UNINDEXED,
        tokenize = 'porter unicode61' 
    );
    """)

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