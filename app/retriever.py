import time
from app.db import get_db_connection


def clean_fts_query(query: str) -> str:
    """Sanitize user query for SQLite FTS5 syntax safety"""
    sanitized = "".join(c for c in query if c.isalnum() or c.isspace())
    terms = sanitized.strip().split()

    if not terms:
        return '""'
    return " OR ".join(terms)


def retrieve_context(query: str, top_k: int = 3) -> dict:
    """
    Search SQLite FTS5 table using BM25 scoring.
    Returns matched content and retrieval execution time in milliseconds.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    formatted_query = clean_fts_query(query)
    start_time = time.perf_counter()

    cursor.execute("""
        SELECT chunk_id, file_name, content, bm25(fts_chunks) AS score
        FROM fts_chunks
        WHERE fts_chunks MATCH ?
        ORDER BY score
        LIMIT ?;
    """, (formatted_query, top_k))

    results = cursor.fetchall()
    retrieval_time_ms = (time.perf_counter() - start_time) * 1000
    conn.close()

    retrieved_chunks = [
        {
            "chunk_id": row["chunk_id"],
            "file_name": row["file_name"],
            "content": row["content"],
            "score": round(row["score"], 4)
        }
        for row in results
    ]

    return {
        "query": query,
        "retrieval_time_ms": round(retrieval_time_ms, 2),
        "results_count": len(retrieved_chunks),
        "chunks": retrieved_chunks
    }


if __name__ == "__main__":
    res = retrieve_context("database")
    
    print(f"Retrieval finished in {res['retrieval_time_ms']} ms")
    print(f"Found {res['results_count']} chunks")