import os
import time
import sqlite3
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pypdf import PdfReader
from docx import Document
import json
from app.db import get_db_connection, init_db
from app.llm import generate_rag_response
from app.retriever import retrieve_context

DATA_DIR = Path(__file__).parent.parent / "data"

app = FastAPI(
    title="microRAG API",
    description="Sub-15ms Local BM25 Retrieval System with SQLite FTS5",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    init_db()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3




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
    """Batch store extracted chunks inside SQLite and FTS5 tables."""
    if not chunks:
        return 0

    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.cursor()
            
            
            doc_rows = [(file_name, idx, chunk) for idx, chunk in enumerate(chunks)]
            cursor.executemany(
                "INSERT INTO document_chunks (file_name, chunk_index, content) VALUES (?, ?, ?)",
                doc_rows
            )
            
            
            first_id = cursor.lastrowid - len(chunks) + 1
            
            
            fts_rows = [
                (chunk, file_name, str(first_id + idx)) 
                for idx, chunk in enumerate(chunks)
            ]
            cursor.executemany(
                "INSERT INTO fts_chunks (content, file_name, chunk_id) VALUES (?, ?, ?)",
                fts_rows
            )
        return len(chunks)
    finally:
        conn.close()


def ingest_data_folder() -> int:
    """Parse files from /data, generate chunks, and insert into tables."""
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

    return total_chunks




@app.get("/api/status")
def read_status():
    """System health check endpoint."""
    return {
        "system": "microRAG API",
        "status": "online",
        "docs_url": "http://127.0.0.1:8000/docs",
    }


@app.post("/api/ingest")
def trigger_ingestion():
    """Reads documents from /data folder, chunks them, and updates FTS5 database."""
    try:
        total_chunks = ingest_data_folder()
        return {
            "status": "success",
            "message": f"Successfully ingested {total_chunks} chunks into SQLite FTS5 index.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """Handles direct file upload from UI."""
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
        "message": f"Successfully ingested {total_inserted} chunks from {file.filename}.",
        "filename": file.filename,
        "chunks_ingested": total_inserted
    }


@app.post("/api/query")
def process_query(request: QueryRequest):
    """Executes FTS5 BM25 match, fetches LLM response, and logs metrics."""
    total_start_time = time.perf_counter()

    retrieval_data = retrieve_context(query=request.query, top_k=request.top_k)

    if not retrieval_data["chunks"]:
        total_latency = round((time.perf_counter() - total_start_time) * 1000, 2)
        return {
            "query": request.query,
            "answer": "No relevant context found in database.",
            "metrics": {
                "retrieval_time_ms": retrieval_data["retrieval_time_ms"],
                "generation_time_ms": 0,
                "total_latency_ms": total_latency,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "retrieved_chunks": [],
        }

    llm_data = generate_rag_response(
        query=request.query, retrieved_chunks=retrieval_data["chunks"]
    )

    total_latency_ms = (time.perf_counter() - total_start_time) * 1000

    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO query_logs (query, retrieval_time_ms, total_latency_ms, chunks_retrieved)
        VALUES (?, ?, ?, ?)
    """,
        (
            request.query,
            retrieval_data["retrieval_time_ms"],
            round(total_latency_ms, 2),
            len(retrieval_data["chunks"]),
        ),
    )
    conn.commit()
    conn.close()

    return {
        "query": request.query,
        "answer": llm_data["answer"],
        "metrics": {
            "retrieval_time_ms": retrieval_data["retrieval_time_ms"],
            "generation_time_ms": llm_data["generation_time_ms"],
            "total_latency_ms": round(total_latency_ms, 2),
            "prompt_tokens": llm_data["prompt_tokens"],
            "completion_tokens": llm_data["completion_tokens"],
            "total_tokens": llm_data["total_tokens"],
        },
        "retrieved_chunks": retrieval_data["chunks"],
    }
@app.get("/api/logs")
def get_query_logs():
    """Fetches execution logs for UI table rendering and cleans query formatting."""
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, query, retrieval_time_ms, total_latency_ms, chunks_retrieved, created_at AS timestamp
        FROM query_logs
        ORDER BY id DESC
        LIMIT 10
    """)
    rows = cursor.fetchall()
    conn.close()

    logs = []
    for row in rows:
        item = dict(row)
        
        if item["query"] and item["query"].strip().startswith("{"):
            try:
                parsed = json.loads(item["query"])
                if isinstance(parsed, dict) and "query" in parsed:
                    item["query"] = parsed["query"]
            except Exception:
                pass
        logs.append(item)


    return logs


@app.get("/api/metrics")
def get_performance_metrics():
    """Calculates aggregate performance stats across executed queries."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            COUNT(*) as total_queries,
            AVG(retrieval_time_ms) as avg_retrieval_ms,
            AVG(total_latency_ms) as avg_total_ms
        FROM query_logs
    """)
    stats = cursor.fetchone()
    conn.close()

    return {
        "total_queries_logged": stats["total_queries"] or 0,
        "avg_retrieval_time_ms": round(stats["avg_retrieval_ms"] or 0, 2),
        "avg_total_latency_ms": round(stats["avg_total_ms"] or 0, 2),
    }


# --- UI ROUTE ---

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>microRAG | 10x Performance Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-900 text-slate-100 min-h-screen p-6 font-sans">
        <div class="max-w-5xl mx-auto space-y-6">
            
            <!-- Header -->
            <header class="border-b border-slate-700 pb-4 flex justify-between items-center">
                <div>
                    <h1 class="text-3xl font-bold text-sky-400">microRAG Dashboard</h1>
                    <p class="text-slate-400 text-sm">SQLite FTS5 BM25 + Groq LLM (Sub-15ms Target)</p>
                </div>
                <button onclick="fetchMetrics()" class="bg-slate-800 hover:bg-slate-700 text-xs px-3 py-2 rounded border border-slate-600">
                    🔄 Refresh System Metrics
                </button>
            </header>

            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                
                <!-- Left Column: Document Upload & Ingestion -->
                <div class="bg-slate-800 p-5 rounded-xl border border-slate-700 space-y-4">
                    <h2 class="text-lg font-semibold text-slate-200">1. Ingest Knowledge</h2>
                    <p class="text-xs text-slate-400">Upload a `.txt` file or run the default ingestion script to seed FTS5 chunks.</p>
                    
                    <form id="uploadForm" class="space-y-3">
                        <input type="file" id="fileInput" accept=".txt,.pdf,.docx,.md" class="block w-full text-xs text-slate-400 file:mr-2 file:py-2 file:px-3 file:rounded file:border-0 file:text-xs file:font-semibold file:bg-sky-500 file:text-white hover:file:bg-sky-600 cursor-pointer"/>
                        <button type="button" onclick="uploadFile()" class="w-full bg-slate-700 hover:bg-slate-600 text-xs py-2 rounded font-medium border border-slate-600">
                            📁 Upload Custom Document
                        </button>
                    </form>

                    <div class="relative flex py-1 items-center">
                        <div class="flex-grow border-t border-slate-700"></div>
                        <span class="flex-shrink mx-2 text-xs text-slate-500">OR</span>
                        <div class="flex-grow border-t border-slate-700"></div>
                    </div>

                    <button onclick="runDefaultIngest()" class="w-full bg-emerald-600 hover:bg-emerald-500 text-xs py-2 rounded font-medium transition">
                        ⚡ Ingest Default Sample Docs
                    </button>
                    
                    <div id="ingestStatus" class="text-xs text-emerald-400 font-mono hidden"></div>
                </div>

                <!-- Right Column: Search & Evaluation -->
                <div class="md:col-span-2 space-y-6">
                    <div class="bg-slate-800 p-5 rounded-xl border border-slate-700 space-y-4">
                        <h2 class="text-lg font-semibold text-slate-200">2. Query & Evaluate</h2>
                        
                        <div class="flex gap-2">
                            <input id="queryInput" type="text" placeholder="e.g., What storage engine does microRAG use?" 
                                   class="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-sky-500">
                            <button onclick="runQuery()" class="bg-sky-500 hover:bg-sky-600 px-5 py-2 text-sm rounded-lg font-semibold transition">
                                Search
                            </button>
                        </div>

                        <!-- Real-time Metrics Banner -->
                        <div id="metricsBar" class="hidden grid grid-cols-3 gap-2 bg-slate-900 p-3 rounded-lg border border-slate-700 text-center text-xs">
                            <div><span class="text-slate-500 block">Retrieval Latency</span><span id="retrievalMs" class="text-sky-400 font-mono font-bold text-sm">-</span></div>
                            <div><span class="text-slate-500 block">Generation Time</span><span id="generationMs" class="text-emerald-400 font-mono font-bold text-sm">-</span></div>
                            <div><span class="text-slate-500 block">Total End-to-End</span><span id="totalMs" class="text-purple-400 font-mono font-bold text-sm">-</span></div>
                        </div>

                        <!-- Generated Answer -->
                        <div id="answerBox" class="hidden bg-slate-900/80 p-4 rounded-lg border border-slate-700 space-y-1">
                            <span class="text-xs font-semibold text-sky-400 uppercase tracking-wider">AI Generated Response (Groq)</span>
                            <p id="answerText" class="text-slate-200 text-sm leading-relaxed"></p>
                        </div>

                        <!-- Retrieved Chunks Context -->
                        <div id="chunksBox" class="hidden space-y-2">
                            <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Retrieved Chunks (FTS5 BM25)</span>
                            <div id="chunksList" class="space-y-2"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- System Logs / Historical Performance -->
            <div class="bg-slate-800 p-5 rounded-xl border border-slate-700 space-y-3">
                <h2 class="text-md font-semibold text-slate-300">3. Execution Benchmark Report (`query_logs`)</h2>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs text-slate-400">
                        <thead class="bg-slate-900 text-slate-300">
                            <tr>
                                <th class="p-2">ID</th>
                                <th class="p-2">Query</th>
                                <th class="p-2">Retrieval (ms)</th>
                                <th class="p-2">Total Latency (ms)</th>
                                <th class="p-2">Chunks</th>
                                <th class="p-2">Timestamp</th>
                            </tr>
                        </thead>
                        <tbody id="metricsTableBody" class="divide-y divide-slate-700">
                            <tr><td colspan="6" class="p-2 text-center text-slate-500">Loading query history...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <script>
            document.addEventListener('DOMContentLoaded', fetchMetrics);

            async function uploadFile() {
                const fileInput = document.getElementById('fileInput');
                if (!fileInput.files[0]) return alert('Please select a file first.');
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                const status = document.getElementById('ingestStatus');
                status.classList.remove('hidden');
                status.innerText = 'Uploading and processing file...';

                try {
                    const res = await fetch('/api/ingest/file', { method: 'POST', body: formData });
                    const data = await res.json();
                    status.innerText = '✅ ' + (data.message || 'File ingested successfully!');
                } catch (e) {
                    status.innerText = '❌ Failed to upload file.';
                }
            }

            async function runDefaultIngest() {
                const status = document.getElementById('ingestStatus');
                status.classList.remove('hidden');
                status.innerText = 'Ingesting sample documents...';
                
                const res = await fetch('/api/ingest', { method: 'POST' });
                const data = await res.json();
                status.innerText = '✅ ' + data.message;
            }

            async function runQuery() {
                const query = document.getElementById('queryInput').value;
                if (!query) return;

                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, top_k: 3 })
                });
                const data = await res.json();

                document.getElementById('metricsBar').classList.remove('hidden');
                document.getElementById('retrievalMs').innerText = data.metrics.retrieval_time_ms + ' ms';
                document.getElementById('generationMs').innerText = data.metrics.generation_time_ms + ' ms';
                document.getElementById('totalMs').innerText = data.metrics.total_latency_ms + ' ms';

                document.getElementById('answerBox').classList.remove('hidden');
                document.getElementById('answerText').innerText = data.answer;

                document.getElementById('chunksBox').classList.remove('hidden');
                const list = document.getElementById('chunksList');
                list.innerHTML = '';
                data.retrieved_chunks.forEach(c => {
                    list.innerHTML += `<div class="bg-slate-900/60 p-3 rounded border border-slate-700/50 text-xs font-mono text-slate-300">
                        <span class="text-sky-400 font-bold">[Chunk ${c.chunk_id}] Score: ${c.score}</span> - ${c.content}
                    </div>`;
                });

                fetchMetrics();
            }

            async function fetchMetrics() {
                try {
                    const res = await fetch('/api/logs');
                    const logs = await res.json();
                    const tbody = document.getElementById('metricsTableBody');
                    tbody.innerHTML = '';

                    if (!logs || logs.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="6" class="p-2 text-center text-slate-500">No query logs recorded yet. Run a search to populate logs.</td></tr>';
                        return;
                    }

                    logs.forEach(m => {
                        tbody.innerHTML += `<tr class="border-b border-slate-800">
                            <td class="p-2">${m.id}</td>
                            <td class="p-2 font-medium text-slate-200">${m.query}</td>
                            <td class="p-2 font-mono text-sky-400">${m.retrieval_time_ms} ms</td>
                            <td class="p-2 font-mono text-purple-400">${m.total_latency_ms} ms</td>
                            <td class="p-2">${m.chunks_retrieved}</td>
                            <td class="p-2 text-slate-500">${m.timestamp || '-'}</td>
                        </tr>`;
                    });
                } catch(e) {
                    console.error("Failed to fetch logs:", e);
                }
            }
            window.addEventListener('DOMContentLoaded', () => {
                fetchMetrics();
            });
        </script>
    </body>
    </html>
    """