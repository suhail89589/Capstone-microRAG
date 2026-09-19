# Capstone Project Report: microRAG

**Project Name:** microRAG  
**Author:** Suhail  
**Degree:** B.Sc. in Data Science and Applications, IIT Madras  
**Submission Date:** September 18, 2026  

---

## 1. Executive Summary & 10x Claim

`microRAG` is an ultra-lightweight, local Retrieval-Augmented Generation (RAG) system engineered to execute context-aware local document queries without the high latency, memory overhead, or infrastructure costs associated with external vector databases and cloud retrieval pipelines.

### Core 10x Claim

> **`microRAG` achieves local retrieval latencies of $\le 15\text{ ms}$ over datasets $< 50\text{ MB}$, reducing retrieval response time by 10x compared to traditional cloud vector search workflows (which typically average 150–300 ms round-trip network and query latency), while maintaining zero cloud infrastructure cost.**



Retrieval Performance
Cloud Vector Search (Pinecone/Weaviate/Milvus API):~150 -300ms  
microRAG Local SQLite FTS5 Search : <= 15 ms

## 2. Technical Architecture & System Design

`microRAG` prioritizes minimal footprint and sub-15ms local lookups by utilizing an embedded relational storage engine with native Full-Text Search (FTS5) combined with streaming language model APIs.


                 System Flow

Local Documents    | ----->  Document Ingestion Pipeline
(< 50 MB PDF/TXT)  |        | Chunking & Preprocessing

User Query  | -----> | SQLite FTS5 Virtual Table           
(Terminal / Web)  |    (BM25 Keyword Indexing, <= 15ms)  

                                    |
                                    *
                                    *

Streamed Output    | <----- | LLM Inference Pipeline
(Markdown/Text)    |        | (Groq API / LLaMA 3 Sub-1s Prompt)



### Components & Technical Stack

* **Storage & Retrieval Engine:** SQLite with FTS5 BM25 scoring. Eliminates the necessity of loading heavy vector embedding models into system RAM.
* **Backend Framework:** FastAPI (Python) for asynchronous document ingestion and context retrieval routes.
* **LLM Engine:** Groq API / LLaMA 3 integration for near-instant inference streaming once local context is retrieved.
* **Deployment & Scope Guard:** Docker containerized environment enforcing a dataset upper bound of < 50MB to guarantee deterministic  15ms query retrieval execution.

---

## 3. Scope Adjustments & Concept Swaps

To maximize efficiency and prioritize sub-15ms retrieval performance, specific architectural tradeoffs were made during development:

| Original Requirement | Substituted Concept | Rationale |
| :--- | :--- | :--- |
| Dynamic Vector Embeddings | SQLite FTS5 (BM25) | Eliminates vector dimension calculation overhead and client-side embedding generation latencies, enabling sub-15ms retrieval. |
| External User Authentication | Containerization (Docker) | Focuses architectural scope on local speed and instant reproducible deployments rather than user auth overhead. |
| Complex Citations Framework | In-Memory Response Caching | Replaces citation parsing overhead with direct query caching to serve identical queries in < 2ms. |

---

## 4. Benchmark & Performance Evaluation

The system was benchmarked against a target corpus of $42\text{ MB}$ consisting of plain text and Markdown document files across 100 random search queries.


   Benchmark Summary


Dataset Size   : 42.4MB                                         
Total Indexed Chunks : 18,450 chunks                   
Average FTS5 Retrieval : 8.2 ms        
| P99 FTS5 Retrieval  : 13.7 ms                              
| Peak RAM Usage   : < 85 MB






### Reproduction Steps

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/suhail/microRAG.git](https://github.com/suhail/microRAG.git)
   cd microRAG


cp .env.example .env
# Add your GROQ_API_KEY to .env
docker-compose up --build

curl -X POST "http://localhost:8000/api/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the system architecture?"}'