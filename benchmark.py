import time
import requests
import numpy as np

URL = "http://127.0.0.1:8000/api/query"
QUERIES = [
    "What storage engine does microRAG use?",
    "What is the average retrieval execution latency?",
    "What are the operational constraints?",
    "How are documents parsed during ingestion?"
]

retrieval_times = []
total_times = []

print("Starting benchmark run...")
session = requests.Session()

for i in range(20):
    query = QUERIES[i % len(QUERIES)]
    start = time.perf_counter()
    res = session.post(URL, json={"query": query, "top_k": 3}).json()
    elapsed = (time.perf_counter() - start) * 1000
    
    retrieval_times.append(res["metrics"]["retrieval_time_ms"])
    total_times.append(elapsed)

print("\n--- Benchmark Results (20 Requests) ---")
print(f"Avg Retrieval Latency : {np.mean(retrieval_times):.2f} ms")
print(f"P95 Retrieval Latency : {np.percentile(retrieval_times, 95):.2f} ms")
print(f"Avg Total Latency     : {np.mean(total_times):.2f} ms")