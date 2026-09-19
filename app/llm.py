import os 
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))



def generate_rag_response(
    query: str, 
    retrieved_chunks: list[dict], 
    model: str = "openai/gpt-oss-20b"  
) -> dict:
    """
    Construct a prompt using retrieved context chunks and calls Groq LLM API.
    Returns response text, token usage, and generation latency.
    """

    context_str = "\n\n".join([
        f"[Source: {chunk['file_name']} | Chunk ID: {chunk['chunk_id']}]\n{chunk['content']}"
        for chunk in retrieved_chunks
    ])

   
    system_prompt = (
        "You are microRAG, a concise and precise retrieval assistant.\n"
        "Answer the user's question using ONLY the provided context below.\n"
        "If the answer cannot be found in the context, explicitly state 'Information not found in context.'\n"
        "Keep your response direct and factual."
    )
    user_prompt = f"Context:\n{context_str}\n\nQuestions: {query}"
    start_time = time.perf_counter()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )
    generation_time_ms = (time.perf_counter() - start_time) * 1000

    answer = response.choices[0].message.content
    usage = response.usage

    return {
        "answer": answer,
        "generation_time_ms": round(generation_time_ms, 2),
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0
    }