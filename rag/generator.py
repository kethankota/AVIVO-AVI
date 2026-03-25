import os
import requests
import logging

logger = logging.getLogger(__name__)

ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
llm_model = os.environ.get("LLM_MODEL", "mistral:7b")
max_context_chars = int(os.environ.get("MAX_CONTEXT_CHARS", 3000))
max_history_turns = int(os.environ.get("MAX_HISTORY", 3))


def _ollama_generate(prompt: str) -> str:

    url = f"{ollama_host}/api/generate"
    payload = {
        "model": llm_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 512, 'num_ctx': 32000},
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.RequestException as e:
        logger.error("Ollama request failed: %s", e)
        raise RuntimeError(f"LLM unavailable: {e}") from e


def _build_rag_prompt(
        query: str,
        chunks: list[dict],
        history: list[dict],) -> str:
    context_parts = []
    total_chars = 0
    for chunk in chunks:
        if total_chars + len(chunk["text"]) > max_context_chars:
            break
        context_parts.append(f"[{chunk['source']}]\n{chunk['text']}")
        total_chars += len(chunk["text"])
    context = "\n\n---\n\n".join(
        context_parts) if context_parts else "No context found."

    history_lines = []
    for turn in history[-max_history_turns:]:
        if turn["type"] == "text":
            history_lines.append(
                f"User: {turn['input']}\nAssistant: {turn['output']}")
    history_block = ""
    if history_lines:
        history_block = "Previous conversation:\n" + \
            "\n\n".join(history_lines) + "\n\n"

    prompt = f"""{history_block}You are a helpful assistant. Answer ONLY using the context below.
    If the answer is not in the context, say "I don't have that information in my knowledge base."
    Keep your answer concise and accurate.
    
    Context:
    {context}
    
    Question: {query}
    Answer:"""
    return prompt


def generate(query: str, chunks: list[dict], history: list[dict]) -> dict:
    """
    Run RAG generation. Returns dict with keys: answer, source.
    """
    prompt = _build_rag_prompt(query, chunks, history)
    answer = _ollama_generate(prompt)

    source = None
    if chunks:
        top = chunks[0]
        excerpt = top["text"].replace("\n", " ").strip()
        source = f"{top['source']} — \"{excerpt}...\""

    return {"answer": answer, "source": source}


def summarize_history(history: list[dict]) -> str:
    max_history_turns = int(os.environ.get("MAX_HISTORY", 3))
    if not history:
        return "No history to summarise."

    lines = []
    for turn in history[-max_history_turns:]:
        if turn["type"] == "text":
            lines.append(f"- Asked: {turn['input'][:80]}")
        elif turn["type"] == "image":
            lines.append(f"- Sent an image. Caption: {turn['output'][:80]}")

    history_text = "\n".join(lines)
    prompt = f"""Summarise these recent interactions in 2–3 sentences:
    
    {history_text}
    
    Summary:"""
    return _ollama_generate(prompt)
