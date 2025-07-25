import os
import json
import re
import pymupdf as fitz  
import faiss
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llm_wrapper import call_llm 

def load_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text: str, chunk_size=500, overlap=50) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def build_faiss_index(chunks: List[str], model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings))
    return index, embeddings, chunks

def search_faiss(index, chunks: List[str], query: str, k=5):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb), k)
    return [chunks[i] for i in I[0]]

def extract_query_info(raw_query: str) -> dict:
    prompt = f"""
Extract the following structured information from the user's query:
Query: "{raw_query}"

Return ONLY a valid JSON like:
{{
  "age": 46,
  "gender": "male",
  "procedure": "knee surgery",
  "location": "Pune",
  "policy_duration_months": 3
}}
"""

    result = call_llm(prompt)
    print("\n[DEBUG] Gemini Response:\n", repr(result))
    json_text_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", result, re.DOTALL)
    if json_text_match:
        clean_json = json_text_match.group(1)
        return json.loads(clean_json)
    try:
        return json.loads(result)
    except json.JSONDecodeError as e:
        print("[ERROR] Gemini response could not be parsed as JSON.")
        print("Gemini returned:", result)
        raise e

def run_decision_engine(structured_query: dict, retrieved_clauses: List[str]) -> dict:
    context = "\n\n".join(retrieved_clauses)

    prompt = f"""
Context from policy clauses:
{context}

User Query (Structured):
{json.dumps(structured_query, indent=2)}

Using the above, determine:
- If the procedure is covered (decision: approved/rejected)
- Payout amount (if any)
- Justification based on exact clauses

Return ONLY a valid JSON like:
{{
  "decision": "approved",
  "amount": 25000,
  "justification": "Clause 5.2 states that ... (copy exact policy line)"
}}
"""

    result = call_llm(prompt)
    print("\n[DEBUG] Gemini Decision Response:\n", repr(result))
    import re
    try:
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", result, re.DOTALL)
        if json_match:
            result = json_match.group(1)

        parsed = json.loads(result)
        if parsed.get("decision", "").lower() == "approved":
            parsed["summary"] = "✅ Yes, the procedure is covered under the policy."
        else:
            parsed["summary"] = "❌ No, the procedure is not covered under the policy."

        return parsed

    except Exception as e:
        print("[ERROR] Failed to parse LLM decision output.")
        print(result)
        raise e

def main():
    pdf_path = "input_docs/BAJHLIP23020V012223.pdf" 
    text = load_text_from_pdf(pdf_path)

    
    chunks = chunk_text(text)
    index, _, chunk_store = build_faiss_index(chunks)

    
    raw_query = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"

   
    structured = extract_query_info(raw_query)
    print("\n[Structured Query]")
    print(json.dumps(structured, indent=2))
    relevant = search_faiss(index, chunk_store, raw_query)
    decision = run_decision_engine(structured, relevant)
    print("\n[Final Decision]")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()