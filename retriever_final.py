# retriever_final.py
import os
import json
import math
from pathlib import Path
from typing import List, Tuple
import string
import textwrap
import re

import numpy as np
import torch
from tqdm import tqdm

# PDF (fpdf2)
from fpdf import FPDF
# HTML cleaning
import fitz  # PyMuPDF

# Langchain & FAISS
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Cross-encoder
from sentence_transformers import CrossEncoder

# -------------------------
# Configuration
# -------------------------
FAISS_DB_PATHS = [
    r"C:\minor_1\dataset\vector_db_lawcases_01",
    r"C:\minor_1\dataset\vector_db_lawcases_02",
    r"C:\minor_1\dataset\vector_db_lawcases_03",
    r"C:\minor_1\dataset\vector_db_lawcases_04",
    r"C:\minor_1\dataset\vector_db_lawcases_05",
    r"C:\minor_1\dataset\vector_db_lawcases_06",
    r"C:\minor_1\dataset\vector_db_lawcases_07",
    r"C:\minor_1\dataset\vector_db_lawcases_08",
    r"C:\minor_1\dataset\vector_db_lawcases_09",
    r"C:\minor_1\dataset\vector_db_lawcases_10",
    r"C:\minor_1\dataset\vector_db_lawcases_11",
    r"C:\minor_1\dataset\vector_db_lawcases_12",
]

JSON_PARENT_FOLDER = r"C:\minor_1\dataset\split_lawcases"

EMBEDDING_MODEL_NAME = "amixh/sentence-embedding-model-InLegalBERT-2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

EMBEDDING_DEVICE = "cuda"
CROSS_ENCODER_DEVICE = "cuda"

k_per_faiss_db = 20
top_k_merge = 50
top_k_rerank = 50
final_k = 10
embedding_batch_size = 64

PDF_OUTPUT_DIR = r"C:\minor_1\dataset\retrieved_pdfs"
os.makedirs(PDF_OUTPUT_DIR, exist_ok=True)

# -------------------------
# Utilities
# -------------------------
def preprocess_query(q: str) -> str:
    return " ".join(q.strip().lower().split())

def interpret_faiss_score(raw_score: float) -> float:
    if raw_score is None:
        return 0.0
    if raw_score > 1.5:  # likely L2 distance
        return 1.0 / (1.0 + raw_score)
    return float(max(0.0, min(1.0, raw_score)))

def load_full_case_json(case_id: str, parent_folder: str):
    for root, dirs, files in os.walk(parent_folder):
        if case_id in files:
            path = os.path.join(root, case_id)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load {case_id}: {e}")
                return None
    return None

def sanitize_text(text: str) -> str:
    # Keep printable ASCII only
    return "".join(c if c in string.printable else " " for c in text)

def remove_html_tags(text):
    """Removes HTML tags using regex."""
    clean = re.compile('<[^>]+>')
    return re.sub(clean, '', text)

def clean_text_content(text):
    """Clean text by removing HTML tags and other unwanted content."""
    # Remove HTML tags
    text = remove_html_tags(text)
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def json_to_pdf(json_data: dict, output_pdf_path: str, case_id: str = "", max_line_length=80):
    """
    Convert JSON data to PDF with recursive rendering for better structure.
    
    Parameters:
    - json_data: dict, the JSON data to convert
    - output_pdf_path: str, path where to save the PDF
    - case_id: str, optional case identifier for the header
    - max_line_length: int, maximum line length for text wrapping
    """
    # Create a PDF instance
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)  # Avoid deprecated Arial warning

    def add_line(text):
        # Clean HTML tags and unwanted content first
        cleaned_text = clean_text_content(text)
        # Sanitize text to handle Unicode characters that can't be encoded in Helvetica
        sanitized_text = "".join(c if c in string.printable else " " for c in cleaned_text)
        # Wrap text manually if it's too long
        lines = textwrap.wrap(sanitized_text, max_line_length)
        for line in lines:
            pdf.cell(0, 10, line)
            pdf.ln()

    # Add header if case_id is provided
    if case_id:
        pdf.set_font("Helvetica", size=14, style="B")
        add_line(f"Legal Case Document: {case_id}")
        pdf.set_font("Helvetica", size=12)
        add_line("=" * 60)

    # Check if json_data is None or empty
    if json_data is None:
        add_line("ERROR: No data available for this case.")
        pdf.output(output_pdf_path)
        return output_pdf_path

    # Recursively render JSON
    def render_json(obj, indent=0):
        spacing = "    " * indent
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    add_line(f"{spacing}{key}:")
                    render_json(value, indent + 1)
                else:
                    line = f"{spacing}{key}: {str(value)}"
                    add_line(line)
        elif isinstance(obj, list):
            for index, item in enumerate(obj):
                add_line(f"{spacing}- Item {index + 1}:")
                render_json(item, indent + 1)
        else:
            add_line(f"{spacing}{str(obj)}")

    # Render the JSON content
    try:
        render_json(json_data)
    except Exception as e:
        add_line(f"ERROR: Failed to render JSON data: {str(e)}")

    # Save the PDF
    pdf.output(output_pdf_path)
    print(f"PDF created at: {output_pdf_path}")
    return output_pdf_path

def save_json_to_pdf(case_id: str, json_data: dict, output_dir: str):
    """Wrapper function to maintain compatibility with existing code"""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{case_id}.pdf")
    return json_to_pdf(json_data, out_path, case_id)


# -------------------------
# Load FAISS DBs
# -------------------------
def load_faiss_dbs(db_paths: List[str], embeddings):
    dbs = []
    for p in db_paths:
        if not os.path.exists(p):
            print(f"Warning: FAISS path not found: {p}")
            continue
        try:
            db = FAISS.load_local(p, embeddings, allow_dangerous_deserialization=True)
            dbs.append((p, db))
            print(f"Loaded FAISS DB: {p}")
        except Exception as e:
            print(f"Failed loading FAISS DB {p}: {e}")
    return dbs

# -------------------------
# Main retrieval
# -------------------------
def hybrid_retrieve(query: str):
    q = preprocess_query(query)
    print(f"\nQuery (preprocessed): {q}\n")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": embedding_batch_size},
    )
    faiss_dbs = load_faiss_dbs(FAISS_DB_PATHS, embeddings)
    if not faiss_dbs:
        raise RuntimeError("No FAISS DBs loaded.")

    semantic_candidates = []
    for db_path, db in faiss_dbs:
        try:
            res = db.similarity_search_with_score(q, k=k_per_faiss_db)
            for doc, raw_score in res:
                src = doc.metadata.get("source") or Path(db_path).name
                sem_score = interpret_faiss_score(float(raw_score))
                case_id = doc.metadata.get("case_id") or src
                semantic_candidates.append((case_id, doc, sem_score))
        except Exception as e:
            print(f"Warning: semantic search failed on {db_path}: {e}")

    print(f"Collected {len(semantic_candidates)} semantic candidates from FAISS DBs.")

    sem_by_case = {}
    for case_id, doc, s in semantic_candidates:
        prev = sem_by_case.get(case_id)
        if (prev is None) or (s > prev["score"]):
            sem_by_case[case_id] = {"score": s, "doc": doc}

    candidate_case_ids = list(sem_by_case.keys())
    print(f"Total unique candidate cases: {len(candidate_case_ids)}")

    combined_list = [(cid, sem_by_case[cid]["score"]) for cid in candidate_case_ids]
    combined_list.sort(key=lambda x: x[1], reverse=True)
    top_candidates_list = combined_list[:top_k_merge]

    reranker = CrossEncoder(CROSS_ENCODER_MODEL, device=CROSS_ENCODER_DEVICE)
    rerank_inputs = []
    id_to_score = {}
    for case_id, sim_score in top_candidates_list[:top_k_rerank]:
        doc_text = sem_by_case[case_id]["doc"].page_content
        rerank_inputs.append((q, doc_text))
        id_to_score[case_id] = sim_score

    print(f"Reranking {len(rerank_inputs)} candidates with cross-encoder...")
    cross_scores = reranker.predict(rerank_inputs, batch_size=32)

    final_results = []
    for idx, (case_id, _) in enumerate(top_candidates_list[:top_k_rerank]):
        raw_ce = float(cross_scores[idx])
        ce_prob = torch.sigmoid(torch.tensor(raw_ce)).item()
        final_score = 0.5 * id_to_score[case_id] + 0.5 * ce_prob
        full_json = load_full_case_json(case_id, JSON_PARENT_FOLDER)
        final_results.append({
            "case_id": case_id,
            "similarity_score": id_to_score[case_id],
            "crossenc_logit": raw_ce,
            "crossenc_prob": ce_prob,
            "final_score": final_score,
            "full_json": full_json
        })

    final_results.sort(key=lambda x: x["final_score"], reverse=True)
    top_final = final_results[:final_k]

    print(f"\nTop results (absolute scoring, PDFs saved to): {PDF_OUTPUT_DIR}")
    for rank, r in enumerate(top_final, start=1):
        full_json = r["full_json"] or {"error": "JSON not found"}
        if rank <= 5:
            save_json_to_pdf(r["case_id"], full_json, PDF_OUTPUT_DIR)
        print(f"\nRank {rank}: Case file: {r['case_id']}")
        print(f"  Similarity score: {r['similarity_score']:.4f}")
        print(f"  CrossEnc logit:   {r['crossenc_logit']:.4f}")
        print(f"  CrossEnc prob:    {r['crossenc_prob']:.4f}")
        print(f"  Final score:      {r['final_score']:.4f}")

    return top_final

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    example_query =input("Enter your query: ")
    results = hybrid_retrieve(example_query)