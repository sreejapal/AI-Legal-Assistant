import os
import json
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


def build_vector_db(json_folder, db_path):
    """
    Build a FAISS vector DB from JSON files in a folder.
    
    Parameters:
    - json_folder: str, path to folder containing JSON files
    - db_path: str, path where the vector DB should be saved
    """
    # Load embeddings on GPU with cosine normalization
    embeddings = HuggingFaceEmbeddings(
        model_name="amixh/sentence-embedding-model-InLegalBERT-2",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )

    # Text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,  # you can reduce to 50 if desired
        separators=["\n\n", "\n", ".", " ", ""],
    )

    all_docs = []
    json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

    print(f"📂 Found {len(json_files)} JSON files in {json_folder}")

    # Step 1: Read & split files
    for file in tqdm(json_files, desc="🔄 Reading & splitting"):
        try:
            with open(os.path.join(json_folder, file), "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                for section, content in data.items():
                    if not isinstance(content, str):
                        continue
                    docs = splitter.create_documents(
                        [content],
                        metadatas=[{"source": file, "section": section}],
                    )
                    all_docs.extend(docs)
            else:
                text = json.dumps(data, indent=2)
                docs = splitter.create_documents([text], metadatas=[{"source": file}])
                all_docs.extend(docs)

        except Exception as e:
            print(f"⚠️ Error processing {file}: {e}")

    print(f"📑 Total chunks to embed: {len(all_docs)}")

    # Step 2: Embed & build FAISS index with progress bar
    db = None
    for i in tqdm(range(0, len(all_docs), 64), desc="⚡ Embedding & indexing (GPU)"):
        batch = all_docs[i:i + 64]
        if db is None:
            db = FAISS.from_documents(batch, embeddings)
        else:
            db.add_documents(batch)

    # Save DB
    db.save_local(db_path)
    print(f"\n✅ Vector DB created with {len(all_docs)} chunks at: {db_path}")


if __name__ == "__main__":
    build_vector_db(r"C:\minor_1\dataset\split_lawcases\lawcases_04", r"C:\minor_1\dataset\vector_db_lawcases_04")
    build_vector_db(r"C:\minor_1\dataset\split_lawcases\lawcases_05", r"C:\minor_1\dataset\vector_db_lawcases_05")
    build_vector_db(r"C:\minor_1\dataset\split_lawcases\lawcases_06", r"C:\minor_1\dataset\vector_db_lawcases_06")
    build_vector_db(r"C:\minor_1\dataset\split_lawcases\lawcases_07", r"C:\minor_1\dataset\vector_db_lawcases_07")
    build_vector_db(r"C:\minor_1\dataset\split_lawcases\lawcases_08", r"C:\minor_1\dataset\vector_db_lawcases_08")
    build_vector_db(r"C:\minor_1\dataset\split_lawcases\lawcases_09", r"C:\minor_1\dataset\vector_db_lawcases_09")
    build_vector_db(r"C:\minor_1\dataset\split_lawcases\lawcases_10", r"C:\minor_1\dataset\vector_db_lawcases_10")
    build_vector_db(r"C:\minor_1\dataset\split_lawcases\lawcases_11", r"C:\minor_1\dataset\vector_db_lawcases_11")
    build_vector_db(r"C:\minor_1\dataset\split_lawcases\lawcases_12", r"C:\minor_1\dataset\vector_db_lawcases_12")