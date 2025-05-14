import os, zipfile, shutil, gzip
import numpy as np
import faiss
from utils.constants import CHUNK_SIZE, DIMENSION
from databases.db_connect import collection
from utils.logger import logger

index = faiss.IndexFlatL2(DIMENSION)
stored_chunks = []

def extract_log_content(path: str) -> str | None:
    try:
        if path.endswith(".gz"):
            with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as f:
                return f.read()
        else:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        return None

def fake_embed(text):
    np.random.seed(abs(hash(text)) % (10**8))
    return np.random.rand(DIMENSION).astype("float32")

def extract_and_save(uploaded_file):
    folder = "extracted_logs"
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

    file_ext = uploaded_file.filename.lower()
    temp_path = os.path.join(folder, uploaded_file.filename)

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(uploaded_file.file, f)

    logs = []

    try:
        if file_ext.endswith(".zip"):
            with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                zip_ref.extractall(folder)
            for root, _, files in os.walk(folder):
                for name in files:
                    path = os.path.join(root, name)
                    if name.endswith((".log", ".gz")):
                        content = extract_log_content(path)
                        if content:
                            logs.append({"filename": name, "content": content})
        elif file_ext.endswith((".gz", ".log")):
            content = extract_log_content(temp_path)
            if content:
                logs.append({"filename": uploaded_file.filename, "content": content})
        else:
            return False
    except Exception as e:
        logger.error(f"Extraction error: {e}")
        return False

    if logs:
        collection.delete_many({})
        collection.insert_many(logs)

    index.reset()
    stored_chunks.clear()
    for log in logs:
        chunks = [log["content"][i:i+CHUNK_SIZE] for i in range(0, len(log["content"]), CHUNK_SIZE)]
        for chunk in chunks:
            vec = fake_embed(chunk)
            index.add(np.array([vec]))
            stored_chunks.append((chunk, vec))

    return True
