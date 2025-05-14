from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import MongoClient
import os, zipfile, shutil, uuid, gzip
import faiss
import numpy as np
import openai
import tiktoken
from openai.types.chat import ChatCompletionChunk

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB setup
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["logdb"]
collection = db["logs"]

# FAISS setup
dimension = 384
index = faiss.IndexFlatL2(dimension)
stored_chunks = []

CHUNK_SIZE = 500

def fake_embed(text):
    np.random.seed(abs(hash(text)) % (10**8))
    return np.random.rand(dimension).astype("float32")

tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(tokenizer.encode(text))

def extract_log_content(path: str) -> str | None:
    try:
        if path.endswith(".gz"):
            with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as f:
                return f.read()
        else:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None

def extract_and_save(uploaded_file: UploadFile):
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
                    if name.endswith(".log") or name.endswith(".gz"):
                        content = extract_log_content(path)
                        if content:
                            logs.append({"filename": name, "content": content})
        elif file_ext.endswith(".gz") or file_ext.endswith(".log"):
            content = extract_log_content(temp_path)
            if content:
                logs.append({"filename": uploaded_file.filename, "content": content})
        else:
            return False
    except Exception as e:
        print("Extraction error:", e)
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

openai.api_key = "Put api key here"

SYSTEM_PROMPT = """
You are a senior Java backend engineer and expert in log analysis. Your job is to help identify errors, their causes, and patterns in Java production logs.

### Context:
You will receive **log chunk(s)**. These logs may include:
- Stack traces
- Exceptions and error messages
- GC logs
- Thread dumps
- Application-level debug/info logs
- Timestamps, thread names, component/service markers

Logs may be fragmented, out-of-order, repetitive, or partially truncated.

---

### Tasks & Output Rules:

Depending on the user query, adapt your response style as follows:

---

🟢 **1. If the question involves counts (e.g., \"how many exceptions\")**:
- Output a **table** with:
  - Exception Type
  - Count
  - First Occurrence Timestamp (if available)
  - Threads involved (if detectable)

---

🟢 **2. If asked about a specific exception's root cause**:
- Trace from the **exception backwards** through stack traces, thread info, or related log entries.
- Output using **structured Markdown**:
  - **🔍 Root Cause**
  - **📍 Location** (Thread, Class, Line, Timestamp)
  - **🧠 Suggestion** (Fix or mitigation if inferrable)

---

🟢 **3. If asked for a general diagnostic (e.g., “do you see anything concerning?”)**:
- Scan logs for red flags:
  - High GC activity
  - Repeated timeouts
  - Frequent restarts
  - Thread starvation or lock contention
- Output a **bullet-point diagnostic summary** using these labels:
  - ✅ Healthy signs
  - ⚠️ Warnings
  - ❌ Critical errors
  - 🧩 Observed patterns
  
🔢 **4. Whenever your answer includes structured records (lists of items):**
- Format them as a **Markdown table**.
- Use clear headers and readable rows (avoid long cells).
- Use this formatting even if the user didn’t explicitly ask for a table.
---

If context is insufficient:
Say clearly: "_More log context is required to make a definitive conclusion._"

If logs are irrelevant:
Say clearly: "_No relevant log entries found._"

Your answers are read by backend developers and SREs. Be concise, technical, and accurate. Avoid speculation. Format your answer using Markdown or tables for maximum clarity.
"""

def ask_question_to_gpt(question: str):
    if not stored_chunks or index.ntotal == 0:
        return "No logs found. Please upload the logs."

    question_vec = fake_embed(question)
    _, I = index.search(np.array([question_vec]), k=20)
    top_chunks = sorted(
        [(stored_chunks[i][0], np.linalg.norm(question_vec - stored_chunks[i][1])) for i in I[0]],
        key=lambda x: x[1]
    )[:5]
    context_chunks = [chunk for chunk, _ in top_chunks]

    MAX_TOKENS_CONTEXT = 6000
    current_tokens = 0
    final_chunks = []

    for chunk in context_chunks:
        chunk_tokens = count_tokens(chunk)
        if current_tokens + chunk_tokens > MAX_TOKENS_CONTEXT:
            break
        final_chunks.append(chunk)
        current_tokens += chunk_tokens

    context = "\n---\n".join(final_chunks)
    prompt = f"""
Context:
{context}

Question:
{question}

Answer:
"""

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": prompt.strip()}
        ],
        stream=False
    )

    return response.choices[0].message.content.strip()

@app.post("/upload")
async def upload_log(file: UploadFile = File(...)):
    success = extract_and_save(file)
    if not success:
        return JSONResponse(status_code=413, content={"error": "Log extraction failed or unsupported format."})
    return JSONResponse(content={"message": "Logs uploaded and processed"})

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    answer = ask_question_to_gpt(question)
    return JSONResponse(content={"answer": answer})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            question = await websocket.receive_text()
            print(f"Received from client: {question}")

            if not stored_chunks or index.ntotal == 0:
                await websocket.send_text("No logs found. Please upload the logs first.")
                await websocket.send_text("[DONE]")
                continue

            question_vec = fake_embed(question)
            _, I = index.search(np.array([question_vec]), k=10)
            top_chunks = sorted(
                [(stored_chunks[i][0], np.linalg.norm(question_vec - stored_chunks[i][1])) for i in I[0]],
                key=lambda x: x[1]
            )[:5]
            context_chunks = [chunk for chunk, _ in top_chunks]

            final_chunks = []
            current_tokens = 0
            MAX_TOKENS_CONTEXT = 6000
            for chunk in context_chunks:
                chunk_tokens = count_tokens(chunk)
                if current_tokens + chunk_tokens > MAX_TOKENS_CONTEXT:
                    break
                final_chunks.append(chunk)
                current_tokens += chunk_tokens

            context = "\n---\n".join(final_chunks)
            prompt_user = f"""
Context:
{context}

Question:
{question}

Answer:
"""

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": prompt_user.strip()}
            ]

            full_prompt_text = SYSTEM_PROMPT.strip() + prompt_user.strip()
            input_tokens = count_tokens(full_prompt_text)
            print(f"[TOKEN USAGE] Input tokens: {input_tokens}", flush=True)

            full_response = ""
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=messages,
                stream=True
            )

            for chunk in response:
                if isinstance(chunk, ChatCompletionChunk):
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        token = delta.content
                        full_response += token
                        await websocket.send_text(token)

            output_tokens = count_tokens(full_response)
            print(f"[TOKEN USAGE] Output tokens: {output_tokens}", flush=True)
            print(f"[TOKEN USAGE] Total tokens: {input_tokens + output_tokens}", flush=True)

            await websocket.send_text("[DONE]")

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print("Error:", e)
        await websocket.send_text("Error while processing question.")
        await websocket.send_text("[DONE]")
