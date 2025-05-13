from fastapi import FastAPI, UploadFile, File, Form
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import MongoClient
import os, zipfile, shutil, uuid, gzip
import faiss
import numpy as np
import openai
import tiktoken

# Initialize FastAPI
app = FastAPI()

# CORS
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

# Chunking and size limit
CHUNK_SIZE = 500
MAX_CHUNKS = 10000  # Limit for the number of chunks allowed

# Dummy embedding function
def fake_embed(text):
    np.random.seed(abs(hash(text)) % (10**8))
    return np.random.rand(dimension).astype("float32")

# Token counter using tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(tokenizer.encode(text))

# Extract and save logs from zip/gz with limit
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
    total_chunks = 0

    try:
        if file_ext.endswith(".zip"):
            with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                zip_ref.extractall(folder)

            # Walk through extracted contents
            for root, _, files in os.walk(folder):
                for name in files:
                    path = os.path.join(root, name)
                    if name.endswith(".log") or name.endswith(".gz"):
                        content = extract_log_content(path)
                        if content is None:
                            continue

                        num_chunks = len(content) // CHUNK_SIZE + (1 if len(content) % CHUNK_SIZE else 0)
                        total_chunks += num_chunks

                        if total_chunks > MAX_CHUNKS:
                            print("Exceeded chunk limit")
                            return False

                        logs.append({"filename": name, "content": content})
        elif file_ext.endswith(".gz"):
            content = extract_log_content(temp_path)
            if content is None:
                return False

            num_chunks = len(content) // CHUNK_SIZE + (1 if len(content) % CHUNK_SIZE else 0)
            if num_chunks > MAX_CHUNKS:
                print("Exceeded chunk limit")
                return False

            logs.append({"filename": uploaded_file.filename, "content": content})
        else:
            return False
    except Exception as e:
        print("Extraction error:", e)
        return False

    # Save logs to DB and FAISS
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

     
# OpenAI client
openai.api_key = "Keep API here please"

# SYSTEM_PROMPT
SYSTEM_PROMPT = """
You are a senior Java backend engineer with deep expertise in log analysis and diagnosing production issues using incomplete, fragmented, or noisy runtime logs.

## Context:
You will be given **log chunks** retrieved from a vector store. These are from Java applications and may include:
- Stack traces
- Exception messages
- Thread states
- GC activity
- Debug statements
- Timestamps and logs from different services

These logs may be out of order, partial, or truncated.

## Task:
Analyze the provided chunks to answer a question related to Java application behavior. Use structured reasoning. If context is insufficient, clearly state that more detail is required.

## Audience:
Your explanation is for engineers and DevOps teams. Avoid speculation, and base your output strictly on the log data.

## Response Format:
Always answer using the following structure:

### Observation:
Summarize what is explicitly seen in the log chunks (errors, patterns, timestamps, stack traces, etc.).

### Interpretation:
Explain what these observations mean in the context of Java runtime behavior.

### Hypothesis / Implication:
Suggest what the likely root cause or consequence could be. If information is incomplete, say what additional data would help.

## Notes:
- For logs with fewer than 10 lines, format using Markdown.
- For longer or multi-part logs, present relevant sections in a **table format** with columns: `Timestamp`, `Log Level`, `Message`.
- If logs are unrelated to the question, state: _"No relevant log entries were found in the retrieved chunks."_

## Additional Instructions:
- If the user's message appears to be a greeting (e.g., "hi", "hello", "good morning"), respond briefly and politely, acknowledging the greeting before asking for a specific question related to the logs.
- Do not ignore or dismiss greetings, but keep the response concise and professional
Be precise, conservative, and structured. Use headings and formatting to improve clarity.
"""

# Ask GPT with logs as context
def ask_question_to_gpt(question: str):
    if not stored_chunks or index.ntotal == 0:
        return "No logs found. Please upload the logs."

    needs_full_context = any(word in question.lower() for word in [
        "how many", "count", "list all", "show all", "give me all"
    ])

    if needs_full_context:
        context_chunks = [chunk for chunk, _ in stored_chunks]
    else:
        question_vec = fake_embed(question)
        _, I = index.search(np.array([question_vec]), k=5)
        context_chunks = [stored_chunks[i][0] for i in I[0]]

    MAX_TOKENS_CONTEXT = 100000
    current_tokens = 0
    final_chunks = []

    for chunk in context_chunks:
        chunk_tokens = count_tokens(chunk)
        if current_tokens + chunk_tokens > MAX_TOKENS_CONTEXT:
            print("Exceeded")
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

# Upload endpoint with limit check
@app.post("/upload")
async def upload_log(file: UploadFile = File(...)):
    success = extract_and_save(file)
    if not success:
        return JSONResponse(status_code=413, content={"error": "Log size exceeds allowed limit."})
    return JSONResponse(content={"message": "Logs uploaded and processed"})

# Ask endpoint
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
                continue

            question_vec = fake_embed(question)
            _, I = index.search(np.array([question_vec]), k=5)
            context_chunks = [stored_chunks[i][0] for i in I[0]]

            # Truncate to token limit
            final_chunks = []
            current_tokens = 0
            MAX_TOKENS_CONTEXT = 100000
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

            #STREAMING GPT CALL
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.strip()},
                    {"role": "user", "content": prompt.strip()}
                ],
                stream=True
            )

            # Send tokens as they arrive
            full_response = ""
            for chunk in response:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    token = delta.content
                    full_response += token
                    await websocket.send_text(token)

            await websocket.send_text("[DONE]")
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print("Error:", e)
        await websocket.send_text("Error while processing question.")
