import openai
import numpy as np
import tiktoken
from utils.constants import SYSTEM_PROMPT, CHUNK_SIZE, DIMENSION
from utils.file_utils import index, stored_chunks, fake_embed
from utils.env_config import EnvConfig

env = EnvConfig()
openai.api_key = env.openai_api_key

tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(tokenizer.encode(text))

def ask_question_to_gpt(question: str):
    if not stored_chunks or index.ntotal == 0:
        return "No logs found. Please upload the logs."

    question_vec = fake_embed(question)
    _, I = index.search(np.array([question_vec]), k=20)
    top_chunks = sorted(
        [(stored_chunks[i][0], np.linalg.norm(question_vec - stored_chunks[i][1])) for i in I[0]],
        key=lambda x: x[1]
    )[:5]

    context = "\n---\n".join(chunk for chunk, _ in top_chunks)

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

async def handle_websocket_question(websocket):
    while True:
        question = await websocket.receive_text()
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

        context = "\n---\n".join(chunk for chunk, _ in top_chunks)

        prompt = f"""
Context:
{context}

Question:
{question}

Answer:
"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": prompt.strip()}
        ]

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            stream=True
        )

        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
                await websocket.send_text(token)
        await websocket.send_text("[DONE]")
