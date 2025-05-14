from fastapi import APIRouter, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from service.gpt_service import ask_question_to_gpt, handle_websocket_question
from utils.file_utils import extract_and_save, index, stored_chunks

router = APIRouter()

@router.post("/upload")
async def upload_log(file: UploadFile = File(...)):
    success = extract_and_save(file)
    if not success:
        return JSONResponse(status_code=413, content={"error": "Log extraction failed or unsupported format."})
    return JSONResponse(content={"message": "Logs uploaded and processed"})

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        await handle_websocket_question(websocket)
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print("Error:", e)
        await websocket.send_text("Error while processing question.")
        await websocket.send_text("[DONE]")
