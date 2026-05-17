from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import os
from PIL import Image

from backend import (
    init_state,
    cb_analyze,
    cb_create_profile,
    cb_image_chat,
    cb_pdf_chat,
    cb_pdf_upload,
    cb_compare_plan_with_body,
)
from src.pdf_upload import _to_path

app = FastAPI(title="Smart Gym RAG API")
APP_STATE = init_state()


class ProfileRequest(BaseModel):
    age: int
    height: float
    weight: float
    fitness_level: str
    gender: str
    health_issues: str


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {"message": "Smart Gym RAG API is running"}


@app.get("/state")
def get_state():
    global APP_STATE
    return APP_STATE


@app.get("/profile")
def get_profile():
    global APP_STATE
    return {
        "user_profile": APP_STATE.get("user_profile"),
        "profile_path": APP_STATE.get("profile_path")
    }


@app.post("/create-profile")
def api_create_profile(payload: ProfileRequest):
    global APP_STATE

    msg, new_state = cb_create_profile(
        payload.age,
        payload.height,
        payload.weight,
        payload.fitness_level,
        payload.gender,
        payload.health_issues,
        APP_STATE
    )
    APP_STATE = new_state

    return {
        "message": msg,
        "user_profile": APP_STATE.get("user_profile"),
        "profile_path": APP_STATE.get("profile_path")
    }


@app.post("/upload-document")
async def api_upload_document(file: UploadFile = File(...)):
    global APP_STATE

    os.makedirs("uploads/documents", exist_ok=True)

    save_path = os.path.join("uploads/documents", file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    save_path = _to_path(save_path)

    msg, new_state = cb_pdf_upload(save_path, APP_STATE)
    APP_STATE = new_state

    document_state = APP_STATE.get("document", {})

    return {
        "message": msg,
        "pdf_uploaded": document_state.get("uploaded"),
        "pdf_file_paths": document_state.get("file_paths"),
        "document_vectorstore_path": document_state.get("vectorstore_path"),
        "plan_summary": document_state.get("plan_summary"),
    }


@app.post("/chat-document")
def api_chat_document(payload: ChatRequest):
    global APP_STATE

    APP_STATE["llm_provider"] = payload.provider

    history = []
    history, _, new_state = cb_pdf_chat(payload.message, history, APP_STATE)
    APP_STATE = new_state

    reply = history[-1][1] if history else ""

    return {
        "question": payload.message,
        "provider": payload.provider,
        "answer": reply
    }


@app.post("/analyze-image")
async def api_analyze_image(file: UploadFile = File(...)):
    global APP_STATE

    try:
        os.makedirs("uploads/images", exist_ok=True)
        save_path = os.path.join("uploads/images", file.filename)

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img = Image.open(save_path).convert("RGB")

        msg, new_state = cb_analyze(img, APP_STATE)
        APP_STATE = new_state

        return {
            "message": msg,
            "image": APP_STATE.get("image")
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "message": f"❌ {e}",
            "image": None
        }

@app.post("/chat-image")
def api_chat_image(payload: ChatRequest):
    global APP_STATE

    APP_STATE["llm_provider"] = payload.provider

    history = []
    history, _, new_state = cb_image_chat(payload.message, history, APP_STATE)
    APP_STATE = new_state

    reply = history[-1][1] if history else ""

    return {
        "question": payload.message,
        "provider": payload.provider,
        "answer": reply
    }


@app.post("/compare-plan-body")
def api_compare_plan_body():
    global APP_STATE

    msg, new_state = cb_compare_plan_with_body(APP_STATE)
    APP_STATE = new_state

    return {
        "message": msg,
        "comparison": APP_STATE.get("comparison", {}).get("last_result")
    }