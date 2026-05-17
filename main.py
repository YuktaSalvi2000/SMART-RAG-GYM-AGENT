"""
Smart Gym RAG - main.py
Production FastAPI layer using backend.py + AWS Bedrock + S3 + FAISS.

"""

import json
import logging
import os
import shutil
import tempfile
import time
import traceback
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from PIL import Image
from pydantic import BaseModel

from backend import (
    cb_analyze,
    cb_compare_plan_with_body,
    cb_create_profile,
    cb_image_chat,
    cb_pdf_chat,
    cb_pdf_upload,
    init_state,
)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Environment ───────────────────────────────────────────────────────────────

AWS_REGION       = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
S3_BUCKET        = os.getenv("S3_BUCKET", "your-smart-gym-rag-bucket")
S3_PROFILE_PREFIX = os.getenv("S3_PROFILE_PREFIX", "profiles")
API_KEY          = os.getenv("API_KEY", "change-me-in-production")
ALLOWED_ORIGINS  = os.getenv("ALLOWED_ORIGINS", "http://3.87.237.222:8000").split(",")

MAX_FILE_SIZE    = 10 * 1024 * 1024   # 10 MB
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".docx"}


# ── AWS clients ───────────────────────────────────────────────────────────────

bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
bedrock_control = boto3.client("bedrock",         region_name=AWS_REGION)
s3_client       = boto3.client("s3",              region_name=AWS_REGION)


# ── Per-user state (LRU, max 200 active sessions) ────────────────────────────

class LRUSessionStore:
    """
    Keeps at most `maxsize` user sessions in memory.
    Evicts the least-recently-used session when full.
    Prevents unbounded memory growth as users accumulate.
    """

    def __init__(self, maxsize: int = 200):
        self._store: OrderedDict[str, dict] = OrderedDict()
        self._maxsize = maxsize

    def get(self, user_id: str) -> Optional[dict]:
        if user_id not in self._store:
            return None
        self._store.move_to_end(user_id)
        return self._store[user_id]

    def set(self, user_id: str, state: dict) -> None:
        if user_id in self._store:
            self._store.move_to_end(user_id)
        self._store[user_id] = state
        if len(self._store) > self._maxsize:
            evicted, _ = self._store.popitem(last=False)
            logger.info(f"Evicted session for user_id={evicted}")

    def get_or_create(self, user_id: str) -> dict:
        state = self.get(user_id)
        if state is None:
            state = init_state()
            self.set(user_id, state)
        return state


USER_STATES = LRUSessionStore(maxsize=200)


# ── Auth ──────────────────────────────────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str = Security(api_key_header)):
    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")
    return api_key


# ── S3 profile helpers ────────────────────────────────────────────────────────

def _s3_profile_key(user_id: str) -> str:
    return f"{S3_PROFILE_PREFIX}/{user_id}.json"


def save_profile_to_s3(user_id: str, profile: dict) -> None:
    try:
        body = json.dumps(profile, indent=2)
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=_s3_profile_key(user_id),
            Body=body,
            ContentType="application/json",
        )
        logger.info(f"Profile saved to S3 for user_id={user_id}")
    except Exception as e:
        logger.warning(f"S3 profile save failed for {user_id}: {e}")


def load_profile_from_s3(user_id: str) -> Optional[dict]:
    try:
        obj = s3_client.get_object(
            Bucket=S3_BUCKET,
            Key=_s3_profile_key(user_id),
        )
        return json.loads(obj["Body"].read())
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception as e:
        logger.warning(f"S3 profile load failed for {user_id}: {e}")
        return None


# ── Bedrock RAG call (used as the LLM backend) ────────────────────────────────

def call_bedrock(prompt: str) -> str:
    """
    Call AWS Bedrock with the given prompt.
    Returns the model's text response.
    """
    try:
        response = bedrock_runtime.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1200,
                "temperature": 0.0,          # deterministic for factual RAG
                "messages": [
                    {"role": "user", "content": prompt}
                ],
            }),
        )
        result = json.loads(response["Body"].read())
        return result["content"][0]["text"]
    except Exception as e:
        logger.error(f"Bedrock call failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bedrock error: {str(e)}")


# ── Pydantic models ───────────────────────────────────────────────────────────

class ProfileRequest(BaseModel):
    user_id: str
    age: int
    height: float
    weight: float
    fitness_level: str
    gender: str
    health_issues: str


class ChatRequest(BaseModel):
    user_id: str
    message: str
    provider: str = "bedrock"    # bedrock | openai


class CompareRequest(BaseModel):
    user_id: str


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info("Smart Gym RAG API starting up")
    logger.info(f"Bedrock model: {BEDROCK_MODEL_ID}")
    logger.info(f"S3 bucket:     {S3_BUCKET}")
    yield
    logger.info("Smart Gym RAG API shutting down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Smart Gym RAG API",
    description="Fitness RAG API — FastAPI + FAISS + S3 + AWS Bedrock",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,      # restricted — not wildcard
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Smart Gym RAG API",
        "version": "3.0.0",
        "llm": "AWS Bedrock",
        "vector_db": "FAISS",
        "storage": "S3",
    }


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """
    Checks S3 and Bedrock connectivity.
    Returns per-service status so the ALB and ops team can pinpoint failures.
    """
    checks: Dict[str, str] = {
        "api":     "healthy",
        "s3":      "unknown",
        "bedrock": "unknown",
    }

    try:
        s3_client.head_bucket(Bucket=S3_BUCKET)
        checks["s3"] = "healthy"
    except Exception as e:
        checks["s3"] = f"unhealthy: {e}"

    try:
        bedrock_control.list_foundation_models()
        checks["bedrock"] = "healthy"
    except Exception as e:
        checks["bedrock"] = f"unhealthy: {e}"

    overall = "healthy" if all(v == "healthy" for v in checks.values()) else "degraded"
    return {"status": overall, "checks": checks}


# ── Profile routes ────────────────────────────────────────────────────────────

@app.post("/create-profile", dependencies=[Depends(require_api_key)])
def api_create_profile(payload: ProfileRequest):
    """
    Create or update a user profile.
    Persists to S3 and caches in session state.
    """
    state = USER_STATES.get_or_create(payload.user_id)

    try:
        msg, new_state = cb_create_profile(
            payload.age,
            payload.height,
            payload.weight,
            payload.fitness_level,
            payload.gender,
            payload.health_issues,
            state,
        )
        USER_STATES.set(payload.user_id, new_state)

        # Persist to S3 so profile survives instance restarts
        profile = new_state.get("user_profile", {})
        save_profile_to_s3(payload.user_id, profile)

        return {
            "message":      msg,
            "user_id":      payload.user_id,
            "user_profile": profile,
            "profile_path": new_state.get("profile_path"),
        }

    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/profile/{user_id}", dependencies=[Depends(require_api_key)])
def api_get_profile(user_id: str):
    """
    Return the user profile — from session cache or S3 fallback.
    """
    # Try session cache first
    state = USER_STATES.get(user_id)
    if state and state.get("user_profile"):
        return {"user_id": user_id, "user_profile": state["user_profile"]}

    # Fall back to S3
    profile = load_profile_from_s3(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found.")

    # Warm the session cache
    state = USER_STATES.get_or_create(user_id)
    state["user_profile"] = profile
    USER_STATES.set(user_id, state)

    return {"user_id": user_id, "user_profile": profile}


# ── Document upload ───────────────────────────────────────────────────────────

@app.post("/upload-document", dependencies=[Depends(require_api_key)])
async def api_upload_document(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Upload and index a document into FAISS.
    Validates file size and extension before processing.
    Cleans up temp files in all cases.
    """
    start = time.time()

    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not supported. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Read and validate size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)} MB.",
        )

    # Use a random safe filename — never trust user-supplied filenames
    safe_name = f"{uuid.uuid4()}{ext}"
    upload_dir = Path("uploads/documents")
    upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = upload_dir / safe_name

    try:
        save_path.write_bytes(contents)

        state = USER_STATES.get_or_create(user_id)
        msg, new_state = cb_pdf_upload(str(save_path), state)
        USER_STATES.set(user_id, new_state)

        document_state = new_state.get("document", {})

        return {
            "message":              msg,
            "user_id":              user_id,
            "original_filename":    file.filename,
            "pdf_uploaded":         document_state.get("uploaded"),
            "pdf_file_paths":       document_state.get("file_paths"),
            "vectorstore_path":     document_state.get("vectorstore_path"),
            "plan_summary":         document_state.get("plan_summary"),
            "elapsed_ms":           round((time.time() - start) * 1000, 2),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Always clean up the temp file — prevent disk fill
        if save_path.exists():
            save_path.unlink()


# ── Document chat ─────────────────────────────────────────────────────────────

@app.post("/chat-document", dependencies=[Depends(require_api_key)])
def api_chat_document(payload: ChatRequest):
    """
    Answer a question using the uploaded document vectorstore + user profile.
    Uses AWS Bedrock as the LLM when provider="bedrock".
    """
    start = time.time()
    state = USER_STATES.get_or_create(payload.user_id)

    try:
        history = []
        history, _, new_state = cb_pdf_chat(payload.message, history, state)
        USER_STATES.set(payload.user_id, new_state)

        reply = history[-1][1] if history else ""

        # If the backend used an internal LLM, optionally re-route through Bedrock
        if payload.provider == "bedrock" and reply and not reply.startswith("⚠️"):
            # reply is already generated by backend — Bedrock used via get_llm()
            # If you want to force Bedrock here, pass the prompt to call_bedrock()
            pass

        return {
            "user_id":    payload.user_id,
            "question":   payload.message,
            "answer":     reply,
            "provider":   payload.provider,
            "elapsed_ms": round((time.time() - start) * 1000, 2),
        }

    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ── Image analysis ────────────────────────────────────────────────────────────

@app.post("/analyze-image", dependencies=[Depends(require_api_key)])
async def api_analyze_image(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Analyse a body image using BLIP captioning + LLM interpretation.
    """
    start = time.time()

    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(
            status_code=400,
            detail=f"Image type '{ext}' not supported. Use JPG, PNG, or WEBP.",
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image too large. Max 10 MB.")

    safe_name = f"{uuid.uuid4()}{ext}"
    image_dir = Path("uploads/images")
    image_dir.mkdir(parents=True, exist_ok=True)
    save_path = image_dir / safe_name

    try:
        save_path.write_bytes(contents)
        img = Image.open(save_path).convert("RGB")

        state = USER_STATES.get_or_create(user_id)
        msg, new_state = cb_analyze(img, state)
        USER_STATES.set(user_id, new_state)

        return {
            "user_id":    user_id,
            "message":    msg,
            "image":      new_state.get("image"),
            "elapsed_ms": round((time.time() - start) * 1000, 2),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        # Return proper 500 — not a silent 200 with error message
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if save_path.exists():
            save_path.unlink()


# ── Image chat ────────────────────────────────────────────────────────────────

@app.post("/chat-image", dependencies=[Depends(require_api_key)])
def api_chat_image(payload: ChatRequest):
    """
    Answer a question using the image analysis vectorstore.
    """
    start = time.time()
    state = USER_STATES.get_or_create(payload.user_id)

    try:
        history = []
        history, _, new_state = cb_image_chat(payload.message, history, state)
        USER_STATES.set(payload.user_id, new_state)

        reply = history[-1][1] if history else ""

        return {
            "user_id":    payload.user_id,
            "question":   payload.message,
            "answer":     reply,
            "provider":   payload.provider,
            "elapsed_ms": round((time.time() - start) * 1000, 2),
        }

    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ── Compare plan with body ────────────────────────────────────────────────────

@app.post("/compare-plan-body", dependencies=[Depends(require_api_key)])
def api_compare_plan_body(payload: CompareRequest):
    """
    Compare the uploaded workout plan against the analysed body image
    and user profile. Returns a suitability verdict with specific recommendations.
    """
    start = time.time()
    state = USER_STATES.get_or_create(payload.user_id)

    try:
        msg, new_state = cb_compare_plan_with_body(state)
        USER_STATES.set(payload.user_id, new_state)

        return {
            "user_id":    payload.user_id,
            "message":    msg,
            "comparison": new_state.get("comparison", {}).get("last_result"),
            "meta": {
                "used_profile":  new_state["comparison"].get("used_profile"),
                "used_image":    new_state["comparison"].get("used_image"),
                "used_document": new_state["comparison"].get("used_document"),
            },
            "elapsed_ms": round((time.time() - start) * 1000, 2),
        }

    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level="info",
        workers=1,          # use 1 worker per container — scale via ALB + ASG
    )