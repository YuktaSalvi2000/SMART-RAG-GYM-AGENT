"""
Smart Gym RAG - backend.py
Production-ready version using src/state_manager.py as single source of truth.

State schema is flat (from state_manager.py) — no nested document/image dicts.
All session state flows through the state dict passed in and returned from each cb_ function.
"""

import os
import json
import threading
import traceback
from functools import lru_cache

from PIL import Image

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import (
    CLIPProcessor,
    CLIPModel,
    BlipProcessor,
    BlipForConditionalGeneration,
)

from src.llm_setup import get_model_adaptive, get_blip_captioner, get_clip_model
from src.profile_store import create_profile
from src.image_pipeline import analyze_image
from src.pdf_upload import upload_and_index, load_documents, _to_path
from src.st_embedding import STEmbedding

# ── Single source of truth for state schema ───────────────────────────────────
from src.state_manager import init_state


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_CONTEXT_CHARS = 3000    # ~750 tokens — cap retrieved context before prompting
MAX_PREVIEW_CHARS = 500     # cap raw_preview injection
DEFAULT_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"

# Thread lock — prevents concurrent FAISS writes corrupting the index
_faiss_lock = threading.Lock()


# ── LLM singleton ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_llm():
    """
    Return a cached LLM instance.
    lru_cache ensures the model initialises once and is reused across requests.
    Avoids repeated cold-start latency without sharing mutable state.
    """
    return get_model_adaptive()


# ── Vectorstore loaders ───────────────────────────────────────────────────────

def load_document_vectorstore(
    vectorstore_path: str,
    embedding_kind: str = DEFAULT_EMBEDDING,
):
    """Load a LangChain FAISS vectorstore from disk using HuggingFace embeddings."""
    if not vectorstore_path or not os.path.exists(vectorstore_path):
        return None

    embeddings = HuggingFaceEmbeddings(model_name=embedding_kind)
    return FAISS.load_local(
        vectorstore_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def load_image_vectorstore(vectorstore_path: str):
    """Load the image analysis FAISS vectorstore using STEmbedding."""
    if not vectorstore_path or not os.path.exists(vectorstore_path):
        return None

    text_embeddings = STEmbedding()
    return FAISS.load_local(
        vectorstore_path,
        embeddings=text_embeddings,
        allow_dangerous_deserialization=True,
    )


# ── Vectorstore cache helpers ─────────────────────────────────────────────────

def _get_document_vs(state: dict):
    """
    Return the document vectorstore.
    Loads from disk once and caches in state["document_vs"].
    Subsequent calls return the cached object — no repeated disk reads.
    """
    if state.get("document_vs") is not None:
        return state["document_vs"]

    vs_path = state.get("document_vectorstore_path")
    embedding_kind = state.get("document_embedding_kind", DEFAULT_EMBEDDING)

    vs = load_document_vectorstore(vs_path, embedding_kind=embedding_kind)
    state["document_vs"] = vs
    return vs


def _get_image_vs(state: dict):
    """
    Return the image vectorstore.
    Loads from disk once and caches in state["image_vs"].
    """
    if state.get("image_vs") is not None:
        return state["image_vs"]

    vs_path = state.get("image_vectorstore_path")
    vs = load_image_vectorstore(vs_path)
    state["image_vs"] = vs
    return vs


# ── Safe string helpers ───────────────────────────────────────────────────────

def _safe_str(value, max_chars: int = None, fallback: str = "Not available") -> str:
    """
    Convert a value to a safe string for prompt injection.
    Handles None, dicts, and enforces a character cap.
    """
    if value is None:
        return fallback
    if isinstance(value, dict):
        text = json.dumps(value, indent=2)
    else:
        text = str(value)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + "...[truncated]"
    return text


def _format_profile(profile: dict) -> str:
    """Format user profile fields for prompt injection."""
    if not profile:
        return "No profile available."
    return "\n".join([
        f"- Age: {profile.get('age', 'Unknown')}",
        f"- Height: {profile.get('height', 'Unknown')}",
        f"- Weight: {profile.get('weight', 'Unknown')}",
        f"- Fitness Level: {profile.get('fitness_level', 'Unknown')}",
        f"- Gender: {profile.get('gender', 'Unknown')}",
        f"- Health Issues: {profile.get('health_issues', 'None')}",
    ])


# ── Callback: create profile ──────────────────────────────────────────────────

def cb_create_profile(
    age, height, weight, fitness_level, gender, health_issues, state
):
    """
    Create and persist a user profile.
    Returns (message, updated_state).

    Session state updated:
        state["user_profile"]  — profile dict
        state["profile_path"]  — path to saved profile file
        state["last_action"]   — "create_profile"
    """
    state = state or init_state()

    try:
        profile, path = create_profile(
            age=int(age) if age is not None else 0,
            height=float(height) if height is not None else 0.0,
            weight=float(weight) if weight is not None else 0.0,
            fitness_level=fitness_level or "",
            gender=gender or "",
            health_issues=health_issues or "",
        )

        state["user_profile"] = profile
        state["profile_path"] = path
        state["last_action"] = "create_profile"

        return (
            f"✅ Profile created and saved.\n📁 {path}\nNow choose a mode below.",
            state,
        )

    except Exception as e:
        traceback.print_exc()
        state["last_error"] = str(e)
        return f"❌ Profile creation error: {e}", state


# ── Callback: PDF / document upload ──────────────────────────────────────────

def cb_pdf_upload(file_path, state):
    """
    Upload and index a document (PDF, DOCX, CSV) into FAISS.
    Returns (message, updated_state).

    Session state updated:
        state["pdf_uploaded"]               — True on success
        state["pdf_file_paths"]             — list of uploaded file paths
        state["document_vectorstore_path"]  — set by upload_and_index
        state["plan_summary"]               — set by upload_and_index
        state["pdf_raw_preview"]            — set by upload_and_index
        state["pdf_meta"]                   — set by upload_and_index
        state["document_vs"]               — invalidated (None) so next chat reloads
        state["last_action"]               — "pdf_upload"
    """
    state = state or init_state()

    if not state.get("user_profile"):
        return "⚠️ Please create your profile first.", state

    if not file_path:
        return "❌ No file received.", state

    try:
        file_path = _to_path(file_path)

        if not os.path.exists(file_path):
            return "❌ Uploaded file not found on server.", state

        # Thread lock prevents concurrent FAISS writes corrupting the index
        with _faiss_lock:
            msg, state = upload_and_index(file_path, state)

        # Mark document as uploaded using flat state keys
        state["pdf_uploaded"] = True
        state["pdf_file_paths"] = [str(file_path)]
        state["last_action"] = "pdf_upload"

        # Invalidate cached vectorstore — new upload means new index
        state["document_vs"] = None

        return msg, state

    except Exception as e:
        traceback.print_exc()
        state["last_error"] = str(e)
        return f"❌ Document upload error: {e}", state


# ── Callback: PDF / document chat ─────────────────────────────────────────────

def cb_pdf_chat(message, history, state):
    """
    Answer a question using the uploaded document vectorstore + user profile.
    Returns (history, "", updated_state).

    Session state read:
        state["pdf_uploaded"]               — guard check
        state["document_vectorstore_path"]  — FAISS index path
        state["document_embedding_kind"]    — embedding model name
        state["document_vs"]               — cached vectorstore (loaded once)
        state["plan_summary"]              — injected into prompt
        state["pdf_meta"]                  — injected into prompt
        state["pdf_raw_preview"]           — injected into prompt
        state["user_profile"]              — injected into prompt

    Session state updated:
        state["document_vs"]               — cached after first load
        state["last_action"]               — "pdf_chat"
    """
    history = history or []

    if not message or not message.strip():
        history.append(("", "Please type a question."))
        return history, "", state

    # Guard: flat key from state_manager
    if not state.get("pdf_uploaded"):
        history.append((message, "⚠️ Upload your PDF/DOCX/CSV first."))
        return history, "", state

    vs_path = state.get("document_vectorstore_path")
    if not vs_path or not os.path.exists(vs_path):
        history.append((
            message,
            "⚠️ Document vectorstore not found. Please re-upload the document.",
        ))
        return history, "", state

    # Load vectorstore once, cache in state
    vs = _get_document_vs(state)
    if vs is None:
        history.append((message, "⚠️ Could not load the document vectorstore."))
        return history, "", state

    docs = vs.similarity_search(message, k=4)
    if not docs:
        history.append((message, "⚠️ No relevant content found in the document."))
        return history, "", state

    retrieved = "\n\n".join(d.page_content for d in docs)
    retrieved = retrieved[:MAX_CONTEXT_CHARS]

    # Read from flat state keys
    plan_summary = _safe_str(state.get("plan_summary"))
    meta         = _safe_str(state.get("pdf_meta"))
    raw_preview  = _safe_str(state.get("pdf_raw_preview"), max_chars=MAX_PREVIEW_CHARS)
    profile_text = _format_profile(state.get("user_profile", {}))

    prompt = f"""You are a smart fitness assistant helping the user understand and modify their uploaded workout or diet plan.
Use plain text only — avoid LaTeX or special formatting.

Additional document context:
- Plan summary: {plan_summary}
- Document metadata: {meta}
- Raw preview: {raw_preview}

User profile:
{profile_text}

Retrieved plan context:
{retrieved}

User question:
{message}

Rules:
- Answer using ONLY the retrieved context and the user profile.
- If the retrieved context does not contain the answer, clearly say what information is missing.
- If the user asks for changes, modify only the relevant section.
- Respect health issues and suggest safe alternatives.
- For medical concerns, recommend professional guidance.
"""

    try:
        llm = get_llm()
        reply = llm.invoke(prompt).content.strip()
    except Exception as e:
        reply = f"❌ Chat error: {e}"
        state["last_error"] = str(e)

    state["last_action"] = "pdf_chat"
    history.append((message, reply))
    return history, "", state


# ── Callback: image analysis ──────────────────────────────────────────────────

def cb_analyze(img, state):
    """
    Analyse an image using BLIP captioning + LLM interpretation.
    Returns (message, updated_state).

    Session state updated:
        state["last_analysis_text"]     — full analysis text
        state["last_image_items"]       — detected items list
        state["image_vectorstore_path"] — FAISS index path for image
        state["clip_index_path"]        — CLIP index path
        state["clip_meta_path"]         — CLIP metadata path
        state["image_vs"]               — invalidated (None)
        state["last_action"]            — "analyze_image"
        state["last_error"]             — set on failure
    """
    state = state or init_state()

    profile = state.get("user_profile")
    if not profile:
        return "⚠️ Please create your profile first.", state

    try:
        llm = get_llm()
        blip_processor, blip_model = get_blip_captioner()

        msg, state = analyze_image(
            img=img,
            user_profile=profile,
            llm=llm,
            blip_processor=blip_processor,
            blip_model=blip_model,
            state=state,
            save_dir=state.get("image_save_dir", "stores/image"),
        )

        # Invalidate cached image vectorstore — new analysis means new index
        state["image_vs"] = None
        state["last_action"] = "analyze_image"

        return msg, state

    except Exception as e:
        traceback.print_exc()
        state["last_error"] = str(e)
        return f"❌ Image analysis error: {e}", state


# ── Callback: image chat ──────────────────────────────────────────────────────

def cb_image_chat(message, history, state):
    """
    Answer a question using the image analysis vectorstore.
    Returns (history, "", updated_state).

    Session state read:
        state["last_analysis_text"]     — injected into prompt as summary
        state["image_vectorstore_path"] — FAISS index path
        state["image_vs"]               — cached vectorstore (loaded once)
        state["last_error"]             — checked for prior failure

    Session state updated:
        state["image_vs"]               — cached after first load
        state["last_action"]            — "image_chat"
    """
    history = history or []
    state   = state or init_state()

    # Guard: require image analysis to have been run
    if not state.get("last_analysis_text"):
        history.append((message, "⚠️ Please analyse an image first."))
        return history, "", state

    vs_path = state.get("image_vectorstore_path")
    if not vs_path or not os.path.exists(vs_path):
        history.append((
            message,
            "⚠️ Image vectorstore not found. Please re-run image analysis.",
        ))
        return history, "", state

    # Load vectorstore once, cache in state
    vs = _get_image_vs(state)
    if vs is None:
        history.append((message, "⚠️ Could not load the image vectorstore."))
        return history, "", state

    analysis_text = state.get("last_analysis_text", "")
    docs = vs.similarity_search(message, k=4)
    retrieved = "\n\n".join(d.page_content for d in docs)
    retrieved = retrieved[:MAX_CONTEXT_CHARS]

    prompt = f"""You are a smart fitness assistant. You have two information sources:

1) Full image analysis summary (high-level overview):
{_safe_str(analysis_text, max_chars=MAX_CONTEXT_CHARS)}

2) Retrieved detail from vector database (most relevant chunks):
{retrieved}

User question:
{message}

Rules:
- Use the retrieved chunks first if they directly answer the question.
- Use the full analysis summary to maintain consistency and context.
- If the user asks for changes to workouts or diet, give specific actionable steps.
- If a safety or medical issue is mentioned, suggest safer alternatives.
- Use plain text only — avoid LaTeX formatting.
"""

    try:
        llm = get_llm()
        reply = llm.invoke(prompt).content.strip()
    except Exception as e:
        reply = f"❌ Image chat error: {e}"
        state["last_error"] = str(e)

    state["last_action"] = "image_chat"
    history.append((message, reply))
    return history, "", state


# ── Callback: compare plan with body ─────────────────────────────────────────

def cb_compare_plan_with_body(state):
    """
    Compare the uploaded workout plan against the analysed body image
    and the user profile. Returns (comparison_text, updated_state).

    Session state read:
        state["user_profile"]              — user profile dict
        state["last_analysis_text"]        — body analysis summary
        state["image_vectorstore_path"]    — image FAISS index path
        state["document_vectorstore_path"] — document FAISS index path
        state["plan_summary"]              — plan summary text
        state["image_vs"]                  — cached image vectorstore
        state["document_vs"]               — cached document vectorstore

    Session state updated:
        state["comparison"]                — dict with last_result and flags
        state["last_action"]               — "compare_plan_body"
        state["last_error"]                — set on failure
    """
    state = state or init_state()

    profile       = state.get("user_profile")
    analysis_text = state.get("last_analysis_text")
    vs_path_doc   = state.get("document_vectorstore_path")

    # Guard: all three inputs required
    if not profile:
        return "⚠️ Please create your profile first.", state

    if not analysis_text:
        return "⚠️ Please analyse an image first.", state

    if not vs_path_doc:
        return "⚠️ Please upload and process a workout document first.", state

    # Load both vectorstores (cached in state after first load)
    image_vs = _get_image_vs(state)
    doc_vs   = _get_document_vs(state)

    # Retrieve body-analysis context
    image_context = ""
    if image_vs is not None:
        image_docs = image_vs.similarity_search(
            "body proportions posture muscle tone fat distribution "
            "workout suitability weak areas strong areas",
            k=4,
        )
        image_context = "\n\n".join(d.page_content for d in image_docs)
        image_context = image_context[:MAX_CONTEXT_CHARS]

    # Retrieve workout plan context
    plan_context = ""
    if doc_vs is not None:
        plan_docs = doc_vs.similarity_search(
            "workout split exercises sets reps intensity muscle groups "
            "progression recovery suitability",
            k=4,
        )
        plan_context = "\n\n".join(d.page_content for d in plan_docs)
        plan_context = plan_context[:MAX_CONTEXT_CHARS]

    image_summary = _safe_str(analysis_text, max_chars=MAX_CONTEXT_CHARS)
    plan_summary  = _safe_str(state.get("plan_summary"), max_chars=MAX_CONTEXT_CHARS)
    profile_text  = _format_profile(profile)

    prompt = f"""You are a fitness expert evaluating whether a workout plan suits a user's body and profile.
Use plain text only — avoid LaTeX formatting.

User profile:
{profile_text}

Saved body analysis summary:
{image_summary}

Retrieved body-analysis context:
{image_context if image_context else "Not available"}

Saved workout plan summary:
{plan_summary}

Retrieved workout-plan context:
{plan_context if plan_context else "Not available"}

Your task:
1. Decide whether the workout is suitable for this body and profile.
2. Mention what matches well.
3. Mention what seems unsuitable, excessive, missing, or risky.
4. Suggest specific improvements.
5. Give a final verdict — one of:
   - Suitable
   - Partially suitable
   - Not suitable

Be specific and practical.
"""

    try:
        llm = get_llm()
        comparison = llm.invoke(prompt).content.strip()
    except Exception as e:
        traceback.print_exc()
        state["last_error"] = str(e)
        comparison = f"❌ Comparison error: {e}"

    state["comparison"] = {
        "last_result":   comparison,
        "used_profile":  bool(profile),
        "used_image":    bool(image_summary or image_context),
        "used_document": bool(plan_summary or plan_context),
    }
    state["last_action"] = "compare_plan_body"

    return comparison, state