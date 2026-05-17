# src/state_manager.py

from backend import DEFAULT_EMBEDDING

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def init_state():
    """
    Return a clean, typed application state dictionary.
    All keys used anywhere in the backend must be initialised here.
    """
    return {
        "user_profile": None,
        "profile_path": None,

        # Cached vectorstore objects — loaded once, reused across messages
        "document_vs": None,
        "image_vs": None,

        "image": {
            "analysis_text": "",
            "items": [],
            "vectorstore_path": None,
            "clip_index_path": None,
            "clip_meta_path": None,
            "uploaded": False,
            "error": None,
        },

        "document": {
            "vectorstore_path": None,
            "embedding_kind": DEFAULT_EMBEDDING,
            "plan_summary": "",
            "raw_preview": "",
            "meta": None,
            "uploaded": False,
            "file_paths": [],
        },

        "comparison": {
            "last_result": None,
            "used_profile": False,
            "used_image": False,
            "used_document": False,
        },

        "document_vs": None,        # cached FAISS vectorstore object
        "image_vs": None,           # cached image FAISS vectorstore object
        "comparison": {             # comparison results
            "last_result": None,
            "used_profile": False,
            "used_image": False,
            "used_document": False,
        },
    }



STATE = init_state()

# Backward-compatible alias for code that imports GLOBAL_STATE
GLOBAL_STATE = STATE