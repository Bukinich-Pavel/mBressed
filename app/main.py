from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os

# Configure writable caches for read-only environments (e.g., Vercel). Must run before importing ML libs.
CACHE_ROOT = os.environ.get("EMBED_CACHE_ROOT", "/tmp")
if (
    os.environ.get("VERCEL") == "1"
    or os.environ.get("READ_ONLY_FS") == "1"
    or not os.access("/", os.W_OK)
):
    hf_home = os.environ.setdefault("HF_HOME", os.path.join(CACHE_ROOT, "hf"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(CACHE_ROOT, "cache"))
    # Best-effort create dirs; ignore failures in restricted envs
    for key in (
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "XDG_CACHE_HOME",
    ):
        try:
            os.makedirs(os.environ[key], exist_ok=True)
        except Exception:
            pass

from fastembed import TextEmbedding


app = FastAPI(title="Embedding Service", version="0.1.0")


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# Initialize the embedding model at startup (lazy fallback if initial init fails)
try:
    embedding_model = TextEmbedding(model_name=MODEL_NAME)
    model_init_error = None
except Exception as exc:  # pragma: no cover - environment dependent
    embedding_model = None
    model_init_error = exc


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embedding: List[float]
    model: str
    dim: int


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "ready": embedding_model is not None and model_init_error is None,
    }


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    try:
        global embedding_model, model_init_error

        # Lazy re-init if startup failed
        if embedding_model is None:
            embedding_model = TextEmbedding(model_name=MODEL_NAME)
            model_init_error = None

        vectors = list(embedding_model.embed([req.text]))
        if not vectors:
            raise RuntimeError("No embedding returned from model")

        vec = vectors[0]
        embedding_list = [float(x) for x in vec]

        return EmbedResponse(
            embedding=embedding_list,
            model=MODEL_NAME,
            dim=len(embedding_list),
        )
    except Exception as e:  # pragma: no cover - runtime safety
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")


