import os
import time

from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODELS: dict[str, SentenceTransformer] = {}
MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-mpnet-base-v2")


def load_model(name: str) -> SentenceTransformer:
    loaded_model = MODELS.get(name)
    if loaded_model:
        return loaded_model

    model_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".models", MODEL_NAME)
    )
    print(f"🤖 Loading model `{MODEL_NAME}` from {model_path}")
    model = SentenceTransformer(model_path, trust_remote_code=True)
    MODELS[name] = model
    return model


class CalculateEmbeddingsResponse(BaseModel):
    vectors: list[float]
    tokens: int
    duration: float


async def calculate_embeddings(input: str) -> CalculateEmbeddingsResponse:
    start_time = time.time()

    model = load_model(MODEL_NAME)
    vectors = model.encode(input)

    end_time = time.time()
    duration = end_time - start_time

    return CalculateEmbeddingsResponse(
        vectors=vectors.tolist(),
        tokens=len(vectors),
        duration=duration,
    )
