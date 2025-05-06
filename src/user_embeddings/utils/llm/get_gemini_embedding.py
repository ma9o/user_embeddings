from google import genai
from google.genai.types import EmbedContentConfig
from google.oauth2 import service_account
from json_repair import repair_json
import logging
import os

logger = logging.getLogger(__name__)

GEMINI_EMBEDDING_MODEL_ID = "text-embedding-large-exp-03-07"
EMBEDDING_DIMENSION = 3072


def get_gemini_embedding_client() -> genai.Client:
    return genai.Client(
        vertexai=True,
        project="enclaveid",
        location="us-central1",
        credentials=service_account.Credentials.from_service_account_info(
            repair_json(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"], return_objects=True),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        ),
    )


def get_gemini_embedding(embedding_client: genai.Client, text: str) -> list[float]:
    """Generates embedding for a given text using Google Generative AI."""

    response = embedding_client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL_ID,
        contents=[text],
        config=EmbedContentConfig(
            task_type="SEMANTIC_SIMILARITY",
            output_dimensionality=EMBEDDING_DIMENSION,
        ),
    )

    if response and response.embeddings and len(response.embeddings) > 0:
        embedding_values = response.embeddings[0].values

        if len(embedding_values) != EMBEDDING_DIMENSION:
            logger.warning(
                f"Embedding dimension mismatch. Expected {EMBEDDING_DIMENSION}, got {len(embedding_values)}"
            )
            raise ValueError(
                f"Embedding dimension mismatch. Expected {EMBEDDING_DIMENSION}, got {len(embedding_values)}"
            )

        return embedding_values
    else:
        logger.warning("No embedding values returned")
        raise ValueError("No embedding values returned")
