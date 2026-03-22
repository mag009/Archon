"""
Embedding Service

Handles all OpenAI embedding operations with proper rate limiting and error handling.
"""

import asyncio
import inspect
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np
import openai

from ...config.logfire_config import safe_span, search_logger
from ..credential_service import credential_service
from ..llm_provider_service import get_embedding_model, get_llm_client
from ..threading_service import get_threading_service
from .embedding_exceptions import (
    EmbeddingAPIError,
    EmbeddingError,
    EmbeddingQuotaExhaustedError,
    EmbeddingRateLimitError,
)


@dataclass
class EmbeddingBatchResult:
    """Result of batch embedding creation with success/failure tracking."""

    embeddings: list[list[float]] = field(default_factory=list)
    failed_items: list[dict[str, Any]] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    texts_processed: list[str] = field(default_factory=list)  # Successfully processed texts

    def add_success(self, embedding: list[float], text: str):
        """Add a successful embedding."""
        self.embeddings.append(embedding)
        self.texts_processed.append(text)
        self.success_count += 1

    def add_failure(self, text: str, error: Exception, batch_index: int | None = None):
        """Add a failed item with error details."""
        error_dict = {
            "text": text[:200] if text else None,
            "error": str(error),
            "error_type": type(error).__name__,
            "batch_index": batch_index,
        }

        # Add extra context from EmbeddingError if available
        if isinstance(error, EmbeddingError):
            error_dict.update(error.to_dict())

        self.failed_items.append(error_dict)
        self.failure_count += 1

    @property
    def has_failures(self) -> bool:
        return self.failure_count > 0

    @property
    def total_requested(self) -> int:
        return self.success_count + self.failure_count


class EmbeddingProviderAdapter(ABC):
    """Adapter interface for embedding providers."""

    @abstractmethod
    async def create_embeddings(
        self,
        texts: list[str],
        model: str,
        dimensions: int | None = None,
    ) -> list[list[float]]:
        pass


class OpenAICompatibleEmbeddingAdapter(EmbeddingProviderAdapter):
    """Adapter for OpenAI-compatible embedding endpoints."""

    def __init__(self, client: openai.AsyncOpenAI):
        self.client = client

    async def create_embeddings(
        self,
        texts: list[str],
        model: str,
        dimensions: int | None = None,
    ) -> list[list[float]]:
        # Use dimensions if supported (OpenAI text-embedding-3 models)
        kwargs = {"input": texts, "model": model}
        if dimensions:
            kwargs["dimensions"] = dimensions

        response = await self.client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]


class GoogleEmbeddingAdapter(EmbeddingProviderAdapter):
    """Adapter for Google's native embedding endpoint."""

    async def create_embeddings(
        self,
        texts: list[str],
        model: str,
        dimensions: int | None = None,
    ) -> list[list[float]]:
        try:
            google_api_key = await credential_service.get_credential("GOOGLE_API_KEY")
            if not google_api_key:
                raise EmbeddingAPIError("Google API key not found")

            embeddings = []
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                # Process sequentially with small delay to avoid 429
                # Google free tier is 1500 RPM but bursty
                for i, text in enumerate(texts):
                    emb = await self._fetch_single_embedding(http_client, google_api_key, model, text, dimensions)
                    embeddings.append(emb)
                    if i < len(texts) - 1:
                        await asyncio.sleep(0.5)

            return embeddings

        except httpx.HTTPStatusError as error:
            error_content = error.response.text
            search_logger.error(
                f"Google embedding API returned {error.response.status_code} - {error_content}",
                exc_info=True,
            )
            raise EmbeddingAPIError(
                f"Google embedding API error: {error.response.status_code} - {error_content}",
                original_error=error,
            ) from error
        except Exception as error:
            search_logger.error(f"Error calling Google embedding API: {error}", exc_info=True)
            raise EmbeddingAPIError(
                f"Google embedding error: {str(error)}", original_error=error
            ) from error

    async def _fetch_single_embedding(
        self,
        http_client: httpx.AsyncClient,
        api_key: str,
        model: str,
        text: str,
        dimensions: int | None = None,
    ) -> list[float]:
        if model.startswith("models/"):
            url_model = model[len("models/") :]
            payload_model = model
        else:
            url_model = model
            payload_model = f"models/{model}"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{url_model}:embedContent"
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "model": payload_model,
            "content": {"parts": [{"text": text}]},
        }

        # Add output_dimensionality parameter if dimensions are specified and supported
        if dimensions is not None and dimensions > 0:
            model_name = payload_model.removeprefix("models/")
            if model_name.startswith("textembedding-gecko"):
                supported_dimensions = {128, 256, 512, 768}
            else:
                supported_dimensions = {128, 256, 512, 768, 1024, 1536, 2048, 3072}

            if dimensions in supported_dimensions:
                payload["outputDimensionality"] = dimensions
            else:
                search_logger.warning(
                    f"Requested dimension {dimensions} is not supported by Google model '{model_name}'. "
                    "Falling back to the provider default."
                )

        response = await http_client.post(url, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()
        embedding = result.get("embedding", {})
        values = embedding.get("values") if isinstance(embedding, dict) else None
        if not isinstance(values, list):
            raise EmbeddingAPIError(f"Invalid embedding payload from Google: {result}")

        # Normalize embeddings for dimensions < 3072 as per Google's documentation
        actual_dimension = len(values)
        if actual_dimension > 0 and actual_dimension < 3072:
            values = self._normalize_embedding(values)

        return values

    def _normalize_embedding(self, embedding: list[float]) -> list[float]:
        """Normalize embedding vector for dimensions < 3072."""
        try:
            embedding_array = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                normalized = embedding_array / norm
                return normalized.tolist()
            else:
                search_logger.warning("Zero-norm embedding detected, returning unnormalized")
                return embedding
        except Exception as e:
            search_logger.error(f"Failed to normalize embedding: {e}")
            return embedding


def _get_embedding_adapter(provider: str, client: Any) -> EmbeddingProviderAdapter:
    provider_name = (provider or "").lower()
    if provider_name == "google":
        return GoogleEmbeddingAdapter()
    return OpenAICompatibleEmbeddingAdapter(client)


async def _maybe_await(value: Any) -> Any:
    """Await the value if it is awaitable, otherwise return as-is."""
    return await value if inspect.isawaitable(value) else value


async def create_embedding(text: str, provider: str | None = None) -> list[float]:
    """
    Create an embedding for a single text using the configured provider.
    """
    try:
        result = await create_embeddings_batch([text], provider=provider)
        if not result.embeddings:
            if result.has_failures and result.failed_items:
                error_info = result.failed_items[0]
                error_msg = error_info.get("error", "Unknown error")
                raise EmbeddingAPIError(f"Failed to create embedding: {error_msg}", text_preview=text)
            else:
                raise EmbeddingAPIError("No embeddings returned from batch creation", text_preview=text)
        return result.embeddings[0]
    except EmbeddingError:
        raise
    except Exception as e:
        raise EmbeddingAPIError(f"Embedding error: {str(e)}", text_preview=text, original_error=e)


async def create_embeddings_batch(
    texts: list[str],
    progress_callback: Any | None = None,
    provider: str | None = None,
) -> EmbeddingBatchResult:
    """
    Create embeddings for multiple texts with graceful failure handling.
    """
    if not texts:
        return EmbeddingBatchResult()

    result = EmbeddingBatchResult()
    validated_texts = [str(t) if not isinstance(t, str) else t for t in texts]
    threading_service = get_threading_service()

    with safe_span("create_embeddings_batch", text_count=len(texts)) as span:
        try:
            embedding_config = await _maybe_await(credential_service.get_active_provider(service_type="embedding"))
            embedding_provider = provider or embedding_config.get("provider")

            search_logger.info(f"Using embedding provider: '{embedding_provider}'")
            async with get_llm_client(provider=embedding_provider, use_embedding_provider=True) as client:
                try:
                    rag_settings = await _maybe_await(credential_service.get_credentials_by_category("rag_strategy"))
                    batch_size = int(rag_settings.get("EMBEDDING_BATCH_SIZE", "100"))
                    embedding_dimensions = int(rag_settings.get("EMBEDDING_DIMENSIONS", "1536"))
                except Exception:
                    batch_size = 100
                    embedding_dimensions = 1536

                total_tokens_used = 0
                adapter = _get_embedding_adapter(embedding_provider, client)
                dimensions_to_use = embedding_dimensions if embedding_dimensions > 0 else None

                for i in range(0, len(validated_texts), batch_size):
                    batch = validated_texts[i : i + batch_size]
                    batch_index = i // batch_size

                    try:
                        batch_tokens = sum(len(text.split()) for text in batch) * 1.3
                        total_tokens_used += batch_tokens

                        async with threading_service.rate_limited_operation(batch_tokens, None):
                            embedding_model = await get_embedding_model(provider=embedding_provider)
                            embeddings = await adapter.create_embeddings(batch, embedding_model, dimensions=dimensions_to_use)

                            for text, vector in zip(batch, embeddings, strict=False):
                                result.add_success(vector, text)

                    except Exception as e:
                        search_logger.error(f"Batch {batch_index} failed: {e}")
                        for text in batch:
                            result.add_failure(text, e, batch_index)

                    if progress_callback:
                        processed = result.success_count + result.failure_count
                        await progress_callback(f"Processed {processed}/{len(validated_texts)} texts", (processed / len(validated_texts)) * 100)

                return result

        except Exception as e:
            search_logger.error(f"Catastrophic failure in batch embedding: {e}", exc_info=True)
            return result


async def get_openai_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY")

# Provider-aware client factory
get_openai_client = get_llm_client
