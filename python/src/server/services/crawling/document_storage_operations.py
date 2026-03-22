"""
Document Storage Operations

Handles the storage and processing of crawled documents.
Extracted from crawl_orchestration_service.py for better modularity.
"""

import asyncio
import uuid
from collections.abc import Callable
from typing import Any

from ...config.logfire_config import get_logger, safe_logfire_error, safe_logfire_info
from ..source_management_service import extract_source_summary, update_source_info
from ..storage.document_storage_service import add_documents_to_supabase
from ..storage.storage_services import DocumentStorageService
from .code_extraction_service import CodeExtractionService
from .crawl_url_state_service import get_crawl_url_state_service
from .helpers.url_handler import URLHandler

logger = get_logger(__name__)


class DocumentStorageOperations:
    """
    Handles document storage operations for crawled content.
    """

    def __init__(self, supabase_client):
        """
        Initialize document storage operations.

        Args:
            supabase_client: The Supabase client for database operations
        """
        self.supabase_client = supabase_client
        self.doc_storage_service = DocumentStorageService(supabase_client)
        self.url_handler = URLHandler()
        self.code_extraction_service = CodeExtractionService(supabase_client)

    async def process_and_store_documents(
        self,
        crawl_results: list[dict],
        request: dict[str, Any],
        crawl_type: str,
        original_source_id: str,
        progress_callback: Callable | None = None,
        cancellation_check: Callable | None = None,
    ) -> dict[str, Any]:
        """
        Process crawled documents, chunk them, and store in database.

        Args:
            crawl_results: Results from crawling
            request: Original crawl request parameters
            crawl_type: Type of crawl (normal, sitemap, llms-txt)
            original_source_id: The source ID provided in the request
            progress_callback: Optional callback for progress updates
            cancellation_check: Optional function to check for cancellation

        Returns:
            Dictionary containing storage results (source_id, documents_stored, etc.)
        """
        if not crawl_results:
            return {"source_id": original_source_id, "chunks_stored": 0, "url_to_full_document": {}}

        storage_service = self.doc_storage_service

        all_urls = []
        all_chunk_numbers = []
        all_contents = []
        all_metadatas = []
        source_word_counts = {}
        url_to_full_document = {}
        processed_docs = 0

        # Process and chunk each document
        for doc_index, doc in enumerate(crawl_results):
            # Check for cancellation during document processing
            if cancellation_check:
                try:
                    cancellation_check()
                except asyncio.CancelledError:
                    if progress_callback:
                        await progress_callback(
                            "cancelled",
                            99,
                            f"Document processing cancelled at document {doc_index + 1}/{len(crawl_results)}",
                        )
                    raise

            doc_url = (doc.get("url") or "").strip()
            markdown_content = (doc.get("markdown") or "").strip()

            # Skip documents with empty or whitespace-only content or missing URLs
            if not markdown_content or not doc_url:
                logger.debug(f"Skipping document {doc_index}: empty {'URL' if not doc_url else 'content'}")
                continue

            # Increment processed document count
            processed_docs += 1

            # Store full document for code extraction context
            url_to_full_document[doc_url] = markdown_content

            # CHUNK THE CONTENT
            chunks = await storage_service.smart_chunk_text_async(markdown_content, chunk_size=5000)

            # Use the original source_id for all documents
            source_id = original_source_id
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                # Check for cancellation during chunk processing (every chunk)
                if cancellation_check:
                    try:
                        cancellation_check()
                    except asyncio.CancelledError:
                        if progress_callback:
                            await progress_callback(
                                "cancelled",
                                99,
                                f"Chunk processing cancelled at chunk {i + 1}/{len(chunks)} of document {doc_index + 1}",
                            )
                        raise

                all_urls.append(doc_url)
                all_chunk_numbers.append(i)
                all_contents.append(chunk)

                # Create metadata for each chunk
                word_count = len(chunk.split())
                metadata = {
                    "url": doc_url,
                    "title": doc.get("title", ""),
                    "description": doc.get("description", ""),
                    "source_id": source_id,
                    "knowledge_type": request.get("knowledge_type", "documentation"),
                    "page_id": None,
                    "crawl_type": crawl_type,
                    "word_count": word_count,
                    "char_count": len(chunk),
                    "chunk_index": i,
                    "tags": request.get("tags", []),
                }
                all_metadatas.append(metadata)

                # Accumulate word count
                source_word_counts[source_id] = source_word_counts.get(source_id, 0) + word_count

                if i > 0 and i % 10 == 0: await asyncio.sleep(0)
            if doc_index > 0 and doc_index % 5 == 0: await asyncio.sleep(0)

        # Create/update source record FIRST
        if all_contents and all_metadatas:
            # Gather URL info for source metadata
            source_url = request.get("url")
            source_display_name = self.url_handler.extract_display_name(source_url) if source_url else original_source_id
            
            await self._create_source_records(
                all_metadatas, all_contents, source_word_counts, request, source_url, source_display_name
            )

        # Store pages and chunks
        from .page_storage_operations import PageStorageOperations
        page_storage_ops = PageStorageOperations(self.supabase_client)

        # 1. Store pages
        # CRITICAL: Ensure we have a valid source_id for pages
        effective_source_id = original_source_id
        if not effective_source_id or effective_source_id == "None":
            effective_source_id = self.url_handler.generate_unique_source_id(request.get("url", ""))
            
        safe_logfire_info(f"Storing {len(url_to_full_document)} unique pages for source {effective_source_id}")
        url_to_page_id = await page_storage_ops.store_pages(
            crawl_results=crawl_results,
            source_id=effective_source_id,
            request=request,
            crawl_type=crawl_type,
        )

        # 2. Update chunk metadatas with page_ids and ensure source_id is set
        for metadata in all_metadatas:
            metadata["page_id"] = url_to_page_id.get(metadata["url"])
            if not metadata.get("source_id") or metadata.get("source_id") == "None":
                metadata["source_id"] = effective_source_id

        # 3. Store chunks
        safe_logfire_info(f"Storing {len(all_contents)} document chunks")
        if progress_callback:
            await progress_callback("document_storage", 50, f"Storing {len(all_contents)} chunks...")

        storage_results_final = await add_documents_to_supabase(
            client=self.supabase_client,
            urls=all_urls,
            chunk_numbers=all_chunk_numbers,
            contents=all_contents,
            metadatas=all_metadatas,
            url_to_full_document=url_to_full_document,
            progress_callback=progress_callback,
            cancellation_check=cancellation_check,
            url_to_page_id=url_to_page_id,
        )

        return {
            "source_id": effective_source_id,
            "chunks_stored": storage_results_final.get("chunks_stored", 0),
            "url_to_full_document": url_to_full_document,
            "url_to_page_id": url_to_page_id,
        }

    async def _create_source_records(
        self, all_metadatas, all_contents, source_id_word_counts, request, source_url, source_display_name
    ):
        unique_source_ids = set(m["source_id"] for m in all_metadatas)
        
        # Group content by source_id
        source_id_contents = {}
        for i, m in enumerate(all_metadatas):
            sid = m["source_id"]
            if sid not in source_id_contents: source_id_contents[sid] = []
            source_id_contents[sid].append(all_contents[i])

        for source_id in unique_source_ids:
            effective_sid = source_id
            if not effective_sid or effective_sid == "None":
                effective_sid = self.url_handler.generate_unique_source_id(request.get("url", ""))
                # Update metadatas
                for m in all_metadatas:
                    if m["source_id"] == source_id: m["source_id"] = effective_sid
                if source_id in source_id_word_counts:
                    source_id_word_counts[effective_sid] = source_id_word_counts.pop(source_id)
                if source_id in source_id_contents:
                    source_id_contents[effective_sid] = source_id_contents.pop(source_id)
            
            # Generate summary
            content_sample = " ".join(source_id_contents[effective_sid][:3])[:15000]
            try:
                summary = await extract_source_summary(effective_sid, content_sample)
            except Exception:
                summary = f"Documentation from {effective_sid}"

            # Update DB
            try:
                from ..credential_service import credential_service
                emb_config = await credential_service.get_credentials_by_category("embedding")
                
                await update_source_info(
                    client=self.supabase_client,
                    source_id=effective_sid,
                    summary=summary,
                    word_count=source_id_word_counts.get(effective_sid, 0),
                    content=content_sample,
                    knowledge_type=request.get("knowledge_type", "documentation"),
                    tags=request.get("tags", []),
                    original_url=request.get("url"),
                    source_url=source_url,
                    source_display_name=source_display_name,
                    embedding_model=emb_config.get("EMBEDDING_MODEL"),
                    embedding_dimensions=int(emb_config.get("EMBEDDING_DIMENSIONS", 1536)),
                    embedding_provider=emb_config.get("EMBEDDING_PROVIDER"),
                )
            except Exception as e:
                # Fallback
                fallback = {
                    "source_id": effective_sid,
                    "title": effective_sid,
                    "summary": summary,
                    "metadata": {"original_url": request.get("url")},
                    "updated_at": "now()",
                }
                self.supabase_client.table("archon_sources").upsert(fallback).execute()

    async def extract_and_store_code_examples(
        self,
        crawl_results: list[dict],
        url_to_full_document: dict[str, str],
        source_id: str,
        progress_callback: Callable | None = None,
        cancellation_check: Callable | None = None,
        provider: str = None,
        embedding_provider: str = None,
    ) -> int:
        return await self.code_extraction_service.extract_and_store_code_examples(
            crawl_results, url_to_full_document, source_id, progress_callback, cancellation_check, provider, embedding_provider
        )
