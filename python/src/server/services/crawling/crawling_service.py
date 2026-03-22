"""
Crawling Service Module for Archon RAG

This module combines crawling functionality and orchestration.
It handles web crawling operations including single page crawling,
batch crawling, recursive crawling, and overall orchestration with progress tracking.
"""

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, Optional
from urllib.parse import urlparse

import tldextract

from ...config.logfire_config import get_logger, safe_logfire_error, safe_logfire_info
from ...utils import get_supabase_client
from ...utils.progress.progress_tracker import ProgressTracker
from ..credential_service import credential_service
from .crawl_url_state_service import get_crawl_url_state_service

# Import operations
from .discovery_service import DiscoveryService
from .document_storage_operations import DocumentStorageOperations
from .helpers.site_config import SiteConfig

# Import helpers
from .helpers.url_handler import URLHandler
from .page_storage_operations import PageStorageOperations
from .progress_mapper import ProgressMapper
from .strategies.batch import BatchCrawlStrategy
from .strategies.recursive import RecursiveCrawlStrategy
from .strategies.single_page import SinglePageCrawlStrategy
from .strategies.sitemap import SitemapCrawlStrategy

logger = get_logger(__name__)


class CancellationReason(Enum):
    """Tracks why a crawl was cancelled."""

    NONE = "none"  # Not cancelled
    PAUSED = "paused"  # User paused for later resume
    STOPPED = "stopped"  # User explicitly stopped/cancelled

# Global registry to track active orchestration services for cancellation support
_active_orchestrations: dict[str, "CrawlingService"] = {}
_orchestration_lock: asyncio.Lock | None = None


def get_root_domain(host: str) -> str:
    """
    Extract the root domain from a hostname using tldextract.
    """
    try:
        extracted = tldextract.extract(host)
        if extracted.domain and extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}"
        return host
    except Exception:
        return host


def _ensure_orchestration_lock() -> asyncio.Lock:
    global _orchestration_lock
    if _orchestration_lock is None:
        _orchestration_lock = asyncio.Lock()
    return _orchestration_lock


async def get_active_orchestration(progress_id: str) -> Optional["CrawlingService"]:
    """Get an active orchestration service by progress ID."""
    lock = _ensure_orchestration_lock()
    async with lock:
        return _active_orchestrations.get(progress_id)


async def register_orchestration(progress_id: str, orchestration: "CrawlingService"):
    """Register an active orchestration service."""
    lock = _ensure_orchestration_lock()
    async with lock:
        _active_orchestrations[progress_id] = orchestration


async def unregister_orchestration(progress_id: str):
    """Unregister an orchestration service."""
    lock = _ensure_orchestration_lock()
    async with lock:
        _active_orchestrations.pop(progress_id, None)


class CrawlingService:
    """
    Service class for web crawling and orchestration operations.
    """

    def __init__(self, crawler=None, supabase_client=None, progress_id=None):
        self.crawler = crawler
        self.supabase_client = supabase_client or get_supabase_client()
        self.progress_id = progress_id
        self.progress_tracker = None

        # Initialize helpers
        self.url_handler = URLHandler()
        self.site_config = SiteConfig()
        self.markdown_generator = self.site_config.get_markdown_generator()
        self.link_pruning_markdown_generator = self.site_config.get_link_pruning_markdown_generator()

        # Initialize strategies
        self.batch_strategy = BatchCrawlStrategy(crawler, self.link_pruning_markdown_generator)
        self.recursive_strategy = RecursiveCrawlStrategy(crawler, self.link_pruning_markdown_generator)
        self.single_page_strategy = SinglePageCrawlStrategy(crawler, self.markdown_generator)
        self.sitemap_strategy = SitemapCrawlStrategy()

        # Initialize operations
        self.doc_storage_ops = DocumentStorageOperations(self.supabase_client)
        self.discovery_service = DiscoveryService()
        self.page_storage_ops = PageStorageOperations(self.supabase_client)

        # Track progress state
        self.progress_state = {"progressId": self.progress_id} if self.progress_id else {}
        self.progress_mapper = ProgressMapper()
        
        # Cancellation support
        self._cancelled = False
        self._cancellation_reason = CancellationReason.NONE

    def set_progress_id(self, progress_id: str):
        """Set the progress ID for HTTP polling updates."""
        self.progress_id = progress_id
        if self.progress_id:
            self.progress_state = {"progressId": self.progress_id}
            self.progress_tracker = ProgressTracker(progress_id, operation_type="crawl")

    def cancel(self, reason: CancellationReason = CancellationReason.STOPPED):
        """Cancel the crawl operation with a specific reason."""
        self._cancelled = True
        self._cancellation_reason = reason
        safe_logfire_info(f"Crawl operation cancelled | progress_id={self.progress_id} | reason={reason.value}")

    def pause(self):
        """Pause the crawl operation for later resume."""
        self.cancel(reason=CancellationReason.PAUSED)

    def is_cancelled(self) -> bool:
        """Check if the crawl operation has been cancelled."""
        return self._cancelled

    def _check_cancellation(self):
        """Check if cancelled and raise an exception if so."""
        if self._cancelled:
            raise asyncio.CancelledError("Crawl operation was cancelled by user")

    async def _create_crawl_progress_callback(self, base_status: str) -> Callable[[str, int, str], Awaitable[None]]:
        async def callback(status: str, progress: int, message: str, **kwargs):
            if self.progress_tracker:
                mapped_progress = self.progress_mapper.map_progress(base_status, progress)
                await self.progress_tracker.update(
                    status=status,
                    progress=mapped_progress,
                    log=message,
                    **kwargs,
                )
        return callback

    async def crawl_batch_with_progress(
        self,
        urls: list[str],
        max_concurrent: int | None = None,
        progress_callback: Callable[[str, int, str], Awaitable[None]] | None = None,
        link_text_fallbacks: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        return await self.batch_strategy.crawl_batch_with_progress(
            urls,
            self.url_handler.transform_url,
            self.site_config.is_documentation_site,
            max_concurrent,
            progress_callback,
            self._check_cancellation,
            link_text_fallbacks
        )

    async def crawl_recursive_with_progress(
        self,
        start_urls: list[str],
        max_depth: int = 3,
        max_concurrent: int | None = None,
        progress_callback: Callable[[str, int, str], Awaitable[None]] | None = None,
        source_id: str | None = None,
        url_state_service: Any | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return await self.recursive_strategy.crawl_recursive_with_progress(
            start_urls,
            self.url_handler.transform_url,
            self.site_config.is_documentation_site,
            max_depth,
            max_concurrent,
            progress_callback,
            self._check_cancellation,
            source_id,
            url_state_service,
            include_patterns,
            exclude_patterns,
        )

    async def crawl_single_page(
        self, url: str, progress_callback: Callable[[str, int, str], Awaitable[None]] | None = None
    ) -> list[dict[str, Any]]:
        result = await self.single_page_strategy.crawl_single_page(
            url,
            self.url_handler.transform_url,
            self.site_config.is_documentation_site
        )
        if result.get("success"):
            return [result]
        else:
            logger.error(f"Failed to crawl single page {url}: {result.get('error')}")
            return []

    async def crawl_markdown_file(
        self, url: str, progress_callback: Callable[[str, int, str], Awaitable[None]] | None = None
    ) -> list[dict[str, Any]]:
        return await self.single_page_strategy.crawl_markdown_file(
            url,
            self.url_handler.transform_url,
            progress_callback=progress_callback
        )

    def parse_sitemap(self, sitemap_url: str) -> list[str]:
        return self.sitemap_strategy.parse_sitemap(sitemap_url, self._check_cancellation)

    async def orchestrate_crawl(self, request: dict[str, Any], task_id: str = None) -> dict[str, Any]:
        progress_id = self.progress_id or str(uuid.uuid4())
        self.set_progress_id(progress_id)
        await register_orchestration(progress_id, self)
        task = asyncio.create_task(self._async_orchestrate_crawl(request, task_id))
        return {"progressId": progress_id, "task": task, "task_id": task_id}

    async def _handle_progress_update(self, task_id: str, update: dict[str, Any]) -> None:
        if self.progress_tracker:
            status = update.get("status", "in_progress")
            progress = update.get("progress", 0)
            log = update.get("log", "")
            extra_fields = {k: v for k, v in update.items() if k not in ["status", "progress", "log"]}
            await self.progress_tracker.update(status=status, progress=progress, log=log, **extra_fields)

    async def _async_orchestrate_crawl(self, request: dict[str, Any], task_id: str = None) -> None:
        try:
            url = request.get("url")
            if not url: raise ValueError("URL is required")
            source_id = request.get("source_id")
            has_existing_state = request.get("resume", False) and source_id is not None

            async def update_mapped_progress(stage: str, stage_progress: int, log: str, **kwargs):
                if self.progress_tracker:
                    mapped = self.progress_mapper.map_progress(stage, stage_progress)
                    await self.progress_tracker.update(status=stage, progress=mapped, log=log, **kwargs)

            await update_mapped_progress("starting", 0, f"Starting crawl orchestration for {url}", current_url=url, source_id=source_id)

            # Perform discovery or resume
            crawl_results, crawl_type = await self._crawl_by_url_type(url, request, source_id, has_existing_state)

            if not crawl_results: raise ValueError(f"No pages could be crawled from {url}")

            storage_results = await self.doc_storage_ops.process_and_store_documents(
                crawl_results=crawl_results,
                request=request,
                crawl_type=crawl_type,
                original_source_id=source_id,
                progress_callback=await self._create_crawl_progress_callback("document_storage"),
                cancellation_check=self._check_cancellation,
            )

            actual_chunks_stored = storage_results["chunks_stored"]
            
            # Code Extraction
            code_examples_count = 0
            if request.get("extract_code", True):
                async def code_progress_callback(data: dict[str, Any]):
                    if self.progress_tracker:
                        raw_progress = data.get("percentage", 0)
                        mapped_progress = self.progress_mapper.map_progress("code_extraction", raw_progress)
                        await self.progress_tracker.update(
                            status=data.get("status", "code_extraction"),
                            progress=mapped_progress,
                            log=data.get("log", "Extracting code examples..."),
                            **{k: v for k, v in data.items() if k not in ["status", "progress", "percentage", "log"]},
                        )

                provider = request.get("provider")
                if not provider:
                    provider_config = await credential_service.get_active_provider("llm")
                    provider = provider_config.get("provider", "openai")

                embedding_config = await credential_service.get_active_provider("embedding")
                embedding_provider = embedding_config.get("provider")

                code_examples_count = await self.doc_storage_ops.extract_and_store_code_examples(
                    crawl_results,
                    storage_results["url_to_full_document"],
                    storage_results["source_id"],
                    code_progress_callback,
                    self._check_cancellation,
                    provider,
                    embedding_provider,
                )

            await update_mapped_progress("completed", 100, "Crawl completed", chunks_stored=actual_chunks_stored, code_examples_found=code_examples_count)
            if self.progress_tracker:
                await self.progress_tracker.complete({"sourceId": storage_results.get("source_id", "")})
            if self.progress_id:
                await unregister_orchestration(self.progress_id)

        except asyncio.CancelledError:
            final_status = "paused" if self._cancellation_reason == CancellationReason.PAUSED else "cancelled"
            if self.progress_tracker:
                await self.progress_tracker.update(status=final_status, progress=self.progress_mapper.map_progress(final_status, 0), log=f"Crawl {final_status} by user")
            if self.progress_id:
                await unregister_orchestration(self.progress_id)
        except Exception as e:
            logger.error("Async crawl orchestration failed", exc_info=True)
            if self.progress_tracker:
                await self.progress_tracker.error(f"Crawl failed: {str(e)}")
            if self.progress_id:
                await unregister_orchestration(self.progress_id)

    async def _crawl_by_url_type(self, url: str, request: dict[str, Any], source_id: str | None = None, has_existing_state: bool = False) -> tuple:
        crawl_results = []
        crawl_type = None

        async def update_crawl_progress(stage_progress: int, message: str, **kwargs):
            if self.progress_tracker:
                mapped_progress = self.progress_mapper.map_progress("crawling", stage_progress)
                await self.progress_tracker.update(status="crawling", progress=mapped_progress, log=message, current_url=url, **kwargs)

        if self.url_handler.is_txt(url) or self.url_handler.is_markdown(url) or self.url_handler.is_github_directory(url):
            crawl_type = "llms-txt" if "llms" in url.lower() else "text_file"
            if self.url_handler.is_github_directory(url):
                crawl_type = "github_directory"
                # For GitHub directories, use normal crawl to get the UI page first
                crawl_results = await self.crawl_single_page(url, progress_callback=await self._create_crawl_progress_callback("crawling"))
            else:
                crawl_results = await self.crawl_markdown_file(url, progress_callback=await self._create_crawl_progress_callback("crawling"))

            if crawl_results and len(crawl_results) > 0:
                # Use HTML content for link extraction if it's a GitHub directory, otherwise markdown
                content = crawl_results[0].get("html", "") if crawl_type == "github_directory" else crawl_results[0].get("markdown", "")

                # Check if it's a link collection or a GitHub directory (which acts as one)
                if crawl_type == "github_directory" or self.url_handler.is_link_collection_file(url, content):
                    if crawl_type == "github_directory":
                        # GitHub directory links are in the HTML
                        # Crawl4AI already extracted them into result['links']
                        links_data = crawl_results[0].get("links", {}) or {}
                        extracted_links_with_text = []
                        
                        # Normalize current URL for path comparison
                        current_path = urlparse(url).path.rstrip('/')
                        
                        for link in links_data.get("internal", []):
                            href = link.get("href", "")
                            link_path = urlparse(href).path.rstrip('/')
                            
                            # Include 'blob' URLs (files)
                            if "/blob/" in href:
                                extracted_links_with_text.append((href, link.get("text", "")))
                            # Include 'tree' URLs (subdirectories) ONLY if they are deeper in the same path
                            elif "/tree/" in href and link_path.startswith(current_path + '/'):
                                extracted_links_with_text.append((href, link.get("text", "")))
                    else:
                        extracted_links_with_text = self.url_handler.extract_markdown_links_with_text(content, url)

                    if not extracted_links_with_text: return crawl_results, crawl_type
                    # Filter links
                    original_domain = request.get("original_domain")
                    filtered_links = []
                    for link, text in extracted_links_with_text:
                        if original_domain and not self._is_same_domain_or_subdomain(link, original_domain): continue
                        if self._is_self_link(link, url): continue
                        if self.url_handler.is_binary_file(link): continue
                        
                        # Apply selection and pattern filters
                        selected_urls = request.get("selected_urls")
                        if selected_urls and link not in set(selected_urls): continue
                        
                        include_patterns = request.get("url_include_patterns", [])
                        exclude_patterns = request.get("url_exclude_patterns", [])
                        if (include_patterns or exclude_patterns) and not self.url_handler.matches_glob_patterns(link, include_patterns, exclude_patterns): continue
                        
                        filtered_links.append((link, text))

                    if not filtered_links: return crawl_results, crawl_type

                    # Start crawl of extracted links
                    extracted_urls = [l for l, _ in filtered_links]
                    url_to_link_text = dict(filtered_links)
                    
                    await update_crawl_progress(60, f"Found {len(extracted_urls)} filtered links, crawling now...")
                    
                    max_depth = request.get("max_depth", 1)
                    if max_depth > 1:
                        url_state_service = get_crawl_url_state_service(self.supabase_client) if source_id else None
                        batch_results = await self.crawl_recursive_with_progress(extracted_urls, max_depth=max_depth-1, source_id=source_id, url_state_service=url_state_service, progress_callback=await self._create_crawl_progress_callback("crawling"))
                    else:
                        batch_results = await self.crawl_batch_with_progress(extracted_urls, progress_callback=await self._create_crawl_progress_callback("crawling"), link_text_fallbacks=url_to_link_text)
                    
                    crawl_results.extend(batch_results)
                    crawl_type = "link_collection_with_linked_pages"
                    return crawl_results, crawl_type

        elif self.url_handler.is_sitemap(url):
            crawl_type = "sitemap"
            sitemap_urls = self.parse_sitemap(url)
            if sitemap_urls:
                selected_urls = request.get("selected_urls")
                if selected_urls:
                    sitemap_urls = [u for u in sitemap_urls if u in set(selected_urls)]
                
                if sitemap_urls:
                    crawl_results = await self.crawl_batch_with_progress(sitemap_urls, progress_callback=await self._create_crawl_progress_callback("crawling"))
        
        else:
            crawl_type = "normal"
            max_depth = request.get("max_depth", 1)
            url_state_service = get_crawl_url_state_service(self.supabase_client) if source_id else None
            crawl_results = await self.crawl_recursive_with_progress([url], max_depth=max_depth, source_id=source_id, url_state_service=url_state_service, progress_callback=await self._create_crawl_progress_callback("crawling"))

        return crawl_results, crawl_type

    def _is_same_domain_or_subdomain(self, url: str, base_domain: str) -> bool:
        try:
            u, b = urlparse(url), urlparse(base_domain)
            url_host, base_host = (u.hostname or "").lower(), (b.hostname or "").lower()
            if not url_host or not base_host: return False
            if url_host == base_host: return True
            return get_root_domain(url_host) == get_root_domain(base_host)
        except Exception: return False

    def _is_self_link(self, link: str, base_url: str) -> bool:
        return link.rstrip("/") == base_url.rstrip("/")

    async def _filter_already_processed_urls(self, source_id: str, urls: list[str]) -> list[str]:
        if not urls: return []
        url_state_service = get_crawl_url_state_service(self.supabase_client)
        embedded_urls = set(url_state_service.get_embedded_urls(source_id))
        return [url for url in urls if url not in embedded_urls]

CrawlOrchestrationService = CrawlingService
