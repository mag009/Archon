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

import tldextract

from ...config.logfire_config import get_logger, safe_logfire_error, safe_logfire_info
from ...utils import get_supabase_client
from ...utils.progress.progress_tracker import ProgressTracker
from ..credential_service import credential_service
from .crawl_url_state_service import get_crawl_url_state_service

# Import strategies
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
    Handles multi-part public suffixes correctly (e.g., .co.uk, .com.au).

    Args:
        host: Hostname to extract root domain from

    Returns:
        Root domain (domain + suffix) or original host if extraction fails

    Examples:
        - "docs.example.com" -> "example.com"
        - "api.example.co.uk" -> "example.co.uk"
        - "localhost" -> "localhost"
    """
    try:
        extracted = tldextract.extract(host)
        # Return domain.suffix if both are present
        if extracted.domain and extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}"
        # Fallback to original host if extraction yields no domain or suffix
        return host
    except Exception:
        # If extraction fails, return original host
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
    Combines functionality from both CrawlingService and CrawlOrchestrationService.
    """

    def __init__(self, crawler=None, supabase_client=None, progress_id=None):
        """
        Initialize the crawling service.

        Args:
            crawler: The Crawl4AI crawler instance
            supabase_client: The Supabase client for database operations
            progress_id: Optional progress ID for HTTP polling updates
        """
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

        # Track progress state across all stages to prevent UI resets
        self.progress_state = {"progressId": self.progress_id} if self.progress_id else {}
        # Initialize progress mapper to prevent backwards jumps
        self.progress_mapper = ProgressMapper()
        # Cancellation support
        self._cancelled = False
        self._cancellation_reason = CancellationReason.NONE

    def set_progress_id(self, progress_id: str):
        """Set the progress ID for HTTP polling updates."""
        self.progress_id = progress_id
        if self.progress_id:
            self.progress_state = {"progressId": self.progress_id}
            # Initialize progress tracker for HTTP polling
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
        """Create a progress callback for crawling operations.

        Args:
            base_status: The base status to use for progress updates

        Returns:
            Async callback function with signature (status: str, progress: int, message: str, **kwargs) -> None
        """

        async def callback(status: str, progress: int, message: str, **kwargs):
            if self.progress_tracker:
                # Debug log what we're receiving
                safe_logfire_info(
                    f"Progress callback received | status={status} | progress={progress} | "
                    f"total_pages={kwargs.get('total_pages', 'N/A')} | processed_pages={kwargs.get('processed_pages', 'N/A')} | "       
                    f"kwargs_keys={list(kwargs.keys())}"
                )

                # Map the progress to the overall progress range
                mapped_progress = self.progress_mapper.map_progress(base_status, progress)

                # Update progress via tracker (stores in memory for HTTP polling)
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
        """
        Crawl a batch of URLs with progress reporting.

        Args:
            urls: List of URLs to crawl
            max_concurrent: Maximum concurrent crawls (None uses DB settings)
            progress_callback: Optional progress callback
            link_text_fallbacks: Optional mapping of URL to link text for title fallback

        Returns:
            List of crawl results
        """
        return await self.batch_strategy.crawl_batch_with_progress(
            urls, max_concurrent, progress_callback, self._check_cancellation, link_text_fallbacks
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
        """
        Recursively crawl URLs with progress reporting.

        Args:
            start_urls: List of starting URLs
            max_depth: Maximum crawl depth
            max_concurrent: Maximum concurrent crawls (None uses DB settings)
            progress_callback: Optional progress callback
            source_id: Optional source ID for resume filtering
            url_state_service: Optional URL state service for checkpoint/resume
            include_patterns: Optional list of glob patterns to include
            exclude_patterns: Optional list of glob patterns to exclude

        Returns:
            List of crawl results
        """
        return await self.recursive_strategy.crawl_recursive_with_progress(
            start_urls,
            self.url_handler.transform_url,
            self.site_config.is_documentation_site,
            max_depth,
            max_concurrent,
            progress_callback,
            self._check_cancellation,  # Pass cancellation check
            source_id,
            url_state_service,
            include_patterns,
            exclude_patterns,
        )

    async def crawl_single_page(
        self, url: str, progress_callback: Callable[[str, int, str], Awaitable[None]] | None = None
    ) -> list[dict[str, Any]]:
        """
        Crawl a single page.

        Args:
            url: URL to crawl
            progress_callback: Optional progress callback

        Returns:
            List containing the crawl result
        """
        return await self.single_page_strategy.crawl_single_page(url, progress_callback, self._check_cancellation)

    async def crawl_markdown_file(
        self, url: str, progress_callback: Callable[[str, int, str], Awaitable[None]] | None = None
    ) -> list[dict[str, Any]]:
        """
        Crawl a markdown file directly.

        Args:
            url: URL of the markdown file
            progress_callback: Optional progress callback

        Returns:
            List containing the crawl result
        """
        return await self.single_page_strategy.crawl_markdown_file(url, progress_callback, self._check_cancellation)

    def parse_sitemap(self, sitemap_url: str) -> list[str]:
        """
        Parse a sitemap and return list of URLs.

        Args:
            sitemap_url: URL of the sitemap

        Returns:
            List of URLs found in the sitemap
        """
        return self.sitemap_strategy.parse_sitemap(sitemap_url, self._check_cancellation)

    async def orchestrate_crawl(self, request: dict[str, Any], task_id: str = None) -> str:
        """
        Orchestrate a crawling operation.
        This is the main entry point for starting a crawl.

        Args:
            request: The crawl request parameters
            task_id: Optional task ID for background task management

        Returns:
            The progress ID for tracking the operation
        """
        # Generate progress ID if not provided
        progress_id = self.progress_id or str(uuid.uuid4())
        self.set_progress_id(progress_id)

        # Register this orchestration for cancellation support
        await register_orchestration(progress_id, self)

        # Start the crawl in the background
        # We don't await this, it runs independently and updates progress via ProgressTracker
        asyncio.create_task(self._async_orchestrate_crawl(request, task_id))

        return progress_id

    async def _handle_progress_update(self, task_id: str, update: dict[str, Any]) -> None:
        """
        Handle progress updates from background task.

        Args:
            task_id: The task ID for the progress update
            update: The progress update data
        """
        if self.progress_tracker:
            status = update.get("status", "in_progress")
            progress = update.get("progress", 0)
            log = update.get("log", "")

            # Flatten additional fields
            extra_fields = {k: v for k, v in update.items() if k not in ["status", "progress", "log"]}

            await self.progress_tracker.update(status=status, progress=progress, log=log, **extra_fields)

    async def _async_orchestrate_crawl(self, request: dict[str, Any], task_id: str = None) -> None:
        """
        Background task for crawl orchestration.

        Args:
            request: The crawl request parameters
            task_id: Optional task ID
        """
        try:
            url = request.get("url")
            if not url:
                raise ValueError("URL is required")

            # Check for existing source state
            source_id = request.get("source_id")
            has_existing_state = request.get("resume", False) and source_id is not None

            # Helper to update progress using ProgressMapper
            async def update_mapped_progress(stage: str, stage_progress: int, log: str, **kwargs):
                if self.progress_tracker:
                    mapped = self.progress_mapper.map_progress(stage, stage_progress)
                    await self.progress_tracker.update(status=stage, progress=mapped, log=log, **kwargs)

            # Stage 1: Discovery (0-25%)
            await update_mapped_progress("starting", 0, f"Starting crawl orchestration for {url}")

            # Send initial heartbeat
            last_heartbeat = asyncio.get_event_loop().time()

            async def send_heartbeat_if_needed():
                nonlocal last_heartbeat
                current = asyncio.get_event_loop().time()
                if current - last_heartbeat > 5.0:  # Every 5 seconds
                    if self.progress_tracker:
                        await self.progress_tracker.update(status=None, progress=None, log=None)
                    last_heartbeat = current

            # Perform discovery if needed
            discovery_results = None
            if not has_existing_state:
                # Normal path: discover and crawl
                crawl_results, crawl_type = await self._crawl_by_url_type(url, request, source_id, has_existing_state)
            else:
                # Resume path: we have a source_id and want to continue
                # We skip discovery and go straight to crawling with the filtered list
                logger.info(f"Resuming crawl for source_id={source_id}")
                crawl_results, crawl_type = await self._crawl_by_url_type(url, request, source_id, has_existing_state)

            if not crawl_results:
                raise ValueError(f"No pages could be crawled from {url}")

            # Total pages for progress context
            total_pages = len(crawl_results)

            # Stage 2: Document Storage (25-40%)
            # The progress within document storage is handled by the operations class
            storage_results = await self.doc_storage_ops.store_crawled_pages(
                crawl_results,
                progress_callback=await self._create_crawl_progress_callback("document_storage"),
                cancellation_check=self._check_cancellation,
                source_id=source_id,
            )

            actual_chunks_stored = storage_results["chunks_stored"]

            # Stage 3: Code Extraction (40-90%)
            code_examples_count = 0
            if request.get("extract_code", True):
                # The progress within code extraction is handled by the operations class
                async def code_progress_callback(data: dict[str, Any]):
                    if self.progress_tracker:
                        # Extract percentage and message from callback data
                        raw_progress = data.get("percentage", 0)
                        mapped_progress = self.progress_mapper.map_progress("code_extraction", raw_progress)

                        # Update progress state via tracker
                        await self.progress_tracker.update(
                            status=data.get("status", "code_extraction"),
                            progress=mapped_progress,
                            log=data.get("log", "Extracting code examples..."),
                            total_pages=total_pages,  # Include total context
                            **{k: v for k, v in data.items() if k not in ["status", "progress", "percentage", "log"]},
                        )

                try:
                    # Extract provider from request or use credential service default
                    provider = request.get("provider")
                    embedding_provider = None

                    if not provider:
                        try:
                            provider_config = await credential_service.get_active_provider("llm")
                            provider = provider_config.get("provider", "openai")
                        except Exception as e:
                            logger.warning(f"Failed to get provider from credential service: {e}, defaulting to openai")
                            provider = "openai"

                    try:
                        embedding_config = await credential_service.get_active_provider("embedding")
                        embedding_provider = embedding_config.get("provider")
                    except Exception as e:
                        logger.warning(
                            f"Failed to get embedding provider from credential service: {e}. Using configured default."
                        )
                        embedding_provider = None

                    code_examples_count = await self.doc_storage_ops.extract_and_store_code_examples(
                        crawl_results,
                        storage_results["url_to_full_document"],
                        storage_results["source_id"],
                        code_progress_callback,
                        self._check_cancellation,
                        provider,
                        embedding_provider,
                    )
                except RuntimeError as e:
                    # Code extraction failed, continue crawl with warning
                    logger.error("Code extraction failed, continuing crawl without code examples", exc_info=True)
                    safe_logfire_error(f"Code extraction failed | error={e}")
                    code_examples_count = 0

                    # Report code extraction failure to progress tracker
                    if self.progress_tracker:
                        await self.progress_tracker.update(
                            status="code_extraction",
                            progress=self.progress_mapper.map_progress("code_extraction", 100),
                            log=f"Code extraction failed: {str(e)}. Continuing crawl without code examples.",
                            total_pages=total_pages,
                        )

                # Check for cancellation after code extraction
                self._check_cancellation()

                # Send heartbeat after code extraction
                await send_heartbeat_if_needed()

            # Finalization
            await update_mapped_progress(
                "finalization",
                50,
                "Finalizing crawl results...",
                chunks_stored=actual_chunks_stored,
                code_examples_found=code_examples_count,
            )

            # Complete - send both the progress update and completion event
            await update_mapped_progress(
                "completed",
                100,
                f"Crawl completed: {actual_chunks_stored} chunks, {code_examples_count} code examples",
                chunks_stored=actual_chunks_stored,
                code_examples_found=code_examples_count,
                processed_pages=len(crawl_results),
                total_pages=len(crawl_results),
            )

            # Mark crawl as completed
            if self.progress_tracker:
                await self.progress_tracker.complete(
                    {
                        "chunks_stored": actual_chunks_stored,
                        "code_examples_found": code_examples_count,
                        "processed_pages": len(crawl_results),
                        "total_pages": len(crawl_results),
                        "sourceId": storage_results.get("source_id", ""),
                        "log": "Crawl completed successfully!",
                    }
                )

            # Unregister after successful completion
            if self.progress_id:
                await unregister_orchestration(self.progress_id)
                safe_logfire_info(
                    f"Unregistered orchestration service after completion | progress_id={self.progress_id}"
                )

        except asyncio.CancelledError:
            # Determine final status based on cancellation reason
            if self._cancellation_reason == CancellationReason.PAUSED:
                final_status = "paused"
                log_message = "Crawl operation was paused by user"
                safe_logfire_info(f"Crawl operation paused | progress_id={self.progress_id}")
            else:
                # Default to cancelled for explicit stops or unknown reasons
                final_status = "cancelled"
                log_message = "Crawl operation was cancelled by user"
                safe_logfire_info(f"Crawl operation cancelled | progress_id={self.progress_id}")

            # Use ProgressMapper to get proper progress value
            final_progress = self.progress_mapper.map_progress(final_status, 0)

            await self._handle_progress_update(
                task_id,
                {
                    "status": final_status,
                    "progress": final_progress,
                    "log": log_message,
                },
            )

            # Unregister on cancellation
            if self.progress_id:
                await unregister_orchestration(self.progress_id)
                safe_logfire_info(
                    f"Unregistered orchestration service on {final_status} | progress_id={self.progress_id}"
                )
        except Exception as e:
            # Log full stack trace for debugging
            logger.error("Async crawl orchestration failed", exc_info=True)
            safe_logfire_error(f"Async crawl orchestration failed | error={str(e)}")
            error_message = f"Crawl failed: {str(e)}"
            # Use ProgressMapper to get proper progress value for error state
            error_progress = self.progress_mapper.map_progress("error", 0)
            await self._handle_progress_update(
                task_id, {"status": "error", "progress": error_progress, "log": error_message, "error": str(e)}
            )
            # Mark error in progress tracker with standardized schema
            if self.progress_tracker:
                await self.progress_tracker.error(error_message)
            # Unregister on error
            if self.progress_id:
                await unregister_orchestration(self.progress_id)
                safe_logfire_info(f"Unregistered orchestration service on error | progress_id={self.progress_id}")

    def _is_same_domain(self, url: str, base_domain: str) -> bool:
        """
        Check if a URL belongs to the same domain as the base domain.

        Args:
            url: URL to check
            base_domain: Base domain URL to compare against

        Returns:
            True if the URL is from the same domain
        """
        try:
            from urllib.parse import urlparse

            u, b = urlparse(url), urlparse(base_domain)
            url_host = (u.hostname or "").lower()
            base_host = (b.hostname or "").lower()
            return bool(url_host) and url_host == base_host
        except Exception:
            # If parsing fails, be conservative and exclude the URL
            return False

    def _is_same_domain_or_subdomain(self, url: str, base_domain: str) -> bool:
        """
        Check if a URL belongs to the same root domain or subdomain.

        Examples:
            - docs.supabase.com matches supabase.com (subdomain)
            - api.supabase.com matches supabase.com (subdomain)
            - supabase.com matches supabase.com (exact match)
            - external.com does NOT match supabase.com

        Args:
            url: URL to check
            base_domain: Base domain URL to compare against

        Returns:
            True if the URL is from the same root domain or subdomain
        """
        try:
            from urllib.parse import urlparse

            u, b = urlparse(url), urlparse(base_domain)
            url_host = (u.hostname or "").lower()
            base_host = (b.hostname or "").lower()

            if not url_host or not base_host:
                return False

            # Exact match
            if url_host == base_host:
                return True

            # Check if url_host is a subdomain of base_host using tldextract
            url_root = get_root_domain(url_host)
            base_root = get_root_domain(base_host)

            return url_root == base_root
        except Exception:
            # If parsing fails, be conservative and exclude the URL
            return False

    def _is_self_link(self, link: str, base_url: str) -> bool:
        """
        Check if a link is a self-referential link to the base URL.
        Handles query parameters, fragments, trailing slashes, and normalizes
        scheme/host/ports for accurate comparison.

        Args:
            link: The link to check
            base_url: The base URL to compare against

        Returns:
            True if the link is self-referential, False otherwise
        """
        try:
            from urllib.parse import urlparse

            def _core(u: str) -> str:
                p = urlparse(u)
                scheme = (p.scheme or "http").lower()
                host = (p.hostname or "").lower()
                port = p.port
                if (scheme == "http" and port in (None, 80)) or (scheme == "https" and port in (None, 443)):
                    port_part = ""
                else:
                    port_part = f":{port}" if port else ""
                path = p.path.rstrip("/")
                return f"{scheme}://{host}{port_part}{path}"

            return _core(link) == _core(base_url)
        except Exception as e:
            logger.warning(f"Error checking if link is self-referential: {e}", exc_info=True)
            # Fallback to simple string comparison
            return link.rstrip("/") == base_url.rstrip("/")

    async def _filter_already_processed_urls(self, source_id: str, urls: list[str]) -> list[str]:
        """
        Filter out URLs that are already embedded.

        Args:
            source_id: The source ID
            urls: List of URLs to filter

        Returns:
            List of URLs that have not been embedded yet
        """
        if not urls:
            return []

        url_state_service = get_crawl_url_state_service(self.supabase_client)

        # Get embedded URLs
        embedded_urls = url_state_service.get_embedded_urls(source_id)
        embedded_set = set(embedded_urls)

        # Filter
        filtered = [url for url in urls if url not in embedded_set]

        # Log resume info
        if len(filtered) < len(urls):
            skipped = len(urls) - len(filtered)
            safe_logfire_info(
                f"Resume filtering | skipped={skipped} already-embedded URLs | "
                f"remaining={len(filtered)} | source_id={source_id}",
                progress_id=self.progress_id,
            )

        return filtered

    async def _crawl_by_url_type(
        self, url: str, request: dict[str, Any], source_id: str | None = None, has_existing_state: bool = False
    ) -> tuple:
        """
        Detect URL type and perform appropriate crawling.

        Args:
            url: URL to crawl
            request: Crawl request parameters
            source_id: Optional source ID for resume filtering
            has_existing_state: Whether the source has existing crawl state

        Returns:
            Tuple of (crawl_results, crawl_type)
        """
        crawl_results = []
        crawl_type = None

        # Helper to update progress with mapper
        async def update_crawl_progress(stage_progress: int, message: str, **kwargs):
            if self.progress_tracker:
                mapped_progress = self.progress_mapper.map_progress("crawling", stage_progress)
                await self.progress_tracker.update(
                    status="crawling", progress=mapped_progress, log=message, current_url=url, **kwargs
                )

        if self.url_handler.is_txt(url) or self.url_handler.is_markdown(url):
            # Handle text files
            crawl_type = "llms-txt" if "llms" in url.lower() else "text_file"
            await update_crawl_progress(
                50,  # 50% of crawling stage
                "Detected text file, fetching content...",
                crawl_type=crawl_type,
            )
            crawl_results = await self.crawl_markdown_file(
                url,
                progress_callback=await self._create_crawl_progress_callback("crawling"),
            )
            # Check if this is a link collection file and extract links
            if crawl_results and len(crawl_results) > 0:
                content = crawl_results[0].get("markdown", "")
                if self.url_handler.is_link_collection_file(url, content):
                    # If this file was selected by discovery, check if it's an llms.txt file
                    if request.get("is_discovery_target"):
                        # Check if this is an llms.txt file (not sitemap or other discovery targets)
                        is_llms_file = self.url_handler.is_llms_variant(url)

                        if is_llms_file:
                            logger.info(f"Discovery llms.txt mode: following ALL same-domain links from {url}")

                            # Extract all links from the file
                            extracted_links_with_text = self.url_handler.extract_markdown_links_with_text(content, url)

                            # Filter for same-domain links (all types, not just llms.txt)
                            same_domain_links = []
                            if extracted_links_with_text:
                                original_domain = request.get("original_domain")
                                if original_domain:
                                    for link, text in extracted_links_with_text:
                                        # Check same domain/subdomain for ALL links
                                        if self._is_same_domain_or_subdomain(link, original_domain):
                                            same_domain_links.append((link, text))
                                            logger.debug(f"Found same-domain link: {link}")

                            # Apply glob pattern filtering or selected URLs
                            if same_domain_links:
                                original_count = len(same_domain_links)

                                # Extract filtering parameters from request
                                include_patterns = request.get("url_include_patterns", [])
                                exclude_patterns = request.get("url_exclude_patterns", [])
                                selected_urls = request.get("selected_urls")

                                # Option 1: Use selected_urls from review modal (takes precedence)
                                if selected_urls:
                                    selected_urls_set = set(selected_urls)
                                    same_domain_links = [
                                        (link, text) for link, text in same_domain_links
                                        if link in selected_urls_set
                                    ]
                                    logger.info(
                                        f"Applied selected_urls filter: {original_count} → {len(same_domain_links)} links "
                                        f"({original_count - len(same_domain_links)} filtered)"
                                    )

                                # Option 2: Apply glob pattern filtering
                                elif include_patterns or exclude_patterns:
                                    filtered_links = []
                                    for link, text in same_domain_links:
                                        if self.url_handler.matches_glob_patterns(link, include_patterns, exclude_patterns):
                                            filtered_links.append((link, text))

                                    filtered_count = original_count - len(filtered_links)
                                    same_domain_links = filtered_links

                                    logger.info(
                                        f"Applied glob pattern filter: {original_count} → {len(same_domain_links)} links "
                                        f"({filtered_count} filtered) | "
                                        f"include={include_patterns} | exclude={exclude_patterns}"
                                    )

                            if same_domain_links:
                                # Build mapping and extract just URLs
                                url_to_link_text = dict(same_domain_links)
                                extracted_urls = [link for link, _ in same_domain_links]

                                logger.info(f"Following {len(extracted_urls)} same-domain links from llms.txt")

                                # Notify user about linked files being crawled
                                await update_crawl_progress(
                                    60,  # 60% of crawling stage
                                    f"Found {len(extracted_urls)} links in llms.txt, crawling them now...",
                                    crawl_type="llms_txt_linked_files",
                                    linked_files=extracted_urls,
                                )

                                # Crawl all same-domain links from llms.txt (no recursion, just one level)
                                batch_results = await self.crawl_batch_with_progress(
                                    extracted_urls,
                                    max_concurrent=request.get("max_concurrent"),
                                    progress_callback=await self._create_crawl_progress_callback("crawling"),
                                    link_text_fallbacks=url_to_link_text,
                                )

                                # Combine original llms.txt with linked pages
                                crawl_results.extend(batch_results)
                                crawl_type = "llms_txt_with_linked_pages"
                                logger.info(
                                    f"llms.txt crawling completed: {len(crawl_results)} total pages (1 llms.txt + {len(batch_results)} linked pages)"
                                )
                                return crawl_results, crawl_type

                        # For non-llms.txt discovery targets (sitemaps, robots.txt), keep single-file mode
                        logger.info(f"Discovery single-file mode: skipping link extraction for {url}")
                        crawl_type = "discovery_single_file"
                        logger.info(f"Discovery file crawling completed: {len(crawl_results)} result")
                        return crawl_results, crawl_type

                    # Extract links WITH text from the content
                    extracted_links_with_text = self.url_handler.extract_markdown_links_with_text(content, url)

                    # Filter out self-referential links to avoid redundant crawling
                    if extracted_links_with_text:
                        original_count = len(extracted_links_with_text)
                        extracted_links_with_text = [
                            (link, text)
                            for link, text in extracted_links_with_text
                            if not self._is_self_link(link, url)
                        ]
                        self_filtered_count = original_count - len(extracted_links_with_text)
                        if self_filtered_count > 0:
                            logger.info(
                                f"Filtered out {self_filtered_count} self-referential links from {original_count} extracted links"      
                            )

                    # For discovery targets, only follow same-domain links
                    if extracted_links_with_text and request.get("is_discovery_target"):
                        original_domain = request.get("original_domain")
                        if original_domain:
                            original_count = len(extracted_links_with_text)
                            extracted_links_with_text = [
                                (link, text)
                                for link, text in extracted_links_with_text
                                if self._is_same_domain(link, original_domain)
                            ]
                            domain_filtered_count = original_count - len(extracted_links_with_text)
                            if domain_filtered_count > 0:
                                safe_logfire_info(
                                    f"Discovery mode: filtered out {domain_filtered_count} external links, keeping {len(extracted_links_with_text)} same-domain links"
                                )

                    # Filter out binary files (PDFs, images, archives, etc.) to avoid wasteful crawling
                    if extracted_links_with_text:
                        original_count = len(extracted_links_with_text)
                        extracted_links_with_text = [
                            (link, text)
                            for link, text in extracted_links_with_text
                            if not self.url_handler.is_binary_file(link)
                        ]
                        filtered_count = original_count - len(extracted_links_with_text)
                        if filtered_count > 0:
                            logger.info(
                                f"Filtered out {filtered_count} binary files from {original_count} extracted links"
                            )

                    if extracted_links_with_text:
                        # Build mapping of URL -> link text for title fallback
                        url_to_link_text = dict(extracted_links_with_text)
                        extracted_links = [link for link, _ in extracted_links_with_text]

                        # Apply resume filtering if we have existing state
                        if has_existing_state and source_id:
                            extracted_links = await self._filter_already_processed_urls(source_id, extracted_links)

                        # For discovery targets, respect max_depth for same-domain links
                        max_depth = (
                            request.get("max_depth", 2)
                            if request.get("is_discovery_target")
                            else request.get("max_depth", 1)
                        )

                        if max_depth > 1 and request.get("is_discovery_target"):
                            # Use recursive crawling to respect depth limit for same-domain links
                            logger.info(
                                f"Crawling {len(extracted_links)} same-domain links with max_depth={max_depth - 1}"
                            )
                            url_state_service = get_crawl_url_state_service(self.supabase_client) if source_id else None
                            batch_results = await self.crawl_recursive_with_progress(
                                extracted_links,
                                max_depth=max_depth - 1,  # Reduce depth since we're already 1 level deep
                                max_concurrent=request.get("max_concurrent"),
                                progress_callback=await self._create_crawl_progress_callback("crawling"),
                                source_id=source_id,
                                url_state_service=url_state_service,
                            )
                        else:
                            # Use normal batch crawling (with link text fallbacks)
                            logger.info(f"Crawling {len(extracted_links)} extracted links from {url}")
                            batch_results = await self.crawl_batch_with_progress(
                                extracted_links,
                                max_concurrent=request.get("max_concurrent"),  # None -> use DB settings
                                progress_callback=await self._create_crawl_progress_callback("crawling"),
                                link_text_fallbacks=url_to_link_text,  # Pass link text for title fallback
                            )

                        # Combine original text file results with batch results
                        crawl_results.extend(batch_results)
                        crawl_type = "link_collection_with_crawled_links"

                        logger.info(
                            f"Link collection crawling completed: {len(crawl_results)} total results (1 text file + {len(batch_results)} extracted links)"
                        )
                else:
                    logger.info(f"No valid links found in link collection file: {url}")
                    logger.info(f"Text file crawling completed: {len(crawl_results)} results")

        elif self.url_handler.is_sitemap(url):
            # Handle sitemaps
            crawl_type = "sitemap"
            await update_crawl_progress(
                50,  # 50% of crawling stage
                "Detected sitemap, parsing URLs...",
                crawl_type=crawl_type,
            )

            # If this sitemap was selected by discovery, just return the sitemap itself (single-file mode)
            if request.get("is_discovery_target"):
                logger.info(f"Discovery single-file mode: returning sitemap itself without crawling URLs from {url}")
                crawl_type = "discovery_sitemap"
                # Return the sitemap file as the result
                crawl_results = [
                    {
                        "url": url,
                        "markdown": f"# Sitemap: {url}\n\nThis is a sitemap file discovered and returned in single-file mode.",
                        "title": f"Sitemap - {self.url_handler.extract_display_name(url)}",
                        "crawl_type": crawl_type,
                    }
                ]
                return crawl_results, crawl_type

            sitemap_urls = self.parse_sitemap(url)

            if sitemap_urls:
                # 1. Apply resume filtering if enabled
                if has_existing_state and source_id:
                    sitemap_urls = await self._filter_already_processed_urls(source_id, sitemap_urls)

                if sitemap_urls:
                    original_count = len(sitemap_urls)

                    # 2. Apply glob pattern filtering or selected URLs
                    include_patterns = request.get("url_include_patterns", [])
                    exclude_patterns = request.get("url_exclude_patterns", [])
                    selected_urls = request.get("selected_urls")

                    # Option 1: Use selected_urls from review modal (takes precedence)
                    if selected_urls:
                        selected_urls_set = set(selected_urls)
                        sitemap_urls = [
                            u for u in sitemap_urls
                            if u in selected_urls_set
                        ]
                        logger.info(
                            f"Applied selected_urls filter to sitemap: {original_count} → {len(sitemap_urls)} URLs "
                            f"({original_count - len(sitemap_urls)} filtered)"
                        )

                    # Option 2: Apply glob pattern filtering
                    elif include_patterns or exclude_patterns:
                        filtered_urls = []
                        for sitemap_url in sitemap_urls:
                            if self.url_handler.matches_glob_patterns(sitemap_url, include_patterns, exclude_patterns):
                                filtered_urls.append(sitemap_url)

                        filtered_count = original_count - len(filtered_urls)
                        sitemap_urls = filtered_urls

                        logger.info(
                            f"Applied glob pattern filter to sitemap: {original_count} → {len(sitemap_urls)} URLs "
                            f"({filtered_count} filtered) | "
                            f"include={include_patterns} | exclude={exclude_patterns}"
                        )

                    if sitemap_urls:  # Only proceed if there are URLs left to crawl
                        # Update progress before starting batch crawl
                        await update_crawl_progress(
                            75,  # 75% of crawling stage
                            f"Starting batch crawl of {len(sitemap_urls)} URLs...",
                            crawl_type=crawl_type,
                        )

                        crawl_results = await self.crawl_batch_with_progress(
                            sitemap_urls,
                            progress_callback=await self._create_crawl_progress_callback("crawling"),
                        )
                    else:
                        logger.info("Pattern filtering: all sitemap URLs filtered out, nothing to crawl")
                else:
                    logger.info("Resume filtering: all sitemap URLs already embedded, nothing to crawl")

        else:
            # Handle regular webpages with recursive crawling
            crawl_type = "normal"
            await update_crawl_progress(
                50,  # 50% of crawling stage
                f"Starting recursive crawl with max depth {request.get('max_depth', 1)}...",
                crawl_type=crawl_type,
            )

            max_depth = request.get("max_depth", 1)
            include_patterns = request.get("url_include_patterns", [])
            exclude_patterns = request.get("url_exclude_patterns", [])

            # Log pattern configuration for debugging
            if include_patterns or exclude_patterns:
                logger.info(
                    f"Recursive crawl with glob patterns | "
                    f"include={include_patterns} | exclude={exclude_patterns}"
                )

            url_state_service = get_crawl_url_state_service(self.supabase_client) if source_id else None
            crawl_results = await self.crawl_recursive_with_progress(
                [url],
                max_depth=max_depth,
                max_concurrent=None,  # Let strategy use settings
                progress_callback=await self._create_crawl_progress_callback("crawling"),
                source_id=source_id,
                url_state_service=url_state_service,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
            )

        return crawl_results, crawl_type


# Alias for backward compatibility
CrawlOrchestrationService = CrawlingService
