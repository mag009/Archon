import os

# knowledge_api.py
path = "Archon/python/src/server/api_routes/knowledge_api.py"
with open(path, "r") as f:
    content = f.read()

old_func_def = """        async def document_progress_callback(message: str, percentage: int, batch_info: dict = None):
            \"\"\"Progress callback for tracking document processing\"\"\""""

new_func_def = """        async def document_progress_callback(
            message: str,
            percentage: int,
            batch_info: dict | None = None,
            **extra_fields,
        ):
            \"\"\"Progress callback for tracking document processing\"\"\""""

content = content.replace(old_func_def, new_func_def)

old_tracker_update = """                progress=mapped_percentage,
                log=message,
                currentUrl=f"file://{filename}",
                **(batch_info or {})
            )"""

new_tracker_update = """                progress=mapped_percentage,
                log=message,
                currentUrl=f"file://{filename}",
                **(batch_info or {}),
                **extra_fields,
            )"""

content = content.replace(old_tracker_update, new_tracker_update)

with open(path, "w") as f:
    f.write(content)


# main.py
path = "Archon/python/src/server/main.py"
with open(path, "r") as f:
    content = f.read()

old_main = """async def _check_database_schema():
    \"\"\"Check if required database schema exists - only for existing users who need migration.\"\"\"
    import time

    # If we've already confirmed schema is valid, don't check again
    if _schema_check_cache["valid"] is True:
        return {"valid": True, "message": "Schema is up to date (cached)"}

    # If we recently failed, don't spam the database (wait at least 30 seconds)
    current_time = time.time()
    if _schema_check_cache["valid"] is False and current_time - _schema_check_cache["checked_at"] < 30:
        return _schema_check_cache["result"]

    try:
        from .services.client_manager import get_supabase_client"""

new_main = """async def _check_database_schema_cached():
    \"\"\"
    Check if required database schema exists with caching.

    Returns immediately from cache when possible. Only checks database if:
    - Never checked before
    - Previous check was inconclusive (error)
    - Cached failure result is older than 30 seconds

    This function is designed to be fast and non-blocking for health checks.
    \"\"\"
    import time
    import asyncio

    # If we've already confirmed schema is valid, return immediately (no I/O)
    if _schema_check_cache["valid"] is True:
        return {"valid": True, "message": "Schema is up to date (cached)"}

    # If we recently failed, return cached result immediately (no I/O)
    current_time = time.time()
    if (_schema_check_cache["valid"] is False and
        current_time - _schema_check_cache["checked_at"] < 30):
        return _schema_check_cache["result"]

    # Need to check database - wrap in timeout to prevent blocking
    try:
        # Run database check with timeout
        result = await asyncio.wait_for(
            _perform_database_schema_check(current_time),
            timeout=1.5  # Database check itself must complete in 1.5s
        )
        return result
    except asyncio.TimeoutError:
        # Database check timed out - don't cache, allow retry
        api_logger.warning("Database schema check timed out")
        return {
            "valid": True,  # Assume valid to prevent health check failures
            "message": "Database check timed out",
            "timeout": True
        }


async def _perform_database_schema_check(current_time: float):
    \"\"\"Perform the actual database schema check (can be slow).\"\"\"
    try:
        from .services.client_manager import get_supabase_client"""

content = content.replace(old_main, new_main)

with open(path, "w") as f:
    f.write(content)


# document_storage_service.py
path = "Archon/python/src/server/services/storage/document_storage_service.py"
with open(path, "r") as f:
    content = f.read()

old_storage = """                # Determine the correct embedding column based on dimension
                embedding_dim = len(embedding) if isinstance(embedding, list) else len(embedding.tolist())
                embedding_column = None

                if embedding_dim == 768:
                    embedding_column = "embedding_768"
                elif embedding_dim == 1024:
                    embedding_column = "embedding_1024"
                elif embedding_dim == 1536:
                    embedding_column = "embedding_1536"
                elif embedding_dim == 3072:
                    embedding_column = "embedding_3072"
                else:
                    # Default to closest supported dimension
                    search_logger.warning(f"Unsupported embedding dimension {embedding_dim}, using embedding_1536")
                    embedding_column = "embedding_1536"

                # Get page_id for this URL if available"""

new_storage = """                # Determine the correct embedding column based on dimension
                embedding_dim = len(embedding) if isinstance(embedding, list) else len(embedding.tolist())
                embedding_column = determine_embedding_column(embedding_dim)
                if (
                    not legacy_column_in_use()
                    and embedding_column != f"embedding_{embedding_dim}"
                ):
                    search_logger.warning(
                        f"Unsupported embedding dimension {embedding_dim}, using {embedding_column}"
                    )

                # Get page_id for this URL if available"""

content = content.replace(old_storage, new_storage)

with open(path, "w") as f:
    f.write(content)
