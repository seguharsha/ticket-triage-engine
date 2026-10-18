from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://triage:triage@localhost:5433/triage"
    redis_url: str = "redis://localhost:6379/0"
    log_level: str = "INFO"
    log_json: bool = True

    webhook_secret: str = "dev-secret-change-me"
    webhook_tolerance_seconds: int = 300

    attachment_dir: str = "./attachments"
    max_attachment_bytes: int = 5 * 1024 * 1024
    max_attachments_per_ticket: int = 5
    allowed_attachment_types: set[str] = {
        "image/png",
        "image/jpeg",
        "image/gif",
        "application/pdf",
        "text/plain",
    }

    # Worker: total attempts = 1 + max_retries, then the job goes to the dead-letter list.
    task_max_retries: int = 2
    warm_up_models: bool = True
    # Sweeper: re-enqueue tickets stuck in received (enqueue lost) or processing (worker died).
    requeue_interval_seconds: int = 30
    # Longer than a normal queue wait under burst load, so queued tickets aren't re-sent.
    requeue_received_after_seconds: int = 300
    requeue_processing_after_seconds: int = 600

    # Embeddings: "sentence-transformers" (real model) or "hashing" (fast, dependency-free; tests only).
    embedder: str = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # PyTorch threads per worker process (0 = one per core, which oversubscribes with several processes).
    torch_threads: int = 0

    # Duplicate detection. Tune dedup_threshold with scripts/eval_dedup.py.
    dedup_threshold: float = 0.75
    dedup_window_hours: int = 72
    hnsw_ef_search: int = 100

    # Classification + routing. Tune route_threshold with scripts/eval_classifier.py.
    classifier_path: str = "models/classifier.joblib"
    route_threshold: float = 0.7
    # Fields that must be confident to auto-route; priority is predicted but not used for routing.
    routing_fields: list[str] = ["category", "team"]
    llm_classify_fallback: bool = True

    # LLM: "groq", "ollama", "fake" (fixed answer after a delay; load tests) or "none".
    llm_provider: str = "fake"
    llm_timeout_seconds: float = 120
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    # Ollama's default context can be 2048 tokens, too small for a reply prompt with 5 sources.
    ollama_num_ctx: int = 4096
    # Keep the model loaded between calls.
    ollama_keep_alive: str = "30m"
    fake_llm_delay_ms: int = 300
    # USD per 1M tokens, used to report cost per ticket even when the provider is free.
    llm_price_input_per_mtok: float = 0.05
    llm_price_output_per_mtok: float = 0.08

    reply_top_k: int = 5
    reply_min_similarity: float = 0.5


settings = Settings()
