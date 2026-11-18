"""OpenAI-compatible embedding proxy that batches concurrent requests.

MemMachine issues one embedding API call per query and per ingestion feature.
The upstream embedding gateway (api.gpt.ge) rate-limits concurrent HTTP
requests long before it saturates its own compute. Coalescing concurrent calls
into batches of up to EMBED_BATCH_MAX_ITEMS raises observed throughput from
~28 embeddings/s (with 429s) to 150+/s with no rate-limit errors.
"""

import asyncio
import hashlib
import logging
import os
import random
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("embed-proxy")

UPSTREAM_BASE_URL = os.environ.get(
    "EMBED_UPSTREAM_BASE_URL", "https://api.gpt.ge/v1"
).rstrip("/")
UPSTREAM_API_KEY = os.environ.get("EMBED_UPSTREAM_API_KEY", "")
BATCH_MAX_ITEMS = _int_env("EMBED_BATCH_MAX_ITEMS", 8)
BATCH_MAX_CHARS = _int_env("EMBED_BATCH_MAX_CHARS", 16000)
BATCH_WAIT_MS = float(os.environ.get("EMBED_BATCH_WAIT_MS", "20"))
MAX_INFLIGHT = _int_env("EMBED_MAX_INFLIGHT", 16)
MAX_ATTEMPTS = _int_env("EMBED_MAX_ATTEMPTS", 5)
CACHE_SIZE = _int_env("EMBED_CACHE_SIZE", 8192)
READ_TIMEOUT_SEC = float(os.environ.get("EMBED_READ_TIMEOUT_SEC", "60"))

# Upstream batch limit for text-embedding-v4 via api.gpt.ge is 10.
if BATCH_MAX_ITEMS > 10:
    logger.warning("EMBED_BATCH_MAX_ITEMS=%d exceeds upstream limit 10; clamping", BATCH_MAX_ITEMS)
    BATCH_MAX_ITEMS = 10


class UpstreamError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass
class _Item:
    key: tuple[str, int | None]
    text: str
    cache_key: tuple[str, int | None, bytes]
    waiters: list[asyncio.Future]


_stats = {
    "requests": 0,
    "items_queued": 0,
    "items_embedded": 0,
    "cache_hits": 0,
    "inflight_joined": 0,
    "batches": 0,
    "upstream_calls": 0,
    "upstream_retries": 0,
    "upstream_failures": 0,
}


class EmbeddingProxy:
    def __init__(self) -> None:
        self._pending: deque[_Item] = deque()
        self._available = asyncio.Event()
        self._client: httpx.AsyncClient | None = None
        self._sem: asyncio.Semaphore | None = None
        self._collector: asyncio.Task | None = None
        self._cache: OrderedDict = OrderedDict()
        self._inflight: dict[tuple, _Item] = {}

    async def start(self) -> None:
        if not UPSTREAM_API_KEY:
            logger.warning("EMBED_UPSTREAM_API_KEY is empty; upstream calls will fail")
        self._sem = asyncio.Semaphore(MAX_INFLIGHT)
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {UPSTREAM_API_KEY}",
                "Content-Type": "application/json",
            },
            limits=httpx.Limits(
                max_connections=MAX_INFLIGHT * 2 + 8,
                max_keepalive_connections=MAX_INFLIGHT + 8,
            ),
            timeout=httpx.Timeout(
                connect=10.0,
                read=READ_TIMEOUT_SEC,
                write=30.0,
                pool=30.0,
            ),
        )
        self._collector = asyncio.create_task(self._run_collector(), name="batch-collector")
        logger.info(
            "embedding proxy ready: upstream=%s batch_max=%d chars_max=%d "
            "inflight=%d wait_ms=%.0f cache=%d",
            UPSTREAM_BASE_URL,
            BATCH_MAX_ITEMS,
            BATCH_MAX_CHARS,
            MAX_INFLIGHT,
            BATCH_WAIT_MS,
            CACHE_SIZE,
        )

    async def stop(self) -> None:
        if self._collector is not None:
            self._collector.cancel()
            try:
                await self._collector
            except asyncio.CancelledError:
                pass
            self._collector = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _cache_get(self, key: tuple) -> list[float] | None:
        if CACHE_SIZE <= 0:
            return None
        value = self._cache.get(key)
        if value is not None:
            self._cache.move_to_end(key)
        return value

    def _cache_put(self, key: tuple, vector: list[float]) -> None:
        if CACHE_SIZE <= 0:
            return
        self._cache[key] = vector
        self._cache.move_to_end(key)
        while len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)

    async def embed_many(
        self,
        texts: list[str],
        model: str,
        dimensions: int | None,
    ) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        futures: list[asyncio.Future] = []
        for raw in texts:
            text = raw or "."
            cache_key = (model, dimensions, hashlib.sha256(text.encode()).digest())
            cached = self._cache_get(cache_key)
            if cached is not None:
                _stats["cache_hits"] += 1
                future = loop.create_future()
                future.set_result(cached)
            else:
                future = loop.create_future()
                shared = self._inflight.get(cache_key)
                if shared is not None:
                    shared.waiters.append(future)
                    _stats["inflight_joined"] += 1
                else:
                    item = _Item(
                        key=(model, dimensions),
                        text=text,
                        cache_key=cache_key,
                        waiters=[future],
                    )
                    self._inflight[cache_key] = item
                    self._pending.append(item)
                    _stats["items_queued"] += 1
                    self._available.set()
            futures.append(future)
        # Shield so a cancelled caller cannot cancel an embedding shared with
        # other in-flight requests.
        return list(await asyncio.gather(*(asyncio.shield(f) for f in futures)))

    async def _run_collector(self) -> None:
        while True:
            await self._available.wait()
            self._available.clear()

            while self._pending:
                first = self._pending.popleft()
                batch = [first]
                chars = len(first.text)

                # Give concurrent callers a brief window to join this batch.
                if not self._pending:
                    await asyncio.sleep(BATCH_WAIT_MS / 1000.0)

                while (
                    self._pending
                    and len(batch) < BATCH_MAX_ITEMS
                    and chars < BATCH_MAX_CHARS
                ):
                    nxt = self._pending[0]
                    if nxt.key != first.key:
                        break
                    self._pending.popleft()
                    batch.append(nxt)
                    chars += len(nxt.text)

                _stats["batches"] += 1
                asyncio.create_task(self._process_batch(batch), name="embed-batch")

    async def _process_batch(self, batch: list[_Item]) -> None:
        assert self._sem is not None
        async with self._sem:
            texts = [item.text for item in batch]
            try:
                vectors = await self._call_upstream(batch[0].key, texts)
            except UpstreamError as err:
                if err.status_code == 400 and len(batch) > 1:
                    # Isolate one or more bad inputs instead of failing the batch.
                    logger.warning(
                        "upstream rejected batch of %d, splitting: %s",
                        len(batch),
                        err.message[:200],
                    )
                    await asyncio.gather(*(self._process_single(item) for item in batch))
                    return
                logger.warning(
                    "embedding batch failed (%d items): %s",
                    len(batch),
                    err.message[:300],
                )
                for item in batch:
                    self._fail_item(item, err)
                return
            except Exception as err:  # pragma: no cover - defensive
                logger.exception("unexpected error while embedding batch")
                wrapped = UpstreamError(502, f"embedding proxy internal error: {err!r}")
                for item in batch:
                    self._fail_item(item, wrapped)
                return

            for item, vector in zip(batch, vectors):
                self._resolve_item(item, vector)

    async def _process_single(self, item: _Item) -> None:
        try:
            vectors = await self._call_upstream(item.key, [item.text])
        except UpstreamError as err:
            self._fail_item(item, err)
            return
        self._resolve_item(item, vectors[0])

    def _resolve_item(self, item: _Item, vector: list[float]) -> None:
        self._inflight.pop(item.cache_key, None)
        self._cache_put(item.cache_key, vector)
        for waiter in item.waiters:
            if not waiter.done():
                waiter.set_result(vector)
        _stats["items_embedded"] += 1

    def _fail_item(self, item: _Item, err: UpstreamError) -> None:
        self._inflight.pop(item.cache_key, None)
        for waiter in item.waiters:
            if not waiter.done():
                waiter.set_exception(err)

    async def _call_upstream(
        self,
        key: tuple[str, int | None],
        texts: list[str],
    ) -> list[list[float]]:
        assert self._client is not None
        model, dimensions = key
        payload: dict = {"model": model, "input": texts}
        if dimensions is not None:
            payload["dimensions"] = dimensions

        url = f"{UPSTREAM_BASE_URL}/embeddings"
        delay = 0.5
        last_error: UpstreamError | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = await self._client.post(url, json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as err:
                last_error = UpstreamError(502, f"upstream transport error: {err!r}")
                _stats["upstream_retries"] += 1
            else:
                if response.status_code == 200:
                    try:
                        rows = sorted(response.json()["data"], key=lambda row: row["index"])
                        vectors = [row["embedding"] for row in rows]
                    except (KeyError, TypeError, ValueError) as err:
                        raise UpstreamError(
                            502, f"malformed upstream response: {err!r}"
                        ) from err
                    if len(vectors) != len(texts):
                        raise UpstreamError(
                            502,
                            f"upstream returned {len(vectors)} vectors for {len(texts)} inputs",
                        )
                    _stats["upstream_calls"] += 1
                    return vectors

                body = response.text[:1000]
                if response.status_code in (400, 413, 422):
                    raise UpstreamError(response.status_code, body)
                last_error = UpstreamError(response.status_code, body)
                _stats["upstream_retries"] += 1

            if attempt < MAX_ATTEMPTS:
                await asyncio.sleep(delay + random.uniform(0, delay * 0.25))
                delay = min(delay * 2, 8.0)

        _stats["upstream_failures"] += 1
        status = last_error.status_code if last_error else 502
        message = last_error.message if last_error else "upstream embedding failed"
        raise UpstreamError(502 if status not in (400, 413, 422) else status, message)


proxy = EmbeddingProxy()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await proxy.start()
    yield
    await proxy.stop()


app = FastAPI(title="embedding-proxy", lifespan=lifespan)


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": "embedding_proxy_error",
                "code": status,
            }
        },
    )


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/stats")
async def stats() -> dict:
    return {
        **_stats,
        "pending": len(proxy._pending),
        "inflight": len(proxy._inflight),
        "cache_size": len(proxy._cache),
        "batch_max_items": BATCH_MAX_ITEMS,
        "max_inflight": MAX_INFLIGHT,
    }


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    _stats["requests"] += 1
    try:
        body = await request.json()
    except Exception:
        return _error(400, "invalid JSON body")

    if not isinstance(body, dict):
        return _error(400, "body must be a JSON object")

    model = body.get("model")
    dimensions = body.get("dimensions")
    raw_input = body.get("input")

    if isinstance(raw_input, str):
        texts = [raw_input]
    elif isinstance(raw_input, list) and all(isinstance(t, str) for t in raw_input):
        texts = raw_input
    else:
        return _error(400, "input must be a string or an array of strings")
    if not texts:
        return _error(400, "input must not be empty")
    if not isinstance(model, str) or not model:
        return _error(400, "model is required")
    if dimensions is not None and not isinstance(dimensions, int):
        return _error(400, "dimensions must be an integer")

    try:
        vectors = await proxy.embed_many(texts, model, dimensions)
    except UpstreamError as err:
        status = err.status_code if err.status_code in (400, 413, 422) else 502
        return _error(status, err.message)
    except Exception:
        logger.exception("embedding request failed")
        return _error(502, "embedding proxy failure")

    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ],
        "model": model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }
