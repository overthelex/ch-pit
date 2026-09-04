"""A minimal MCP client over streamable HTTP, for the point-in-time tools
the retrieval baselines call (`ch_get_act_article`, `ch_get_act_history`,
`ch_get_act_text` on mcp.lawrider.ch), plus a disk cache so eleven model
runs over the same items fetch each article once and every model sees
exactly the same context.

Protocol: JSON-RPC `initialize` (captures the `Mcp-Session-Id` header),
`notifications/initialized`, then `tools/list` and `tools/call`. The
server may answer with a plain JSON body or a single SSE frame; both are
parsed. A tool result is the JSON text of its first content part.
"""
from __future__ import annotations

import json
import pathlib
import threading
import time
from typing import Any, Callable

from chpit.provider import RetryableError

PROTOCOL_VERSION = "2025-03-26"
_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


class McpError(RuntimeError):
    pass


def _is_session_error(exc: Exception) -> bool:
    text = str(exc)
    return ("-32600" in text or "-32000" in text or "session" in text.lower()
            or "MCP HTTP 400" in text or "MCP HTTP 404" in text)


def _parse_body(text: str, content_type: str) -> dict[str, Any]:
    if "text/event-stream" in content_type:
        last: dict[str, Any] | None = None
        for line in text.splitlines():
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload:
                    last = json.loads(payload)
        if last is None:
            raise McpError("empty SSE response")
        return last
    return json.loads(text)


class McpClient:
    def __init__(self, url: str, bearer: str = "", timeout: float = 60.0,
                 cache_file: pathlib.Path | None = None, transport: Any = None,
                 client_name: str = "chpit"):
        import httpx
        self.url = url
        self.bearer = bearer
        self.client_name = client_name
        self._http = httpx.Client(timeout=timeout, transport=transport)
        self._session_id: str | None = None
        self._next_id = 1
        self._lock = threading.Lock()
        self._cache_file = cache_file
        self._cache: dict[str, dict[str, Any]] = {}
        if cache_file is not None and cache_file.exists():
            for line in cache_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._cache[row["key"]] = row["result"]

    # -- transport -------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json",
             "Accept": "application/json, text/event-stream"}
        if self.bearer:
            h["Authorization"] = f"Bearer {self.bearer}"
        if self._session_id:
            h["Mcp-Session-Id"] = self._session_id
        return h

    def _rpc(self, method: str, params: dict[str, Any] | None = None,
             notification: bool = False) -> dict[str, Any] | None:
        with self._lock:
            body: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
            if params is not None:
                body["params"] = params
            if not notification:
                body["id"] = self._next_id
                self._next_id += 1
            resp = self._http.post(self.url, json=body, headers=self._headers())
            if resp.status_code in _RETRY_STATUS:
                raise RetryableError(f"MCP HTTP {resp.status_code}: {resp.text[:200]}")
            if resp.status_code == 401:
                raise McpError("MCP 401: LAWRIDER_MCP_TOKEN missing or invalid")
            if resp.status_code >= 400:
                raise McpError(f"MCP HTTP {resp.status_code}: {resp.text[:300]}")
            sid = resp.headers.get("mcp-session-id")
            if sid:
                self._session_id = sid
            if notification or resp.status_code == 202 or not resp.content:
                return None
            data = _parse_body(resp.text, resp.headers.get("content-type", ""))
            if "error" in data:
                err = data["error"]
                raise McpError(f"MCP error {err.get('code')}: {err.get('message')}")
            return data.get("result")

    def initialize(self) -> dict[str, Any]:
        result = self._rpc("initialize", {
            "protocolVersion": PROTOCOL_VERSION, "capabilities": {},
            "clientInfo": {"name": self.client_name, "version": "1"}})
        self._rpc("notifications/initialized", {}, notification=True)
        return result or {}

    def _ensure(self) -> None:
        if self._session_id is None:
            self.initialize()

    def list_tools(self) -> list[dict[str, Any]]:
        self._ensure()
        result = self._rpc("tools/list", {}) or {}
        return list(result.get("tools", []))

    def call_tool_raw(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._ensure()
        for attempt in range(4):
            try:
                result = self._rpc("tools/call", {"name": name, "arguments": arguments}) or {}
                break
            except RetryableError:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
            except McpError as exc:
                # The server drops sessions (restart, eviction, another
                # replica): a -32600 / -32000 on a request that was fine a
                # moment ago means "no session", not "bad call". Open a new
                # session once and retry.
                if attempt < 3 and _is_session_error(exc):
                    self._session_id = None
                    self._ensure()
                    continue
                raise
        parts = result.get("content") or []
        text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        try:
            parsed = json.loads(text) if text else {}
        except json.JSONDecodeError:
            parsed = {"_text": text}
        if result.get("isError") and "error" not in parsed:
            parsed = {"error": "tool_error", "message": text[:500]}
        return parsed

    # -- cached calls ------------------------------------------------------
    @staticmethod
    def cache_key(name: str, arguments: dict[str, Any]) -> str:
        return json.dumps([name, arguments], sort_keys=True, ensure_ascii=False)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        key = self.cache_key(name, arguments)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        result = self.call_tool_raw(name, arguments)
        if result.get("error") == "tool_error":
            return result  # transient server-side failure: never cache it
        self._cache[key] = result
        if self._cache_file is not None:
            with self._lock:
                with self._cache_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "result": result}, ensure_ascii=False) + "\n")
        return result

    def close(self) -> None:
        self._http.close()


def article_fetcher(mcp: McpClient) -> Callable[[dict[str, Any], str | None], dict[str, Any]]:
    """FETCH(item, as_of) -> ch_get_act_article's JSON for the item's act and
    article in the item's language; `as_of=None` asks for today's edition."""

    def fetch(item: dict[str, Any], as_of: str | None) -> dict[str, Any]:
        args: dict[str, Any] = {"sr_number": item["sr_number"], "article": item["article_number"],
                                "lang": item["lang"]}
        if as_of:
            args["as_of"] = as_of
        return mcp.call_tool("ch_get_act_article", args)

    return fetch
