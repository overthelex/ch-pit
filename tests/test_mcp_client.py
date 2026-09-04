import json

import httpx
import pytest

from chpit import mcp_client
from chpit.provider import RetryableError

ARTICLE = {"sr_number": "220", "as_of": "2021-01-01",
           "version": {"version_id": 5, "date_applicability": "2021-01-01"},
           "article": {"e_id": "art_336", "article_number": "336", "text": "1 Die Kündigung ..."}}


class FakeServer:
    """Streamable-HTTP MCP: initialize hands out a session id, later calls
    must carry it; tools/call answers as SSE to exercise both parsers."""

    def __init__(self, fail_first_call: bool = False):
        self.requests: list[dict] = []
        self.fail_first_call = fail_first_call

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append({"body": body, "headers": dict(request.headers)})
        method = body["method"]
        if method == "initialize":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"],
                                             "result": {"serverInfo": {"name": "fake"}}},
                                  headers={"Mcp-Session-Id": "sess-1"})
        if request.headers.get("mcp-session-id") != "sess-1":
            return httpx.Response(400, json={"error": {"code": -32000, "message": "no session"}})
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "tools/list":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": {"tools": [
                {"name": "ch_get_act_article", "inputSchema": {"type": "object", "properties": {
                    "sr_number": {}, "canton": {}, "article": {}, "lang": {}, "as_of": {}},
                    "required": ["sr_number", "article"]}}]}})
        if method == "tools/call":
            if self.fail_first_call:
                self.fail_first_call = False
                return httpx.Response(503, text="busy")
            payload = {"jsonrpc": "2.0", "id": body["id"],
                       "result": {"content": [{"type": "text", "text": json.dumps(ARTICLE)}]}}
            return httpx.Response(200, text=f"event: message\ndata: {json.dumps(payload)}\n\n",
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(404)


def _client(server, tmp_path=None, bearer="tok"):
    return mcp_client.McpClient("https://mcp.example/v2/mcp", bearer=bearer,
                                cache_file=(tmp_path / "mcp-cache.jsonl") if tmp_path else None,
                                transport=httpx.MockTransport(server))


def test_initialize_captures_session_and_bearer_is_sent():
    srv = FakeServer()
    c = _client(srv)
    tools = c.list_tools()
    assert [t["name"] for t in tools] == ["ch_get_act_article"]
    methods = [r["body"]["method"] for r in srv.requests]
    assert methods == ["initialize", "notifications/initialized", "tools/list"]
    assert srv.requests[0]["headers"]["authorization"] == "Bearer tok"
    assert srv.requests[2]["headers"]["mcp-session-id"] == "sess-1"


def test_call_tool_parses_sse_and_caches_on_disk(tmp_path):
    srv = FakeServer()
    c = _client(srv, tmp_path)
    r1 = c.call_tool("ch_get_act_article", {"sr_number": "220", "article": "336", "lang": "de"})
    r2 = c.call_tool("ch_get_act_article", {"sr_number": "220", "article": "336", "lang": "de"})
    assert r1 == ARTICLE and r2 == ARTICLE
    assert sum(1 for r in srv.requests if r["body"]["method"] == "tools/call") == 1
    # a fresh client re-reads the cache and never hits the server for it
    srv2 = FakeServer()
    c2 = _client(srv2, tmp_path)
    assert c2.call_tool("ch_get_act_article", {"sr_number": "220", "article": "336", "lang": "de"}) == ARTICLE
    assert not any(r["body"]["method"] == "tools/call" for r in srv2.requests)
    assert (tmp_path / "mcp-cache.jsonl").read_text().count("\n") == 1


def test_cache_key_is_argument_order_independent():
    a = mcp_client.McpClient.cache_key("t", {"x": 1, "y": 2})
    b = mcp_client.McpClient.cache_key("t", {"y": 2, "x": 1})
    assert a == b


def test_transient_5xx_is_retried(monkeypatch):
    monkeypatch.setattr(mcp_client.time, "sleep", lambda s: None)
    c = _client(FakeServer(fail_first_call=True))
    assert c.call_tool_raw("ch_get_act_article", {"sr_number": "220", "article": "1"}) == ARTICLE


def test_401_is_a_clear_error():
    c = mcp_client.McpClient("https://mcp.example/v2/mcp", bearer="",
                             transport=httpx.MockTransport(lambda r: httpx.Response(401)))
    with pytest.raises(mcp_client.McpError, match="401"):
        c.list_tools()


def test_article_fetcher_passes_as_of_only_when_given():
    srv = FakeServer()
    c = _client(srv)
    fetch = mcp_client.article_fetcher(c)
    item = {"sr_number": "220", "article_number": "336", "lang": "fr"}
    fetch(item, None)
    fetch(item, "2021-01-01")
    calls = [r["body"]["params"]["arguments"] for r in srv.requests if r["body"]["method"] == "tools/call"]
    assert calls[0] == {"sr_number": "220", "article": "336", "lang": "fr"}
    assert calls[1]["as_of"] == "2021-01-01"


def test_retryable_error_type_is_shared_with_the_runner():
    assert issubclass(RetryableError, Exception)


def test_a_dropped_session_is_reopened_and_the_call_retried():
    class DroppingServer(FakeServer):
        def __init__(self):
            super().__init__()
            self.dropped = False

        def __call__(self, request):
            body = json.loads(request.content)
            if body["method"] == "tools/call" and not self.dropped:
                self.dropped = True
                return httpx.Response(400, json={"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}})
            return super().__call__(request)

    srv = DroppingServer()
    c = _client(srv)
    assert c.call_tool_raw("ch_get_act_article", {"sr_number": "220", "article": "1"}) == ARTICLE
    methods = [r["body"]["method"] for r in srv.requests]
    assert methods.count("initialize") == 2 and methods[-1] == "tools/call"


def test_concurrent_first_calls_open_exactly_one_session():
    import threading

    class StrictServer(FakeServer):
        def __call__(self, request):
            body = json.loads(request.content)
            if body["method"] == "initialize" and request.headers.get("mcp-session-id"):
                return httpx.Response(400, json={"jsonrpc": "2.0", "error": {
                    "code": -32600, "message": "Invalid Request: Server already initialized"}})
            return super().__call__(request)

    srv = StrictServer()
    c = _client(srv)
    errors = []

    def worker():
        try:
            c.call_tool_raw("ch_get_act_article", {"sr_number": "220", "article": "1"})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert errors == []
    assert [r["body"]["method"] for r in srv.requests].count("initialize") == 1
