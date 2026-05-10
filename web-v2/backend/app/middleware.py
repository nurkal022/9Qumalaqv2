import json
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


log = logging.getLogger("web_v2")
if not log.handlers:
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
    log.propagate = False


class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception(json.dumps({
                "rid": rid,
                "path": request.url.path,
                "method": request.method,
                "status": 500,
            }))
            raise
        latency_ms = int((time.perf_counter() - t0) * 1000)
        sess = getattr(request.state, "session", None)
        if sess and getattr(sess, "user", None):
            owner = f"u{sess.user.id}"
        elif sess and getattr(sess, "anon", None):
            owner = f"a{sess.anon.id[:6]}"
        else:
            owner = "?"
        log.info(json.dumps({
            "rid": rid,
            "path": request.url.path,
            "method": request.method,
            "status": response.status_code,
            "latency_ms": latency_ms,
            "owner": owner,
        }))
        response.headers["X-Request-ID"] = rid
        return response
