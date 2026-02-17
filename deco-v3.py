# tracing_decorators.py (Helper module)
"""
from infra.tracing_decorators import trace_stage

@trace_stage("agent.request")
async def ask_agent(...):
    ...
"""
import os
import inspect
from functools import wraps

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer("app.tracing.decorators")


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def trace_stage(operation: str):
    """
    Stage-level spans only (business / orchestration):
      - agent.request
      - agent.route
      - agent.aoai
      - agent.rag
      - tool.cosmos.write

    Best practices:
      - Avoid duplicating ultra-low-level spans if OpenLLMetry already instruments them.
      - Avoid putting high-cardinality IDs on every span by default.
      - Avoid attaching full exception messages unless explicitly enabled.
    """

    def decorator(fn):
        is_async = inspect.iscoroutinefunction(fn)

        if is_async:
            @wraps(fn)
            async def async_wrapper(*args, **kwargs):
                if _bool_env("OBS_DISABLE_STAGE_SPANS", False):
                    return await fn(*args, **kwargs)

                request = kwargs.get("request")
                with tracer.start_as_current_span(operation) as span:
                    _set_common_attrs(span, operation, request, kwargs)
                    try:
                        result = await fn(*args, **kwargs)
                        # OK is implied; set only on error to reduce noise.
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_attribute("error.type", type(e).__name__)
                        if _bool_env("OBS_INCLUDE_ERROR_MESSAGE", False):
                            span.set_attribute("error.message", str(e)[:500])
                        span.set_status(Status(StatusCode.ERROR))
                        raise

            return async_wrapper

        @wraps(fn)
        def sync_wrapper(*args, **kwargs):
            if _bool_env("OBS_DISABLE_STAGE_SPANS", False):
                return fn(*args, **kwargs)

            request = kwargs.get("request")
            with tracer.start_as_current_span(operation) as span:
                _set_common_attrs(span, operation, request, kwargs)
                try:
                    result = fn(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_attribute("error.type", type(e).__name__)
                    if _bool_env("OBS_INCLUDE_ERROR_MESSAGE", False):
                        span.set_attribute("error.message", str(e)[:500])
                    span.set_status(Status(StatusCode.ERROR))
                    raise

        return sync_wrapper

    return decorator


def _set_common_attrs(span, operation: str, request, kwargs):
    # Prefer span name for operation; this is optional and stable.
    span.set_attribute("ai.stage", operation)

    if request is not None:
        method = getattr(request, "method", None)
        url = getattr(request, "url", None)
        path = getattr(url, "path", None) if url else None

        if method:
            span.set_attribute("http.method", method)
        if path:
            span.set_attribute("http.route", path)

    agent_name = kwargs.get("agent_name")
    if agent_name:
        span.set_attribute("agent.name", str(agent_name))

    # High-cardinality IDs: disabled by default; enable only when debugging.
    include_ids = _bool_env("OBS_TRACE_INCLUDE_IDS", False)

    request_id = kwargs.get("request_id")
    if request_id and include_ids:
        span.set_attribute("request.id", str(request_id))

    correlation_id = kwargs.get("correlation_id")
    if correlation_id and include_ids:
        span.set_attribute("correlation.id", str(correlation_id))
