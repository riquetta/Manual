# V2
import inspect
from functools import wraps
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer("app.tracing.decorators")


def trace_stage(operation: str):
    def decorator(fn):
        is_async = inspect.iscoroutinefunction(fn)

        @wraps(fn)
        async def async_wrapper(*args, **kwargs):
            request = kwargs.get("request")
            with tracer.start_as_current_span(operation) as span:
                _set_common_attrs(span, operation, request, kwargs)
                try:
                    result = await fn(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_attribute("error", True)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        @wraps(fn)
        def sync_wrapper(*args, **kwargs):
            request = kwargs.get("request")
            with tracer.start_as_current_span(operation) as span:
                _set_common_attrs(span, operation, request, kwargs)
                try:
                    result = fn(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_attribute("error", True)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        return async_wrapper if is_async else sync_wrapper

    return decorator


def _set_common_attrs(span, operation: str, request, kwargs):
    span.set_attribute("operation.name", operation)

    if request is not None:
        method = getattr(request, "method", None)
        url = getattr(request, "url", None)
        path = getattr(url, "path", None) if url else None
        if method:
            span.set_attribute("http.method", method)
        if path:
            span.set_attribute("http.route", path)

    # V1 common attrs
    if kwargs.get("agent_name"):
        span.set_attribute("agent.name", str(kwargs["agent_name"]))
    if kwargs.get("request_id"):
        span.set_attribute("request.id", str(kwargs["request_id"]))
    if kwargs.get("correlation_id"):
        span.set_attribute("correlation.id", str(kwargs["correlation_id"]))

    # V2 optional attrs (only if provided)
    for k in (
        "workflow.name",
        "workflow.step",
        "tool.name",
        "tool.kind",
        "rag.index.name",
        "eval.name",
        "eval.metric",
        "gen_ai.request.model",
        "gen_ai.provider",
    ):
        if k in kwargs and kwargs[k] is not None:
            span.set_attribute(k, str(kwargs[k]))
