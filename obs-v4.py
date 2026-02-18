# observability.py (V2 production-grade)
import os
import atexit
from dataclasses import dataclass
from typing import Dict, Optional

from opentelemetry import trace, metrics
from opentelemetry.trace import TracerProvider as APITracerProvider
from opentelemetry.metrics import MeterProvider as APIMeterProvider

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider as SDKMeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased


# -----------------------------
# Public state container
# -----------------------------
@dataclass
class TelemetryState:
    initialized: bool = False
    tracer = None
    meter = None

    # V1
    req_counter = None
    err_counter = None
    latency_hist = None
    token_counter = None
    cost_counter = None

    # V2 RAG
    rag_req_counter = None
    rag_latency_hist = None
    rag_docs_counter = None
    rag_rerank_latency_hist = None
    rag_context_tokens_counter = None
    rag_no_result_counter = None

    # V2 tools
    tool_calls_counter = None
    tool_latency_hist = None
    tool_errors_counter = None
    tool_timeout_counter = None

    # V2 model orchestration
    model_invocations_counter = None
    model_latency_hist = None
    model_fallback_counter = None
    model_selection_counter = None

    # V2 evals
    eval_runs_counter = None
    eval_pass_counter = None
    eval_fail_counter = None
    eval_score_hist = None
    eval_latency_hist = None

    # V2 streaming
    stream_responses_counter = None
    stream_ttft_hist = None
    stream_tokens_counter = None


STATE = TelemetryState()

# NEW
import threading
_INIT_LOCK = threading.Lock()

MODEL_PRICING_PER_1K = {
    # "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.00060},
    # "gpt-4o": {"prompt": 0.00500, "completion": 0.01500},
}


def _safe_print(msg: str) -> None:
    print(msg, flush=True)


def _safe_bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_int_env(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or not str(v).strip():
        return default
    try:
        return int(str(v).strip())
    except ValueError:
        return default


def _safe_float_env(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or not str(v).strip():
        return default
    try:
        return float(str(v).strip())
    except ValueError:
        return default


def _parse_headers(headers_str: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not headers_str:
        return out
    for part in [p.strip() for p in headers_str.split(",") if p.strip()]:
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _is_sdk_tracer_provider(provider: APITracerProvider) -> bool:
    return isinstance(provider, SDKTracerProvider)


def _is_sdk_meter_provider(provider: APIMeterProvider) -> bool:
    return isinstance(provider, SDKMeterProvider)


def _has_span_processor(tp: SDKTracerProvider, processor_type_name: str) -> bool:
    active = getattr(tp, "_active_span_processor", None)
    if active is None:
        return False
    children = getattr(active, "_span_processors", None)
    if children:
        return any(getattr(p, "__class__", type("X", (), {})).__name__ == processor_type_name for p in children)
    return getattr(active, "__class__", type("X", (), {})).__name__ == processor_type_name


def _init_traceloop_if_enabled() -> None:
    """
    Traceloop is optional. If you use OpenLLMetry/Traceloop, prefer:
      - TRACELOOP_DISABLE_BATCH=true (so OTel BSP is your single batching layer)
    """
    enabled = _safe_bool_env("TRACELOOP_ENABLED", False)
    if not enabled:
        _safe_print("[telemetry] Traceloop disabled")
        return

    try:
        from traceloop.sdk import Traceloop  # type: ignore
    except Exception as ex:
        _safe_print(f"[telemetry] Traceloop import failed: {ex}")
        return

    app_name = os.getenv("TRACELOOP_APP_NAME", os.getenv("OTEL_SERVICE_NAME", "agent_gateway"))
    disable_batch = _safe_bool_env("TRACELOOP_DISABLE_BATCH", True)

    try:
        Traceloop.init(app_name=app_name, disable_batch=disable_batch)
        _safe_print(f"[telemetry] Traceloop initialized app_name={app_name} disable_batch={disable_batch}")
    except Exception as ex:
        _safe_print(f"[telemetry] Traceloop init failed: {ex}")


def init_observability() -> None:
    """
    Production-grade init:
      - Explicit trace batching knobs (BSP)
      - Explicit sampling ratio
      - Metrics export interval configurable
      - Avoid high-cardinality IDs in *metrics* by default
    """
    if STATE.initialized:
        return

    # Thread-safe init-once (NEW)
    with _INIT_LOCK:
        if STATE.initialized:
            return

    _init_traceloop_if_enabled()

    service_name = os.getenv("OTEL_SERVICE_NAME", "agent_gateway")
    service_namespace = os.getenv("OTEL_SERVICE_NAMESPACE", "ai_platform")
    env = os.getenv("APP_ENV", "dev")

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.namespace": service_namespace,
            "deployment.environment": env,
        }
    )

    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    otlp_headers = _parse_headers(os.getenv("OTEL_EXPORTER_OTLP_HEADERS", ""))
    debug_console = _safe_bool_env("OBS_DEBUG_CONSOLE", False)

    _safe_print("[telemetry] init_observability called")
    _safe_print(f"[telemetry] service.name={service_name}")
    _safe_print(f"[telemetry] endpoint_set={bool(otlp_endpoint)} headers_set={bool(otlp_headers)}")
    _safe_print(f"[telemetry] debug_console={debug_console}")

    # -------- Tracing provider --------
    current_tp = trace.get_tracer_provider()
    if _is_sdk_tracer_provider(current_tp):
        tracer_provider: SDKTracerProvider = current_tp
        _safe_print("[telemetry] reusing existing SDK tracer provider")
    else:
        # Sampling: ParentBased(TraceIdRatioBased)
        # Default: 100% in dev, 10% otherwise (override with OTEL_TRACES_SAMPLER_ARG)
        default_ratio = 1.0 if env.lower() in {"dev", "local"} else 0.1
        sample_ratio = _safe_float_env("OTEL_TRACES_SAMPLER_ARG", default_ratio)
        if sample_ratio < 0:
            sample_ratio = 0.0
        if sample_ratio > 1:
            sample_ratio = 1.0

        sampler = ParentBased(TraceIdRatioBased(sample_ratio))
        tracer_provider = SDKTracerProvider(resource=resource, sampler=sampler)
        trace.set_tracer_provider(tracer_provider)
        _safe_print(f"[telemetry] created new SDK tracer provider sampler_ratio={sample_ratio}")

    # -------- Trace exporters/processors --------
    if otlp_endpoint:
        try:
            if not _has_span_processor(tracer_provider, "BatchSpanProcessor"):
                span_exporter = OTLPSpanExporter(
                    endpoint=f"{otlp_endpoint.rstrip('/')}/v1/traces",
                    headers=otlp_headers,
                    timeout=_safe_int_env("OTEL_EXPORTER_OTLP_TIMEOUT_S", 10),
                )

                tracer_provider.add_span_processor(
                    BatchSpanProcessor(
                        span_exporter,
                        schedule_delay_millis=_safe_int_env("OTEL_BSP_SCHEDULE_DELAY_MS", 5000),
                        max_export_batch_size=_safe_int_env("OTEL_BSP_MAX_EXPORT_BATCH_SIZE", 512),
                        max_queue_size=_safe_int_env("OTEL_BSP_MAX_QUEUE_SIZE", 2048),
                        export_timeout_millis=_safe_int_env("OTEL_BSP_EXPORT_TIMEOUT_MS", 30000),
                    )
                )
                _safe_print("[telemetry] OTLP trace exporter attached (BatchSpanProcessor)")
            else:
                _safe_print("[telemetry] BatchSpanProcessor already present (skip)")
        except Exception as ex:
            _safe_print(f"[telemetry] OTLP trace exporter attach failed: {ex}")

    if debug_console:
        try:
            if not _has_span_processor(tracer_provider, "SimpleSpanProcessor"):
                tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
                _safe_print("[telemetry] Console trace exporter attached")
            else:
                _safe_print("[telemetry] SimpleSpanProcessor already present (skip)")
        except Exception as ex:
            _safe_print(f"[telemetry] Console trace exporter attach failed: {ex}")

    # -------- Metrics provider --------
    current_mp = metrics.get_meter_provider()
    if _is_sdk_meter_provider(current_mp):
        meter_provider: SDKMeterProvider = current_mp
        _safe_print("[telemetry] reusing existing SDK meter provider")
    else:
        readers = []

        if otlp_endpoint:
            try:
                metric_exporter = OTLPMetricExporter(
                    endpoint=f"{otlp_endpoint.rstrip('/')}/v1/metrics",
                    headers=otlp_headers,
                    timeout=_safe_int_env("OTEL_EXPORTER_OTLP_TIMEOUT_S", 10),
                )

                export_interval = _safe_int_env("OTEL_METRIC_EXPORT_INTERVAL_MS", 10000)
                readers.append(
                    PeriodicExportingMetricReader(
                        metric_exporter,
                        export_interval_millis=export_interval,
                    )
                )
                _safe_print(f"[telemetry] OTLP metric exporter attached interval_ms={export_interval}")
            except Exception as ex:
                _safe_print(f"[telemetry] OTLP metric exporter attach failed: {ex}")

        if debug_console:
            try:
                readers.append(
                    PeriodicExportingMetricReader(
                        ConsoleMetricExporter(),
                        export_interval_millis=_safe_int_env("OTEL_CONSOLE_METRIC_EXPORT_INTERVAL_MS", 5000),
                    )
                )
                _safe_print("[telemetry] Console metric exporter attached")
            except Exception as ex:
                _safe_print(f"[telemetry] Console metric exporter attach failed: {ex}")

        meter_provider = SDKMeterProvider(resource=resource, metric_readers=readers)
        metrics.set_meter_provider(meter_provider)
        _safe_print("[telemetry] created new SDK meter provider")

    # -------- Instruments --------
    STATE.tracer = trace.get_tracer("ai.agent.tracer")
    STATE.meter = metrics.get_meter("ai.agent.metrics")

    # ---- V1 instruments ----
    STATE.req_counter = STATE.meter.create_counter("ai_requests_total", "Total AI requests", "1")
    STATE.err_counter = STATE.meter.create_counter("ai_requests_errors_total", "Total AI request errors", "1")
    STATE.latency_hist = STATE.meter.create_histogram("ai_request_latency_ms", "AI request latency in milliseconds", "ms")
    STATE.token_counter = STATE.meter.create_counter("ai_tokens_total", "AI token usage by type/model", "1")
    STATE.cost_counter = STATE.meter.create_counter("ai_cost_usd_total", "Estimated AI cost in USD", "1")

    # ---- V2 RAG ----
    STATE.rag_req_counter = STATE.meter.create_counter("rag_retrieval_requests_total", "RAG retrieval request count", "1")
    STATE.rag_latency_hist = STATE.meter.create_histogram("rag_retrieval_latency_ms", "RAG retrieval latency", "ms")
    STATE.rag_docs_counter = STATE.meter.create_counter("rag_retrieved_documents_total", "Retrieved documents count", "1")
    STATE.rag_rerank_latency_hist = STATE.meter.create_histogram("rag_rerank_latency_ms", "Rerank latency", "ms")
    STATE.rag_context_tokens_counter = STATE.meter.create_counter("rag_context_tokens_total", "Context tokens injected from retrieval", "1")
    STATE.rag_no_result_counter = STATE.meter.create_counter("rag_no_result_total", "Retrieval calls with zero results", "1")

    # ---- V2 tools ----
    STATE.tool_calls_counter = STATE.meter.create_counter("ai_tool_calls_total", "Total tool invocations", "1")
    STATE.tool_latency_hist = STATE.meter.create_histogram("ai_tool_latency_ms", "Tool latency", "ms")
    STATE.tool_errors_counter = STATE.meter.create_counter("ai_tool_errors_total", "Tool errors", "1")
    STATE.tool_timeout_counter = STATE.meter.create_counter("ai_tool_timeout_total", "Tool timeouts", "1")

    # ---- V2 model ----
    STATE.model_invocations_counter = STATE.meter.create_counter("ai_model_invocations_total", "LLM invocations by model", "1")
    STATE.model_latency_hist = STATE.meter.create_histogram("ai_model_latency_ms", "Model latency", "ms")
    STATE.model_fallback_counter = STATE.meter.create_counter("ai_model_fallback_total", "Model fallback count", "1")
    STATE.model_selection_counter = STATE.meter.create_counter("ai_model_selection_total", "Model selection decisions", "1")

    # ---- V2 eval ----
    STATE.eval_runs_counter = STATE.meter.create_counter("ai_eval_runs_total", "Eval runs", "1")
    STATE.eval_pass_counter = STATE.meter.create_counter("ai_eval_pass_total", "Eval pass count", "1")
    STATE.eval_fail_counter = STATE.meter.create_counter("ai_eval_fail_total", "Eval fail count", "1")
    STATE.eval_score_hist = STATE.meter.create_histogram("ai_eval_score", "Eval score distribution", "1")
    STATE.eval_latency_hist = STATE.meter.create_histogram("ai_eval_latency_ms", "Eval latency", "ms")

    # ---- V2 streaming ----
    STATE.stream_responses_counter = STATE.meter.create_counter("ai_stream_responses_total", "Streaming responses", "1")
    STATE.stream_ttft_hist = STATE.meter.create_histogram("ai_stream_first_token_latency_ms", "Time to first token", "ms")
    STATE.stream_tokens_counter = STATE.meter.create_counter("ai_stream_tokens_total", "Streamed tokens", "1")

    atexit.register(_force_flush)
    STATE.initialized = True
    _safe_print("[telemetry] observability v2 initialized successfully")


def _force_flush() -> None:
    try:
        tp = trace.get_tracer_provider()
        if hasattr(tp, "force_flush"):
            tp.force_flush()
    except Exception:
        pass
    try:
        mp = metrics.get_meter_provider()
        if hasattr(mp, "force_flush"):
            mp.force_flush()
    except Exception:
        pass


def start_timer() -> float:
    import time
    return time.perf_counter()


def end_timer_ms(start: float) -> float:
    import time
    return (time.perf_counter() - start) * 1000.0


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = MODEL_PRICING_PER_1K.get(model)
    if not price:
        return 0.0
    return (prompt_tokens / 1000.0) * price["prompt"] + (completion_tokens / 1000.0) * price["completion"]


def _base_attrs(
    *,
    agent_name: str,
    endpoint: str,
    status: str,
    model: Optional[str] = None,
    request_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> Dict[str, str]:
    """
    Metrics attributes MUST be low-cardinality.
    IDs are disabled by default for metrics to avoid time-series explosion.
    """
    attrs: Dict[str, str] = {
        "agent.name": agent_name,
        "http.route": endpoint,
        "status": status,
    }
    if model:
        attrs["gen_ai.request.model"] = model

    include_ids = _safe_bool_env("OBS_INCLUDE_IDS_IN_METRICS", False)
    if include_ids:
        if request_id:
            attrs["request.id"] = str(request_id)
        if correlation_id:
            attrs["correlation.id"] = str(correlation_id)

    return attrs


# ---------- V1 ----------
def record_request(
    *,
    agent_name: str,
    model: str,
    endpoint: str,
    status: str,
    latency_ms: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    error_type: Optional[str] = None,
    request_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> None:
    if not STATE.initialized:
        return

    attrs = _base_attrs(
        agent_name=agent_name,
        model=model,
        endpoint=endpoint,
        status=status,
        request_id=request_id,
        correlation_id=correlation_id,
    )

    STATE.req_counter.add(1, attrs)
    STATE.latency_hist.record(latency_ms, attrs)

    if prompt_tokens:
        STATE.token_counter.add(prompt_tokens, {**attrs, "token.type": "prompt"})
    if completion_tokens:
        STATE.token_counter.add(completion_tokens, {**attrs, "token.type": "completion"})
    if total_tokens:
        STATE.token_counter.add(total_tokens, {**attrs, "token.type": "total"})
    if cache_read_tokens:
        STATE.token_counter.add(cache_read_tokens, {**attrs, "token.type": "cache_read"})
    if cache_write_tokens:
        STATE.token_counter.add(cache_write_tokens, {**attrs, "token.type": "cache_write"})

    cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)
    if cost > 0:
        STATE.cost_counter.add(cost, attrs)

    if status == "error":
        err_attrs = dict(attrs)
        if error_type:
            err_attrs["error.type"] = error_type
        STATE.err_counter.add(1, err_attrs)


# ---------- V2 RAG ----------
def record_rag(
    *,
    agent_name: str,
    endpoint: str,
    status: str,
    index_name: str,
    latency_ms: float,
    retrieved_docs: int,
    top_k: Optional[int] = None,
    context_tokens: int = 0,
    rerank_latency_ms: Optional[float] = None,
    request_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> None:
    if not STATE.initialized:
        return

    attrs = _base_attrs(
        agent_name=agent_name,
        endpoint=endpoint,
        status=status,
        model=None,
        request_id=request_id,
        correlation_id=correlation_id,
    )
    attrs["rag.index.name"] = index_name
    if top_k is not None:
        # keep it low-cardinality: top_k is small bounded int; string is okay
        attrs["rag.top_k"] = str(top_k)

    STATE.rag_req_counter.add(1, attrs)
    STATE.rag_latency_hist.record(latency_ms, attrs)
    STATE.rag_docs_counter.add(max(0, retrieved_docs), attrs)

    if retrieved_docs == 0:
        STATE.rag_no_result_counter.add(1, attrs)
    if context_tokens > 0:
        STATE.rag_context_tokens_counter.add(context_tokens, attrs)
    if rerank_latency_ms is not None:
        STATE.rag_rerank_latency_hist.record(rerank_latency_ms, attrs)


# ---------- V2 Tools ----------
def record_tool_call(
    *,
    agent_name: str,
    endpoint: str,
    tool_name: str,
    tool_kind: str,
    status: str,
    latency_ms: float,
    error_type: Optional[str] = None,
    timeout: bool = False,
    request_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> None:
    if not STATE.initialized:
        return

    attrs = _base_attrs(
        agent_name=agent_name,
        endpoint=endpoint,
        status=status,
        model=None,
        request_id=request_id,
        correlation_id=correlation_id,
    )
    # tool.name can be high-cardinality if you embed dynamic IDs. Keep it stable.
    attrs["tool.name"] = tool_name
    attrs["tool.kind"] = tool_kind

    STATE.tool_calls_counter.add(1, attrs)
    STATE.tool_latency_hist.record(latency_ms, attrs)

    if status == "error":
        err_attrs = dict(attrs)
        if error_type:
            err_attrs["error.type"] = error_type
        STATE.tool_errors_counter.add(1, err_attrs)

    if timeout:
        STATE.tool_timeout_counter.add(1, attrs)


# ---------- V2 Model orchestration ----------
def record_model_selection(
    *,
    agent_name: str,
    endpoint: str,
    selected_model: str,
    selection_policy: str,
    request_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> None:
    if not STATE.initialized:
        return

    attrs = _base_attrs(
        agent_name=agent_name,
        endpoint=endpoint,
        status="ok",
        model=None,
        request_id=request_id,
        correlation_id=correlation_id,
    )
    attrs["selected.model"] = selected_model
    attrs["selection.policy"] = selection_policy
    STATE.model_selection_counter.add(1, attrs)


def record_model_invocation(
    *,
    agent_name: str,
    endpoint: str,
    provider: str,
    model: str,
    status: str,
    latency_ms: float,
    request_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> None:
    if not STATE.initialized:
        return

    attrs = _base_attrs(
        agent_name=agent_name,
        endpoint=endpoint,
        status=status,
        model=model,
        request_id=request_id,
        correlation_id=correlation_id,
    )
    attrs["gen_ai.provider"] = provider

    STATE.model_invocations_counter.add(1, attrs)
    STATE.model_latency_hist.record(latency_ms, attrs)


def record_model_fallback(
    *,
    agent_name: str,
    endpoint: str,
    from_model: str,
    to_model: str,
    reason: str,
    request_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> None:
    if not STATE.initialized:
        return

    attrs = _base_attrs(
        agent_name=agent_name,
        endpoint=endpoint,
        status="ok",
        model=None,
        request_id=request_id,
        correlation_id=correlation_id,
    )
    attrs["from.model"] = from_model
    attrs["to.model"] = to_model
    attrs["reason"] = reason
    STATE.model_fallback_counter.add(1, attrs)


# ---------- V2 Evals ----------
def record_eval(
    *,
    agent_name: str,
    eval_name: str,
    eval_metric: str,
    score: float,
    passed: bool,
    latency_ms: float,
    status: str = "ok",
    request_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> None:
    if not STATE.initialized:
        return

    # eval_name/eval_metric should be stable strings (bounded set)
    attrs: Dict[str, str] = {
        "agent.name": agent_name,
        "eval.name": eval_name,
        "eval.metric": eval_metric,
        "eval.result": "pass" if passed else "fail",
        "status": status,
    }

    include_ids = _safe_bool_env("OBS_INCLUDE_IDS_IN_METRICS", False)
    if include_ids:
        if request_id:
            attrs["request.id"] = str(request_id)
        if correlation_id:
            attrs["correlation.id"] = str(correlation_id)

    STATE.eval_runs_counter.add(1, attrs)
    STATE.eval_latency_hist.record(latency_ms, attrs)
    STATE.eval_score_hist.record(score, attrs)

    if passed:
        STATE.eval_pass_counter.add(1, attrs)
    else:
        STATE.eval_fail_counter.add(1, attrs)


# ---------- V2 Streaming ----------
def record_streaming(
    *,
    agent_name: str,
    endpoint: str,
    model: str,
    status: str,
    ttft_ms: Optional[float] = None,
    streamed_tokens: int = 0,
    request_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> None:
    if not STATE.initialized:
        return

    attrs = _base_attrs(
        agent_name=agent_name,
        endpoint=endpoint,
        status=status,
        model=model,
        request_id=request_id,
        correlation_id=correlation_id,
    )

    STATE.stream_responses_counter.add(1, attrs)
    if ttft_ms is not None:
        STATE.stream_ttft_hist.record(ttft_ms, attrs)
    if streamed_tokens > 0:
        STATE.stream_tokens_counter.add(streamed_tokens, attrs)


def extract_usage_from_aoai_response(resp) -> Dict[str, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)

    prompt_details = getattr(usage, "prompt_tokens_details", None)
    cache_read = int(getattr(prompt_details, "cached_tokens", 0) or 0) if prompt_details else 0

    # Keep conservative until provider exposes true cache write fields.
    cache_write = 0

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
    }
