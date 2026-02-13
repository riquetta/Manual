End-to-end capture matrix for your current observability code
Capability / Signal	Captured today?	Condition to capture	How it is captured	Typical gaps / notes
HTTP request trace	Yes	Endpoint is wrapped by @trace_stage(...) (or framework creates server span)	Manual stage span	Best if every public route has a request span
Full trace correlation (parent/child)	Yes	Child operations execute inside active request/span context	OTel context propagation	Can break across detached async/background tasks
Request count (ai_requests_total)	Yes	record_request(...) is called	Custom metric counter	If you forget record_request in path, no metric
Error count (ai_requests_errors_total)	Yes	record_request(status="error", ...) called on failures	Custom metric counter	Must be called in exception path
Latency (ai_request_latency_ms)	Yes	Timer started/ended and record_request(latency_ms=...) called	Custom histogram	You choose scope: whole route vs sub-stage
Token totals (ai_tokens_total)	Yes	Usage fields are available and passed to record_request	Custom metric counter + token.type	No usage payload = token metrics zero/missing
Cost estimate (ai_cost_usd_total)	Conditional	Model exists in MODEL_PRICING_PER_1K	Derived from tokens in record_request	Without pricing map, stays 0
OpenAI LLM call spans	Conditional-Yes	OpenLLMetry enabled + compatible + call path instrumented	Auto-instrumentation	Disabled/mismatch => only manual spans remain
Multi-model visibility	Yes	You set gen_ai.request.model per call and record per stage	Span attrs + metric dimensions	Must pass actual model/deployment used
RAG retrieve stage visibility	Conditional-Yes	You add manual span for retrieval stage	@trace_stage / start_as_current_span	Retrieval libraries may not auto-instrument by default
Vector DB / retriever latency	Conditional	Instrument that client call manually or via library instrumentation	Manual span + attrs	Otherwise appears as a black box
Tool call spans (custom tools)	Conditional-Yes	Tool wrapper is instrumented (manual span)	Manual spans	OpenLLMetry may auto-capture some tool frameworks, not all
Downstream Storage (Blob/Table/etc.)	Conditional	Client/library instrumented OR manual span around operation	Manual or auto instrumentation	Auth/SDK failures still capturable if wrapped
Downstream Cosmos DB	Conditional-Yes	Cosmos call wrapped in manual span (or library instrumentation present)	Manual span + error attrs	Strongly recommend explicit tool.cosmos.* spans
Evals execution span	Conditional-Yes	Eval runner is instrumented manually	Manual stage spans	Add eval metadata attrs (dataset, metric, threshold)
Evals scores/metrics	Conditional	You emit custom metrics for eval outputs	Add new counters/histograms	Not in current v1 metrics yet (needs extension)
Prompt cache signal	Conditional	Provider returns cache fields (cached_tokens) and you map them	ai_tokens_total token.type=cache_read + span attrs	If provider/workload has no cache hits, remains zero
Streaming response telemetry	Conditional	You instrument stream lifecycle/events/chunks	Manual spans/events	Current code is request-level; chunk-level needs extra code
Retries visibility	Conditional	Retry loop emits attempt spans/attrs	Manual spans + attrs (retry.count)	Otherwise retries hidden in one long span
Guardrails/policy checks	Conditional	Wrap guardrail stage manually	Manual spans	Recommended for production explainability
Queue/background worker correlation	Conditional	Trace context propagated into worker	Context injection/extraction + manual spans	Common place where correlation breaks
Dynatrace ingestion (traces/metrics)	Yes	Correct OTLP endpoint + header token + exporter attached	OTLP HTTP exporter	Header format/auth errors are the main failure mode
