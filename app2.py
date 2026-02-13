import json
import os
import random
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load local variables
load_dotenv()

# Observability
from opentelemetry import trace
from infra.tracing_decorators_v2 import trace_stage
from infra.observability_v2 import (
    init_observability,
    start_timer,
    end_timer_ms,
    extract_usage_from_aoai_response,
    record_request,
    record_rag,
    record_tool_call,
    record_model_selection,
    record_model_invocation,
    record_eval,
)

tracer = trace.get_tracer("agent.tracer")

REGISTRY_PATH = os.getenv("AGENT_REGISTRY_PATH", "agent_registry_metadata.json")
ADMIN_API_KEY = os.getenv("REGISTRY_ADMIN_KEY", "dev-admin-key")  # POC only
AOAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]


# ---- Simple file-backed registry (POC) ----
def _load_registry() -> Dict[str, Any]:
    if not os.path.exists(REGISTRY_PATH):
        return {"agents": {}}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_registry(data: Dict[str, Any]) -> None:
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---- Azure OpenAI client ----
def _get_aoai_client() -> AzureOpenAI:
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    return AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)


# ---- Simulated RAG + Tool helpers (POC) ----
def _simulate_rag_retrieve(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Simulates retrieval results. Replace with your real vector DB retriever.
    """
    corpus = [
        {"id": "doc-1", "title": "Agent Security Baseline", "score": 0.91, "chunk": "Use least privilege and scoped tokens."},
        {"id": "doc-2", "title": "Observability Playbook", "score": 0.87, "chunk": "Track latency, errors, tokens, and cost."},
        {"id": "doc-3", "title": "RAG Design Notes", "score": 0.84, "chunk": "Keep retrieved context concise and relevant."},
        {"id": "doc-4", "title": "Tooling SOP", "score": 0.78, "chunk": "Set timeout and retries for external tools."},
    ]
    # tiny deterministic-ish simulation for POC
    if not query.strip():
        return []
    return corpus[:top_k]


def _simulate_cosmos_query(agent_appid: str) -> Dict[str, Any]:
    """
    Simulates a downstream tool call (e.g., Cosmos DB lookup).
    Replace with real Cosmos SDK call.
    """
    return {
        "agent_appid": agent_appid,
        "policy": "standard",
        "region": "canada-central",
    }


def _estimate_context_tokens(docs: List[Dict[str, Any]]) -> int:
    # POC rough approximation
    total_chars = sum(len(d.get("chunk", "")) for d in docs)
    return max(0, total_chars // 4)


# ---- App ----
app = FastAPI(title="BMO-Observability-POC1")
init_observability()


class RegisterAgent(BaseModel):
    appid: str
    name: str
    owner: Optional[str] = None
    description: Optional[str] = None
    allowed_models: Optional[list[str]] = None
    tags: Optional[dict[str, str]] = None


class ChatRequest(BaseModel):
    message: str


@app.post("/registry/register")
@trace_stage("agent.registry.register")
def register_agent(payload: RegisterAgent, x_admin_key: str = Header(default="")):
    if x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    reg = _load_registry()
    reg["agents"][payload.appid] = payload.model_dump()
    _save_registry(reg)
    return {"status": "ok", "agent": reg["agents"][payload.appid]}


@app.get("/registry/discover")
@trace_stage("agent.registry.discover")
def discover_agents():
    reg = _load_registry()
    return {"agents": list(reg["agents"].values())}


@app.post("/chat")
@trace_stage("agent.chat")
def chat(
    req: ChatRequest,
    x_agent_appid: Optional[str] = Header(default=None),
    x_agent_roles: Optional[str] = Header(default=None),
    x_correlation_id: Optional[str] = Header(default=None),
):
    t_req = start_timer()
    request_id = x_agent_appid or "unknown-request"
    correlation_id = x_correlation_id or "no-correlation-id"

    if not x_agent_appid:
        raise HTTPException(status_code=401, detail="Missing agent identity header (did APIM validate JWT?)")

    reg = _load_registry()
    agent_meta = reg["agents"].get(x_agent_appid)
    if not agent_meta:
        raise HTTPException(status_code=403, detail="Agent not registered")

    allowed = agent_meta.get("allowed_models") or [AOAI_DEPLOYMENT]
    if AOAI_DEPLOYMENT not in allowed:
        raise HTTPException(status_code=403, detail="Model not allowed for this agent")

    agent_name = agent_meta.get("name", "unknown-agent")

    try:
        # -------------------------
        # Stage 1: model selection
        # -------------------------
        with tracer.start_as_current_span("agent.model.select") as s_model_select:
            s_model_select.set_attribute("agent.name", agent_name)
            s_model_select.set_attribute("selected.model", AOAI_DEPLOYMENT)
            s_model_select.set_attribute("selection.policy", "static-allowlist")
            record_model_selection(
                agent_name=agent_name,
                endpoint="/chat",
                selected_model=AOAI_DEPLOYMENT,
                selection_policy="static-allowlist",
                request_id=request_id,
                correlation_id=correlation_id,
            )

        # -------------------------
        # Stage 2: RAG retrieval
        # -------------------------
        t_rag = start_timer()
        with tracer.start_as_current_span("rag.retrieve") as s_rag:
            top_k = 3
            s_rag.set_attribute("agent.name", agent_name)
            s_rag.set_attribute("rag.index.name", "poc-index")
            s_rag.set_attribute("rag.top_k", top_k)

            docs = _simulate_rag_retrieve(req.message, top_k=top_k)
            retrieved_docs = len(docs)
            context_tokens = _estimate_context_tokens(docs)

            s_rag.set_attribute("rag.retrieved_count", retrieved_docs)
            s_rag.set_attribute("rag.context_tokens", context_tokens)

            rag_latency = end_timer_ms(t_rag)
            record_rag(
                agent_name=agent_name,
                endpoint="/chat",
                status="ok",
                index_name="poc-index",
                latency_ms=rag_latency,
                retrieved_docs=retrieved_docs,
                top_k=top_k,
                context_tokens=context_tokens,
                request_id=request_id,
                correlation_id=correlation_id,
            )

        # -------------------------
        # Stage 3: tool call (simulated Cosmos)
        # -------------------------
        t_tool = start_timer()
        with tracer.start_as_current_span("tool.cosmos.query") as s_tool:
            s_tool.set_attribute("agent.name", agent_name)
            s_tool.set_attribute("tool.name", "cosmos-policy-lookup")
            s_tool.set_attribute("tool.kind", "database")

            tool_data = _simulate_cosmos_query(x_agent_appid)

            tool_latency = end_timer_ms(t_tool)
            record_tool_call(
                agent_name=agent_name,
                endpoint="/chat",
                tool_name="cosmos-policy-lookup",
                tool_kind="database",
                status="ok",
                latency_ms=tool_latency,
                request_id=request_id,
                correlation_id=correlation_id,
            )

        # -------------------------
        # Stage 4: LLM call
        # -------------------------
        client = _get_aoai_client()
        t_model = start_timer()
        with tracer.start_as_current_span("agent.aoai.chat") as s_llm:
            s_llm.set_attribute("agent.name", agent_name)
            s_llm.set_attribute("gen_ai.provider", "azure_openai")
            s_llm.set_attribute("gen_ai.request.model", AOAI_DEPLOYMENT)

            rag_context = "\n".join([f"- {d['title']}: {d['chunk']}" for d in docs]) if docs else "No context retrieved."
            policy_text = f"Policy profile: {tool_data.get('policy')} in {tool_data.get('region')}."

            resp = client.chat.completions.create(
                model=AOAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": f"You are agent '{agent_name}'. Keep answers concise and accurate."},
                    {"role": "system", "content": policy_text},
                    {"role": "system", "content": f"Retrieved context:\n{rag_context}"},
                    {"role": "user", "content": req.message},
                ],
            )

            usage = extract_usage_from_aoai_response(resp)

            # GenAI attrs (important for dashboards)
            s_llm.set_attribute("gen_ai.usage.input_tokens", usage["prompt_tokens"])
            s_llm.set_attribute("gen_ai.usage.output_tokens", usage["completion_tokens"])
            s_llm.set_attribute("gen_ai.usage.total_tokens", usage["total_tokens"])
            s_llm.set_attribute("gen_ai.usage.cache_read_tokens", usage["cache_read_tokens"])
            s_llm.set_attribute("gen_ai.usage.cache_write_tokens", usage["cache_write_tokens"])
            s_llm.set_attribute("gen_ai.prompt.caching", usage["cache_read_tokens"] > 0)

        model_latency = end_timer_ms(t_model)
        record_model_invocation(
            agent_name=agent_name,
            endpoint="/chat",
            provider="azure_openai",
            model=AOAI_DEPLOYMENT,
            status="ok",
            latency_ms=model_latency,
            request_id=request_id,
            correlation_id=correlation_id,
        )

        answer = resp.choices[0].message.content or ""

        # -------------------------
        # Stage 5: Eval (simulated)
        # -------------------------
        t_eval = start_timer()
        with tracer.start_as_current_span("eval.score") as s_eval:
            s_eval.set_attribute("agent.name", agent_name)
            s_eval.set_attribute("eval.name", "answer_quality_poc")
            s_eval.set_attribute("eval.metric", "groundedness")
            s_eval.set_attribute("eval.threshold", 0.70)

            # Simple simulation (replace with real evaluator)
            score = round(random.uniform(0.65, 0.95), 3)
            passed = score >= 0.70

            s_eval.set_attribute("eval.score", score)
            s_eval.set_attribute("eval.result", "pass" if passed else "fail")

        eval_latency = end_timer_ms(t_eval)
        record_eval(
            agent_name=agent_name,
            eval_name="answer_quality_poc",
            eval_metric="groundedness",
            score=score,
            passed=passed,
            latency_ms=eval_latency,
            status="ok",
            request_id=request_id,
            correlation_id=correlation_id,
        )

        # -------------------------
        # V1 summary request metric
        # -------------------------
        total_latency = end_timer_ms(t_req)
        record_request(
            agent_name=agent_name,
            model=AOAI_DEPLOYMENT,
            endpoint="/chat",
            status="ok",
            latency_ms=total_latency,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            cache_read_tokens=usage["cache_read_tokens"],
            cache_write_tokens=usage["cache_write_tokens"],
            request_id=request_id,
            correlation_id=correlation_id,
        )

        return {
            "agent_appid": x_agent_appid,
            "agent_roles": (x_agent_roles or ""),
            "correlation_id": correlation_id,
            "answer": answer,
            "eval": {
                "name": "answer_quality_poc",
                "metric": "groundedness",
                "score": score,
                "passed": passed,
            },
            "observability": {
                "rag_docs": len(docs),
                "cache_read_tokens": usage["cache_read_tokens"],
            },
        }

    except HTTPException:
        # Preserve explicit HTTP errors
        total_latency = end_timer_ms(t_req)
        record_request(
            agent_name=agent_name if "agent_name" in locals() else "unknown-agent",
            model=AOAI_DEPLOYMENT,
            endpoint="/chat",
            status="error",
            latency_ms=total_latency,
            error_type="HTTPException",
            request_id=request_id,
            correlation_id=correlation_id,
        )
        raise
    except Exception as ex:
        total_latency = end_timer_ms(t_req)
        record_request(
            agent_name=agent_name if "agent_name" in locals() else "unknown-agent",
            model=AOAI_DEPLOYMENT,
            endpoint="/chat",
            status="error",
            latency_ms=total_latency,
            error_type=type(ex).__name__,
            request_id=request_id,
            correlation_id=correlation_id,
        )
        raise
