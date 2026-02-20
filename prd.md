```mermaid
flowchart TB

  subgraph DEV[Developer and Team Workflow]
    Repo[Agent repos\nFastAPI Azure Functions AKS later]
    SDK[Internal pip package agent-sdk\nobservability hooks decorators plugins]
    Env[Env config\nOBS_PLUGINS OTEL_ sampling]
    Repo --> SDK
    SDK --> Env
  end

  subgraph RUNTIME[Runtime Agent Services]
    APIM[API Gateway APIM\nAuth rate limit headers]
    subgraph SVC[Agent Service Instances]
      AutoInst[Optional OTel auto-instrumentation\nHTTP server spans]
      App[Agent App\nUses agent-sdk\nspans events metrics token usage]
      Plugins[SDK Plugins\nguardrail cache rag tools llm evals]
      AutoInst --> App
      App --> Plugins
    end
    APIM --> App
  end

  subgraph COLLECT[OpenTelemetry Collector Layer]
    Ingest[OTLP receiver\ngrpc 4317 http 4318]
    Proc[Processors\nmemory_limiter batch\nredaction enrich sampling]
    Route[Routing fan-out\nby service env team]
    Ingest --> Proc --> Route
  end

  subgraph BACKENDS[Observability Backends]
    DT[Dynatrace\ntraces metrics logs]
    AM[Azure Monitor App Insights\noptional]
    GRAF[Grafana stack\nTempo Mimir Loki]
    SIEM[SIEM\nSplunk Elastic optional]
  end

  App --> Ingest

  Route --> DT
  Route --> GRAF
  Route --> AM
  Route --> SIEM

  subgraph CONTROLS[Production Controls]
    Redact[PII prompt safety\nno prompts in metrics\ntruncate span events\ncollector drops risky attrs]
    Card[Cardinality policy\nstable dims only\nIDs traces-only by default]
    Samp[Sampling strategy\ncollector tail sampling\nkeep errors and slow traces]
    Release[SDK release process\nversioned package\nnew plugins add telemetry]
  end

  Redact -.enforced via.-> Proc
  Card -.enforced via.-> SDK
  Samp -.implemented in.-> Proc
  Release -.drives.-> SDK
