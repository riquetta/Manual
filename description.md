The key principle is:

If a stage produces observable signals (span attrs, usage fields, timing, status), your pipeline can capture and stream them.

If a stage runs “silently” (no instrumentation + no surfaced metadata), it won’t appear.



“If it streams, we capture” rules (simple)

Use this checklist per component (RAG/tool/model/eval/downstream):

Is there an active parent span?

If yes, child can correlate.

Is the component instrumented (auto or manual)?

If yes, spans appear.

Does the component expose usage/timing/status fields?

If yes, metrics/attrs can be emitted.

Does record_request (or equivalent) run for that path?

If yes, your v1 metrics populate.

Does exporter auth/config work?

If yes, data streams to Dynatrace.

If any of these is “no,” that segment is partially visible or invisible.

Recommended minimal additions for a “fully featured agent” rollout

To capture RAG/tools/multi-model/evals robustly, add these stage spans:

agent.route

rag.retrieve

rag.rerank (if used)

tool.<name>.call (e.g., tool.cosmos.query, tool.storage.read)

agent.llm.<model_or_role> (parent stage; auto LLM span stays child)

eval.run

eval.score

And add these attributes consistently:

agent.name

graph.name / workflow.name (if applicable)

gen_ai.request.model

http.route

status

error.type (on failure)

correlation.id, request.id
