```mermaid

flowchart LR
  U[User or Calling Agent] --> G[API Gateway / APIM]
  G --> A[Agent App\nFastAPI / Azure Functions / AKS]

  A -->|uses| SDK[Internal Agent SDK\nhooks + plugins]
  SDK -->|creates| T[Traces\nspans + events]
  SDK -->|emits| M[Metrics\ncounts + latency + tokens]
  SDK -->|optional| L[Logs\nstructured]

  T -->|OTLP| C[OTel Collector]
  M -->|OTLP| C
  L -->|OTLP| C

  C -->|export| B1[Backend 1\n Dynatrace]
  C -->|export| B2[Backend 2\n Grafana]
  C -->|export| B3[Backend 3\n Azure Monitor]
