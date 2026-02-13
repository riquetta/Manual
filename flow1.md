```mermaid
flowchart LR
    A[Client sends request] --> B[API route starts request span]
    B --> C[Start timer]
    C --> D[Run app stages routing RAG tools DB]
    D --> E[Call OpenAI model]
    E --> F[Collect usage tokens and model]
    F --> G[Call record_request]
    G --> H[Emit custom metrics requests errors latency tokens cost]
    H --> I[Export traces and metrics via OTLP]
    I --> J[Dynatrace dashboard end to end visibility]
