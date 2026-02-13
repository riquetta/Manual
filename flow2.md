
```mermaid
flowchart TD
    A[Incoming request] --> B[Create request span and start timer]
    B --> C[Execute AI pipeline]
    C --> D{Success}
    D -- Yes --> E[Extract usage and record_request status ok]
    D -- No --> F[record_request status error]
    E --> G[Export OTLP to Dynatrace]
    F --> G
