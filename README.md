graph TD
    %% Styling
    classDef doc fill:#1e293b,stroke:#475569,color:#f8fafc;
    classDef process fill:#0f172a,stroke:#3b82f6,color:#f8fafc;
    classDef model fill:#1e1b4b,stroke:#6366f1,color:#f8fafc;
    classDef db fill:#064e3b,stroke:#10b981,color:#f8fafc;
    classDef ui fill:#4c1d95,stroke:#8b5cf6,color:#f8fafc;

    %% Document Processing & Ingestion Pipeline
    subgraph "1. Document Processing & Ingestion Pipeline"
        A["SEC 10-Q PDF Filing"]:::doc --> B["pdfplumber Parser"]:::process
        B -->|Narrative Text & Financial Tables| C["Semantic Text Chunker"]:::process
        C --> D["BAAI/bge-small-en-v1.5"]:::model
        D -->|384-dim Vector Embeddings| E["Qdrant Local Vector Store"]:::db
    end

    %% Retrieval-Augmented Generation Pipeline
    subgraph "2. Retrieval-Augmented Generation Pipeline"
        F["User Query"]:::ui --> G["Streamlit App UI"]:::ui
        G -->|Query Vector Search| E
        E -->|Top-K Relevant Chunks| H["Context Assembly Engine"]:::process
        G -->|Original Prompt| H
        H --> I["Fine-Tuned Qwen2.5-1.5B (LoRA)"]:::model
        I -->|Grounded Response + Source Citations| G
    end

    %% Containerized Infrastructure
    subgraph "3. Containerized Infrastructure"
        J["Docker Compose"]:::process
        J --> G
        J --> E
    end