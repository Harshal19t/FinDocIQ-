<canvas identifier="findociq-readme" type="text/markdown" title="FinDocIQ README.md" content="# FinDocIQ: Retrieval-Augmented Generation for Financial Reports
FinDocIQ Architecture
End-to-end RAG assistant for ingesting SEC 10-Q reports and delivering accurate, context-grounded financial answers.

📌 About the Project
FinDocIQ is a Retrieval-Augmented Generation (RAG) assistant designed to extract insights from complex SEC 10-Q financial reports. It addresses two major challenges in applying LLMs to financial documents:

Hallucinations & Missing Context: Out-of-the-box LLMs lack access to real-time financial filings.
Table Structural Breakdown: Raw text extractors distort structured financial data.
FinDocIQ combines advanced document parsing, vector search, and fine-tuned local LLMs to deliver accurate, verifiable answers with source attribution.

🏗️ Architecture Overview
mermaid
Copy



Mermaid Error
Error: Lexical error on line 8. Unrecognized text.
...EC 10-Q PDF Filing\"]:::doc --> B[\"pdfp
-----------------------^


✨ Key Features
✅ Advanced Document Parsing: Uses pdfplumber to extract narrative text and financial tables while preserving semantic context.
✅ High-Density Vector Search: Leverages Qdrant and BAAI/bge-small-en-v1.5 for fast similarity queries and dynamic Top-K chunk retrieval.
✅ Fine-Tuned Local LLM: Qwen2.5-1.5B + LoRA for accurate, grounded responses without excessive GPU usage.
✅ Interactive UI: Streamlit app with dynamic Top-K tuning, source citation inspection, and pipeline cache management.
✅ Containerized Deployment: Docker Compose for seamless setup with PyTorch, Qdrant, and Streamlit.

🛠️ Technical Stack

  
    
      Component
      Technology
    
  
  
    
      Document Parsing
      pdfplumber
    
    
      Embedding Model
      BAAI/bge-small-en-v1.5
    
    
      Vector Store
      Qdrant
    
    
      LLM
      Qwen2.5-1.5B + LoRA
    
    
      UI Framework
      Streamlit
    
    
      Containerization
      Docker Compose + BuildKit
    
    
      Language
      Python 3.10+
    
  





🚀 Getting Started
Prerequisites

Docker (Install Docker)
Docker Compose (included with Docker Desktop)
Python 3.10+ (for local development)
Installation


Clone the repository:
bash
Copy

git clone https://github.com/your-username/FinDocIQ.git
cd FinDocIQ





Set up environment variables:
Create a .env file in the root directory and add:
env
Copy

QDRANT_HOST=localhost
QDRANT_PORT=6333





Build and run with Docker:
bash
Copy

docker-compose up --build




This will:

Start the Qdrant vector store.
Launch the Streamlit UI (accessible at http://localhost:8501).
Load the fine-tuned Qwen2.5-1.5B model.



Access the app:
Open your browser and navigate to:
🔗 http://localhost:8501


📂 Project Structure
text
Copy

FinDocIQ/
├── docker-compose.yml          # Docker Compose configuration
├── Dockerfile                  # Dockerfile for the Streamlit app
├── app/                        # Streamlit application
│   ├── main.py                 # Main Streamlit app
│   ├── config.py               # Configuration settings
│   ├── pipelines/              # RAG and document processing pipelines
│   │   ├── ingestion.py        # Document ingestion logic
│   │   ├── retrieval.py        # Vector search and retrieval
│   │   └── generation.py       # LLM response generation
│   └── utils/                  # Utility functions
│       ├── qdrant_client.py    # Qdrant client wrapper
│       └── model_loader.py      # LLM loading and fine-tuning
├── data/                       # Sample SEC 10-Q filings
├── models/                     # Fine-tuned model weights (if applicable)
├── .env.example                # Example environment variables
└── README.md                   # Project documentation




🔧 Configuration
Environment Variables

  
    
      Variable
      Description
      Default Value
    
  
  
    
      QDRANT_HOST
      Host for Qdrant vector store
      localhost
    
    
      QDRANT_PORT
      Port for Qdrant vector store
      6333
    
    
      TOP_K
      Number of chunks to retrieve
      5
    
    
      MODEL_PATH
      Path to fine-tuned Qwen2.5-1.5B model
      ./models/
    
  




Customizing the Pipeline

Top-K Retrieval: Adjust the TOP_K variable in config.py to control the number of context chunks passed to the LLM.
Embedding Model: Replace BAAI/bge-small-en-v1.5 in pipelines/ingestion.py with another embedding model (e.g., sentence-transformers/all-mpnet-base-v2).
LLM: Swap Qwen2.5-1.5B with another model in utils/model_loader.py.

📈 Usage


Upload a SEC 10-Q PDF:

Use the file uploader in the Streamlit UI to upload a PDF.
The system will automatically parse and chunk the document.


Ask a Question:

Enter your query in the chat interface (e.g., "What was the company's revenue in Q2 2024?").
FinDocIQ will:

Retrieve the most relevant chunks from the vector store.
Generate a grounded response with source citations.



Inspect Sources:

Click on source citations to view the raw text chunks used for the response.


Adjust Top-K:

Use the slider to change the number of retrieved chunks (TOP_K) in real-time.


🔍 Key Technical Components
1. Document Processing & Ingestion

pdfplumber: Extracts text and tables from PDFs while preserving structure.
Semantic Chunking: Splits documents into meaningful chunks (e.g., paragraphs, table rows) for embedding.
2. Vector Search

BAAI/bge-small-en-v1.5: Generates 384-dimensional embeddings for each chunk.
Qdrant: Indexes embeddings for low-latency similarity search.
3. Retrieval-Augmented Generation (RAG)

Context Assembly: Combines the user query and top-K chunks into a prompt for the LLM.
Fine-Tuned Qwen2.5-1.5B: Generates grounded responses using LoRA for parameter-efficient fine-tuning.
4. Containerization

Docker Compose: Orchestrates the Streamlit app, Qdrant, and PyTorch services.
BuildKit Caching: Speeds up Docker builds by caching pip dependencies.

📊 Performance

  
    
      Metric
      Value
    
  
  
    
      Embedding Time
      ~0.5s per chunk
    
    
      Vector Search
      ~10ms per query (Top-K=5)
    
    
      LLM Inference
      ~2s per response
    
    
      Docker Build
      ~5min (first run)
    
  





🛠️ Development
Local Setup (Without Docker)


Install dependencies:
bash
Copy

pip install -r requirements.txt





Start Qdrant locally:
bash
Copy

docker run -p 6333:6333 qdrant/qdrant





Run the Streamlit app:
bash
Copy

streamlit run app/main.py




Adding New Features

Hybrid Search: Integrate BM25 for sparse keyword matching alongside dense embeddings.
Evaluation: Use RAGAS to automate metrics like faithfulness and answer relevancy.
API: Expose the pipeline via FastAPI for programmatic access.

🤝 Contributing
Contributions are welcome! Here’s how you can help:

Fork the repository and create a new branch.
Submit a Pull Request with your changes.
Report bugs or suggest features by opening an Issue.
Contribution Guidelines

Follow PEP 8 for Python code.
Add tests for new features.
Update the documentation (including this README.md).

📜 License
This project is licensed under the MIT License – see the LICENSE file for details.

🙏 Acknowledgments

Qdrant: https://qdrant.tech/
BAAI/bge-small-en-v1.5: https://huggingface.co/BAAI/bge-small-en-v1.5
Qwen2.5-1.5B: https://huggingface.co/Qwen/Qwen2.5-1.5B
Streamlit: https://streamlit.io/

📧 Contact
For questions or feedback, reach out to:

Harshal Trivedi (@harshal19t)
Daemian (@daemain19)

⭐ Star this repo if you find it useful!
</canvas>
