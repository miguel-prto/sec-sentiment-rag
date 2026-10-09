SEC 10-K Financial Intelligence & Sentiment RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline and financial intelligence dashboard designed to process, analyze, and query SEC 10-K filings.

The system leverages a Medallion Data Lakehouse Architecture on Databricks, combining specialized FinBERT sentiment analysis, Databricks Vector Search, continuous Window Context Expansion, and Llama-3-70B via Databricks Foundation Model Serving. It features an interactive Streamlit frontend optimized for lightweight, low-cost serverless execution.

🌟 Key Features

Medallion Data Architecture (Bronze $\rightarrow$ Silver $\rightarrow$ Gold):

Bronze: Ingestion of raw SEC 10-K text filings and metadata.

Silver: Cleaned, standardized text broken down into structured chunks (~1,000 characters with overlapping boundaries) with assigned positional index tracking.

Gold: Enrichment with FinBERT financial sentiment scores (Positive, Negative, Neutral) and BGE embeddings, optimized for Databricks Vector Search.

Smart Context Expansion (Window Retrieval):

Solves sentence fragmentation from arbitrary character splits by retrieving adjacent chunks ($\pm 1$ position).

Concatenates neighboring chunks using a single space (" ") rather than double line breaks ("\n\n"), ensuring uninterrupted grammatical continuity for the LLM.

FinBERT Sentiment Analysis Integration:

Calculates financial sentiment scores directly in the data pipeline to provide contextual risk metrics alongside vector search results.

Serverless Lightweight Frontend Architecture:

Replaces heavy PySpark Connect sessions with the Databricks SDK (WorkspaceClient) and SQL Statement Execution API over a Serverless SQL Warehouse.

Prevents DENY_NEW_AND_EXISTING_RESOURCES runtime errors and budget caps while delivering sub-second context retrieval.

Foundation Model Integration:

Utilizes databricks-meta-llama-3-70b-instruct through Databricks Serving Endpoints for accurate financial insights grounded in SEC filings.

📐 System Architecture

                       ┌───────────────────────────────────────────────┐
                       │          SEC 10-K Raw Filings (JSON)          │
                       └───────────────────────┬───────────────────────┘
                                               │
                                               ▼
                      ┌─────────────────────────────────────────────────┐
                      │     BRONZE: Ingestion & Raw Text Extraction     │
                      └────────────────────────┬────────────────────────┘
                                               │
                                               ▼
                      ┌─────────────────────────────────────────────────┐
                      │  SILVER: Text Cleaning & Positional Chunking    │
                      │  (~1,000 chars/chunk + overlap + chunk_index)   │
                      └────────────────────────┬────────────────────────┘
                                               │
                                               ▼
                      ┌─────────────────────────────────────────────────┐
                      │    GOLD: FinBERT Sentiment UDF Enrichment       │
                      │      (Pos, Neg, Label) + Delta Lake Table       │
                      └────────────────────────┬────────────────────────┘
                                               │
                                               ▼
                      ┌─────────────────────────────────────────────────┐
                      │            Databricks Vector Search             │
                      │          (BGE Embeddings Indexing)              │
                      └────────────────────────┬────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ Streamlit Application (Lightweight Architecture)                                        │
│                                                                                         │
│  1. Vector Similarity Search ──► Retrieves Top Match (Accession + Chunk Index)          │
│  2. Window Retrieval Expansion  ──► SQL Statement Execution API over Serverless Warehouse│
│                                     Fetches adjacent chunks (idx - 1, idx, idx + 1)     │
│  3. Single-Space Merging        ──► Reconstructs unbroken narrative context              │
│  4. Llama 3 Serving Endpoint    ──► Generates financial answer with sentiment metrics   │
└─────────────────────────────────────────────────────────────────────────────────────────┘


🛠️ Project Structure

sec_sentiment_rag/
├── src/
│   ├── notebooks/
│   │   ├── 01_bronze_to_silver.py     # Data ingestion and chunking logic
│   │   ├── 02_silver_to_gold.py       # FinBERT UDF application & Gold table setup
│   │   ├── 03_vector_search_index.py  # Databricks Vector Search index creation
│   │   └── app.py                     # Streamlit web interface
├── .env.example                       # Environment variables template
├── requirements.txt                   # Project dependencies
└── README.md                          # Documentation


⚙️ Environment & Configuration

Create a .env file in the project root containing your workspace configuration:

# Databricks Credentials
DATABRICKS_HOST="https://<your-workspace-url>.cloud.databricks.com"
DATABRICKS_TOKEN="dapiXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

# Compute & Serving Resources
DATABRICKS_SQL_WAREHOUSE_ID="a1b2c3d4e5f67890"  # 16-character Serverless SQL Warehouse ID
LLM_ENDPOINT="databricks-meta-llama-3-70b-instruct"
VECTOR_SEARCH_ENDPOINT_NAME="sec_rag_vs_endpoint"
VECTOR_INDEX_NAME="financial_db.default.sec_rag_gold_index"


🚀 Installation & Setup

Clone the repository and install dependencies:

git clone https://github.com/your-repo/sec-sentiment-rag.git
cd sec-sentiment-rag
pip install -r requirements.txt


Run Pipeline Notebooks on Databricks:

Execute 01_bronze_to_silver.py to ingest and chunk 10-K filings.

Execute 02_silver_to_gold.py to attach FinBERT sentiment metadata.

Execute 03_vector_search_index.py to sync the Delta Table with the Vector Search Index.

Obtain Serverless SQL Warehouse ID:

Go to SQL Warehouses $\rightarrow$ Select your Serverless Warehouse $\rightarrow$ Connection Details.

Copy the 16-character ID from the HTTP Path (/sql/1.0/warehouses/<WAREHOUSE_ID>) and set DATABRICKS_SQL_WAREHOUSE_ID in .env.

Launch the Streamlit App:

streamlit run src/notebooks/app.py


💡 Key Lessons Learned & Technical Solutions

1. Eliminating PySpark Runtime Budget Lockouts

Issue: Instantiating DatabricksSession.builder.getOrCreate() inside Streamlit triggered pyspark.errors.exceptions.connect.UnknownException: (DENY_NEW_AND_EXISTING_RESOURCES) BAD_REQUEST when hitting workspace resource quotas or auto-scaling limits.

Solution: Removed PySpark dependencies from the Streamlit frontend. Used WorkspaceClient() from databricks-sdk with w.statement_execution.execute_statement(...) against a Serverless SQL Warehouse to execute lightweight batch SQL queries in milliseconds.

2. Grammatical Continuity in Character-Based Chunking

Issue: Strict character-count chunking cuts off mid-sentence. Concatenating chunks with double line breaks (\n\n) creates artificial paragraph breaks in the middle of sentences.

Solution: Cleaned boundary spaces with .strip() and joined window sequences with a single space (" "), presenting a smooth, uninterrupted block of text to Llama-3.

3. Vector Search Metadata Unpacking

Issue: databricks-vectorsearch appends similarity scores to returned rows, turning an 8-column query into a 9-element array and causing ValueError: too many values to unpack.

Solution: Explicitly sliced the returned array using doc[:8] to decouple record reading from trailing score fields.

📜 License

This project is licensed under the MIT License.