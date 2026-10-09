# SEC 10-K Sentiment RAG

A Databricks-backed retrieval-augmented generation (RAG) demo for SEC 10-K filings. The notebooks ingest filings, extract and chunk risk-factor text, calculate FinBERT sentiment, and create embeddings for Databricks Vector Search. The Streamlit app retrieves relevant chunks and asks a Databricks model-serving endpoint to answer questions with filing citations.

This project is an educational/demo application, not investment advice. Model output can be incorrect; verify answers against the cited filings.

## Project contents

- `src/notebooks/01_ingest_sec.ipynb` downloads SEC filing data into a Bronze Delta table.
- `src/notebooks/02_parse_clean.ipynb` extracts and chunks filing text into a Silver table.
- `src/notebooks/03_sentiment_rag.ipynb` scores sentiment, creates embeddings, and prepares the Gold table and Vector Search index.
- `src/notebooks/app.py` is the Streamlit interface.
- `pyproject.toml` declares project and development dependencies; `uv.lock` pins the resolved environment.
- `.env.example` documents the local configuration variables. It contains placeholders only.

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- A Databricks workspace with permission to use Databricks Connect, a SQL warehouse, Vector Search, and a model-serving endpoint
- A valid SEC User-Agent that includes a contact email address

The notebooks create/use the `financial_db.filings` catalog and schema. The app's expected table, Vector Search endpoint/index, and model endpoint are configured near the top of `src/notebooks/app.py`; adjust those constants if your workspace uses different names.

## Local setup

```powershell
git clone https://github.com/miguel-prto/sec-sentiment-rag.git
cd sec-sentiment-rag
uv sync
Copy-Item .env.example .env
```

Edit `.env` and provide your own values:

- `SEC_USER_AGENT`: application name and a contact email, as required for SEC requests.
- `DATABRICKS_HOST`: your workspace URL.
- `DATABRICKS_TOKEN`: a Databricks token authorized for the required resources.
- `DATABRICKS_SQL_WAREHOUSE_ID`: the SQL warehouse ID used for context retrieval.

Never commit `.env`, tokens, or other credentials. `.env` and Streamlit secrets files are ignored by Git; use `.env.example` for safe placeholders.

To run the notebooks from a local Jupyter/VS Code kernel, install their Databricks Connect dependencies too:

```powershell
uv sync --group databricks
```

## Run the notebooks

Run the notebooks in order, using the Databricks Connect environment:

1. `01_ingest_sec.ipynb`
2. `02_parse_clean.ipynb`
3. `03_sentiment_rag.ipynb`

Each notebook uses Databricks resources and writes Delta tables. The final notebook creates or updates the Vector Search index; ensure the app's configured endpoint and index names match your workspace.

## Run the app

After the tables and Vector Search index are ready, start Streamlit:

```powershell
uv run streamlit run src/notebooks/app.py
```

The app uses `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, and `DATABRICKS_SQL_WAREHOUSE_ID` from `.env`. It also calls the Vector Search and model-serving resources named in `app.py`; the signed-in identity must be authorized to access them.

## Dependencies

Runtime dependencies are declared in `pyproject.toml`; `uv.lock` records exact resolutions. Development tools are in the `dev` group. Databricks Connect, required when running the notebooks locally, is in the optional `databricks` group and can be installed with `uv sync --group databricks`.

Do not edit `uv.lock` manually. After changing dependencies in `pyproject.toml`, run `uv lock`.

## License

No license file is currently included. Until a license is chosen and added, do not assume the project is licensed for reuse or redistribution.
