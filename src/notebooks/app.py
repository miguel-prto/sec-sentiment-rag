import os

import requests
import streamlit as st
from databricks.ai_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

os.environ.pop("DATABRICKS_METADATA_SERVICE_URL", None)
os.environ.pop("DATABRICKS_AUTH_TYPE", None)

st.set_page_config(
    page_title="SEC 10-K Financial Sentiment & RAG Engine",
    page_icon="📈",
    layout="wide"
)

# Cache models
@st.cache_resource
def init_clients():
    vsc = VectorSearchClient()
    w = WorkspaceClient()
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return vsc, w, embedder

@st.cache_resource
def get_workspace_client():
    return WorkspaceClient()

w = get_workspace_client()

def get_merged_window_sql(accession: str, center_idx: int) -> str:
    min_idx = max(0, center_idx - 1)
    max_idx = center_idx + 1
    warehouse_id = os.getenv("DATABRICKS_SQL_WAREHOUSE_ID")
    if not warehouse_id:
        raise RuntimeError("Set DATABRICKS_SQL_WAREHOUSE_ID in .env before querying Databricks.")
    escaped_accession = accession.replace("'", "''")
    
    query = f"""
        SELECT chunk_text 
        FROM {GOLD_TABLE}
        WHERE accession_number = '{escaped_accession}'
          AND chunk_index BETWEEN {min_idx} AND {max_idx}
        ORDER BY chunk_index ASC
    """

    response = w.statement_execution.execute_statement(
        statement=query,
        warehouse_id=warehouse_id
    )
    
    rows = response.result.data_array if response.result else []
    clean_chunks = [r[0].strip() for r in rows if r[0]]
    
    return " ".join(clean_chunks)

def query_databricks_llm(prompt: str) -> tuple[str, list[str]]:
    host = os.getenv("DATABRICKS_HOST", "").rstrip("/")
    token = os.getenv("DATABRICKS_TOKEN")
    if not host or not token:
        raise RuntimeError("Set DATABRICKS_HOST and DATABRICKS_TOKEN in .env before querying the model endpoint.")
    
    url = f"{host}/serving-endpoints/{LLM_ENDPOINT}/invocations"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful financial analyst assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }
    
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    
    message_content = data["choices"][0]["message"]["content"]

    text = ""
    reasoning = []

    if isinstance(message_content, list):
        for block in message_content:
            if block.get("type") == "text":
                text += block.get("text", "")
                
            elif block.get("type") == "reasoning":
                if "summary" in block:
                    for sub_block in block["summary"]:
                        if sub_block.get("type") == "summary_text":
                            reasoning.append(sub_block.get("text", ""))
                elif "reasoning_text" in block:
                    reasoning.append(block.get("reasoning_text", ""))

    return text, reasoning

vsc, w, embedder = init_clients()

CATALOG = "financial_db"
SCHEMA = "filings"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.sec_gold_vector_index"
ENDPOINT_NAME = "financial_rag_endpoint"
LLM_ENDPOINT = "databricks-gpt-oss-120b"
GOLD_TABLE = f"{CATALOG}.{SCHEMA}.sec_rag_gold"

# Sidebar
st.sidebar.title("RAG Configuration")
ticker_filter = st.sidebar.text_input("Filter by Ticker (Optional)", value="AAPL")
top_k = st.sidebar.slider("Number of Chunks to Retrieve (Top K)", min_value=1, max_value=10, value=4)

st.sidebar.markdown("---")
st.sidebar.info("""
**Architecture Highlights:**
- **Medallion Architecture** (Delta Lake)
- **FinBERT** Sentiment Classifier
- **BGE-Small-v1.5** Vector Embeddings
- **Databricks Vector Search** Index
""")

# Interface
st.title("SEC 10-K RAG & Sentiment Analysis Engine")
st.caption("Ask natural language questions across SEC filings with FinBERT sentiment attribution.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg.get("retrieved_docs"):
            with st.expander("Show reasoning and retrieved evidence"):
                if msg.get("reasoning"):
                    st.caption(msg["reasoning"])
                
                for doc in msg["retrieved_docs"]:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**Source [{doc['source_num']}] — Ticker: {doc['ticker']} ({doc['date']})**")
                        st.caption(doc["excerpt"])
                    with col2:
                        st.metric(
                            label="FinBERT Label", 
                            value=doc["sentiment"].upper(),
                            delta=f"-{doc['negative_probability']*100:.1f}% Neg Risk" if doc["sentiment"] == "negative" else "Normal"
                        )
                    st.divider()

# User input
if user_query := st.chat_input("Ask a question about 10-K Risk Factors (e.g., foreign exchange risks, supply chain delays)..."):
    
    # 1. Show user message in chat
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Vector Search Retrieval
    with st.chat_message("assistant"):
        with st.status("Searching SEC Vector Search Index...", expanded=False) as status:
            
            # Embed user question
            query_vector = embedder.encode(user_query, normalize_embeddings=True).tolist()
            
            # Apply Ticker Filter if specified
            filters_dict = {}
            if ticker_filter.strip():
                filters_dict["ticker"] = ticker_filter.strip().upper()

            # Execute Vector Search
            index = vsc.get_index(endpoint_name=ENDPOINT_NAME, index_name=INDEX_NAME)
            search_results = index.similarity_search(
                query_vector=query_vector,
                columns=[
                    "ticker", 
                    "date", 
                    "accession_number", 
                    "chunk_index", 
                    "chunk_text", 
                    "sentiment_label", 
                    "sentiment_neg",
                    "sentiment_pos"
                ],
                filters=filters_dict,
                num_results=top_k
            )
            
            docs = search_results.get("result", {}).get("data_array", [])
            status.update(label=f" Found {len(docs)} relevant 10-K risk factor chunks!", state="complete")

        if not docs:
            st.warning("No relevant filings found matching your query or filter.")
        else:
            # 3. Format Prompt for LLM
            context_str = ""
            structured_docs = []

            for i, doc in enumerate(docs, 1):
                ticker, date, accession, chunk_idx, raw_text, sent_label, sent_neg, sent_pos = doc[:8]
                
                # Fetch adjacent window text from our pre-fetched Spark result
                merged_window_text = get_merged_window_sql(accession, chunk_idx)
                final_text = merged_window_text if merged_window_text else raw_text
                
                doc_info = {
                    "source_num": i,
                    "ticker": ticker,
                    "date": date,
                    "accession_number": accession,
                    "center_chunk_index": chunk_idx,
                    "sentiment": sent_label,
                    "negative_probability": round(sent_neg, 4),
                    "excerpt": final_text
                }
                structured_docs.append(doc_info)
                
                context_str += (
                    f"\n[SOURCE {i}] Ticker: {ticker} | Date: {date} | "
                    f"Filing ID: {accession} (Window centered around Chunk {chunk_idx}) | "
                    f"FinBERT Sentiment: {sent_label} (Neg Prob: {sent_neg:.2f})\n"
                    f"Excerpt:\n{final_text}\n"
                )

            prompt = f"""You are a senior financial analyst. Answer the user's question using ONLY the provided SEC 10-K excerpts. 
            Cite source numbers [SOURCE X] for every fact or claim you make.

            USER QUESTION:
            {user_query}

            SEC 10-K CONTEXT:
            {context_str}
            """

            # 4. Stream LLM Response using Databricks Foundation Model Serving
            try:                   
                llm_answer, reasoning = query_databricks_llm(prompt)
                reasoning_text = "\n".join(reasoning) if reasoning else "No reasoning available."
                with st.chat_message("assistant"):
                    st.markdown(llm_answer)
                    # 5. Render Expandable Sources & FinBERT Metrics
                    with st.expander("Show reasoning and retrieved evidence."):
                        st.caption(reasoning_text)
                        for doc in structured_docs:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**Source [{doc['source_num']}] — Ticker: {doc['ticker']} ({doc['date']})**")
                                st.caption(doc["excerpt"])
                            with col2:
                                st.metric(
                                    label="FinBERT Label", 
                                    value=doc["sentiment"].upper(),
                                    delta=f"-{doc['negative_probability']*100:.1f}% Neg Risk" if doc["sentiment"] == "negative" else "Normal"
                                )
                            st.divider()
                    # Save response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": llm_answer,
                    "reasoning": reasoning_text,
                    "retrieved_docs": structured_docs
                })
            except (requests.RequestException, RuntimeError, KeyError, IndexError, TypeError) as e:
                st.error(f"Failed to query LLM Endpoint '{LLM_ENDPOINT}': {e!s}")