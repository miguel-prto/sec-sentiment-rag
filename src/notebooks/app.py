import os
import streamlit as st
from dotenv import load_dotenv
from databricks.ai_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
from sentence_transformers import SentenceTransformer

load_dotenv()

os.environ.pop("DATABRICKS_METADATA_SERVICE_URL", None)
os.environ.pop("DATABRICKS_AUTH_TYPE", None)

st.set_page_config(
    page_title="SEC 10-K Financial Sentiment & RAG Engine",
    page_icon="📈",
    layout="wide"
)

# 3. Cache Model & Databricks Clients to avoid re-initializing on every rerun
@st.cache_resource
def init_clients():
    vsc = VectorSearchClient()
    w = WorkspaceClient()
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return vsc, w, embedder

vsc, w, embedder = init_clients()

# Configuration Constants
CATALOG = "financial_db"
SCHEMA = "default"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.sec_gold_vector_index"
ENDPOINT_NAME = "financial_rag_endpoint"
LLM_ENDPOINT = "databricks-meta-llama-3-70b-instruct" # Standard Databricks Foundation Model Endpoint

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ RAG Configuration")
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

# -----------------------------------------------------------------------------
# MAIN APP INTERFACE
# -----------------------------------------------------------------------------
st.title("📈 SEC 10-K RAG & Sentiment Analysis Engine")
st.caption("Ask natural language questions across SEC filings with FinBERT sentiment attribution.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "retrieved_docs" in msg:
            with st.expander("🔍 View Retrieved 10-K Sources & FinBERT Scores"):
                for doc in msg["retrieved_docs"]:
                    st.json(doc)

# Handle User Input
if user_query := st.chat_input("Ask a question about 10-K Risk Factors (e.g., foreign exchange risks, supply chain delays)..."):
    
    # 1. Show user message in chat
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Vector Search Retrieval
    with st.chat_message("assistant"):
        with st.status("🔍 Searching SEC Vector Search Index...", expanded=False) as status:
            
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
                    "filing_date", 
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
                ticker, filing_date, accession, chunk_idx, text, sent_label, sent_neg, sent_pos = doc
                
                doc_info = {
                    "source_num": i,
                    "ticker": ticker,
                    "filing_date": filing_date,
                    "chunk_index": chunk_idx,
                    "sentiment": sent_label,
                    "negative_probability": round(sent_neg, 4),
                    "excerpt": text
                }
                structured_docs.append(doc_info)
                
                context_str += f"\n[SOURCE {i}] Ticker: {ticker} | Date: {filing_date} | FinBERT Sentiment: {sent_label} (Neg Prob: {sent_neg:.2f})\nExcerpt: {text}\n"

            prompt = f"""You are a senior financial analyst. Answer the user's question using ONLY the provided SEC 10-K excerpts. 
Cite source numbers [SOURCE X] for every fact or claim you make.

USER QUESTION:
{user_query}

SEC 10-K CONTEXT:
{context_str}
"""

            # 4. Stream LLM Response using Databricks Foundation Model Serving
            try:
                response = w.serving_endpoints.query(
                    name=LLM_ENDPOINT,
                    messages=[
                        {"role": "system", "content": "You are a helpful financial analyst assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=800
                )
                
                llm_answer = response.choices[0].message.content
                st.markdown(llm_answer)

                # 5. Render Expandable Sources & FinBERT Metrics
                with st.expander("📊 Inspect Retrieved Evidence & FinBERT Sentiment Analysis"):
                    for doc in structured_docs:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**Source [{doc['source_num']}] — Ticker: {doc['ticker']} ({doc['filing_date']})**")
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
                    "retrieved_docs": structured_docs
                })

            except Exception as e:
                st.error(f"Failed to query LLM Endpoint '{LLM_ENDPOINT}': {str(e)}")