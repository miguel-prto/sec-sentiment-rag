import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="SEC Sentiment & RAG Engine", layout="wide")
st.title("📊 Financial Sentiment & SEC Filing RAG Engine")

tab1, tab2 = st.tabs(["📈 Sentiment Analytics", "💬 SEC Chatbot (RAG)"])

# Mock/Connected Data Loader
@st.cache_data
def load_gold_data():
    # Replace with Databricks SQL Connector or PySpark read in production
    return pd.DataFrame({
        "ticker": ["AAPL", "AAPL", "MSFT", "MSFT", "NVDA", "NVDA"],
        "sentiment": ["negative", "neutral", "positive", "negative", "positive", "positive"],
        "chunk": [
            "Supply chain constraints in East Asia could impact Q4 gross margins.",
            "Research and development expenses increased by 12% year-over-year.",
            "Cloud infrastructure adoption continues to drive recurring revenue.",
            "Cybersecurity compliance failure presents material operational risks.",
            "Demand for AI training compute clusters exceeds immediate supply.",
            "Data center expansion driving strong quarterly cash flow growth."
        ]
    })

df = load_gold_data()

with tab1:
    st.header("Corporate Sentiment Analysis (Item 1A Risk Factors)")
    sentiment_counts = df.groupby(["ticker", "sentiment"]).size().reset_index(name="count")
    
    fig = px.bar(
        sentiment_counts,
        x="ticker",
        y="count",
        color="sentiment",
        color_discrete_map={"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"},
        barmode="group",
        title="Risk Factor Sentiment Breakdown by Ticker"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Ask Questions About SEC Filings")
    query = st.text_input("Enter a query (e.g., 'What supply chain risks did Apple mention?'):")
    
    if query:
        st.write("🔍 *Searching Databricks Vector Index and generating response...*")
        
        # Retrieval Simulation Logic
        matching_chunks = df[df["ticker"] == "AAPL"]["chunk"].tolist()
        
        st.subheader("Context Retracted from Filing:")
        for idx, chunk in enumerate(matching_chunks, 1):
            st.info(f"**Chunk {idx}:** {chunk}")
            
        st.subheader("Generated AI Answer:")
        st.success(
            "Apple highlighted potential supply chain disruptions in East Asia as a primary factor "
            "that could negatively affect Q4 gross margins."
        )