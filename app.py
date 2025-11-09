import streamlit as st
import pandas as pd
import openai
from transformers import pipeline
import re

# ------------------------------
# PAGE CONFIGURATION
# ------------------------------
st.set_page_config(page_title="Search Visibility Agent", layout="wide")
st.title("🔍 AI Search Visibility Agent")

st.markdown("""
This app checks how often a **brand or product** appears across AI-generated content.
It creates realistic search queries, sends them to an LLM (OpenAI GPT model),
and analyzes **visibility** and **sentiment**.
""")

# ------------------------------
# USER INPUTS
# ------------------------------
brand = st.text_input("Brand", "Glade")
products = st.text_input("Products", "Glade Plug-Ins, Glade Candles")
category = st.text_input("Category", "Air Care")
competitors = st.text_input("Competitors", "Febreze, Air Wick")
num_queries = st.slider("Number of queries to test", 10, 100, 20)

# Securely read API key from Streamlit secrets
if "OPENAI_API_KEY" not in st.secrets:
    st.warning("⚠️ Add your OpenAI API key under **App → Edit secrets** in Streamlit Cloud.")
else:
    openai.api_key = st.secrets["OPENAI_API_KEY"]

# ------------------------------
# FUNCTIONS
# ------------------------------

def generate_queries(brand, products, category, competitors, n):
    """Generate n realistic search queries for the brand."""
    base = [
        f"best {category} brands",
        f"{brand} {category} reviews",
        f"is {brand} {products.split(',')[0]} safe",
        f"{brand} vs {competitors.split(',')[0] if competitors else 'competitors'}",
        f"eco friendly {category} options",
    ]
    queries = [f"{q} ({i})" for i in range(n // 5 + 1) for q in base]
    return queries[:n]


def run_queries(queries):
    """Send each query to the OpenAI GPT model."""
    responses = []
    progress = st.progress(0)
    for i, q in enumerate(queries):
        try:
            r = openai.chat.completions.create(
                model="gpt-4o-mini",  # lightweight model for cost-efficiency
                messages=[{"role": "user", "content": q}],
                max_tokens=250
            )
            text = r.choices[0].message.content
        except Exception as e:
            text = f"Error: {e}"
        responses.append(text)
        progress.progress((i + 1) / len(queries))
    return responses


sentiment_analyzer = pipeline("sentiment-analysis")


def analyze(queries, responses, brand):
    """Analyze sentiment and visibility for each response."""
    data = []
    for q, r in zip(queries, responses):
        visible = brand.lower() in r.lower()
        sentiment = sentiment_analyzer(r[:512])[0]["label"]
        urls = re.findall(r'https?://\\S+', r)
        data.append({
            "query": q,
            "visible": visible,
            "sentiment": sen
