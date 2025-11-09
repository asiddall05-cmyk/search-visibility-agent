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


# --- Analyze sentiment and visibility ---
sentiment_analyzer = pipeline("sentiment-analysis")

def analyze(queries, responses, brand):
    """Analyze sentiment and visibility for each response."""
    data = []
    for q, r in zip(queries, responses):
        visible = brand.lower() in r.lower()
        sentiment = sentiment_analyzer(r[:512])[0]["label"]
        urls = re.findall(r'https?://\S+', r)
        data.append({
            "query": q,
            "visible": visible,
            "sentiment": sentiment,
            "urls": ";".join(urls)
        })  # ✅ properly closed dictionary
    return pd.DataFrame(data)

# ------------------------------
# MAIN EXECUTION
# ------------------------------

if st.button("🚀 Run Search Visibility Analysis"):
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("You need to add your OpenAI API key first.")
    else:
        with st.spinner("Generating queries..."):
            queries = generate_queries(brand, products, category, competitors, num_queries)

        with st.spinner("Running queries against GPT..."):
            responses = run_queries(queries)

        with st.spinner("Analyzing responses..."):
            df = analyze(queries, responses, brand)
            st.session_state["results"] = df
            df.to_csv("results.csv", index=False)
            st.success("✅ Analysis complete!")

# ------------------------------
# RESULTS
# ------------------------------

if "results" in st.session_state:
    df = st.session_state["results"]
    st.subheader("Results Summary")

    col1, col2 = st.columns(2)
    with col1:
        visibility_rate = df["visible"].mean() * 100
        st.metric("Brand Visibility Rate", f"{visibility_rate:.1f}%")
    with col2:
        sentiment_counts = df["sentiment"].value_counts(normalize=True) * 100
        if "POSITIVE" in sentiment_counts:
            st.metric("Positive Mentions", f"{sentiment_counts['POSITIVE']:.1f}%")

    st.bar_chart(sentiment_counts)

    st.write("### Detailed Results")
    st.dataframe(df.head(100))

    st.download_button("📥 Download Full CSV", df.to_csv(index=False),
                       "search_visibility_results.csv", "text/csv")

st.caption("Built with ❤️ using Streamlit + OpenAI API.")
