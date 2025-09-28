import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time

# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(page_title="Realtime News Sentiment Dashboard", layout="wide")
st.title("📰 Real-time News Sentiment Classification Dashboard")

# -----------------------------
# Load model & vectorizer
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("sentiment_pipeline_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# -----------------------------
# Dummy news fetching function
# (replace this with real API call or your scraping function)
# -----------------------------
def fetch_news():
    data = {
        "headline": [
            "Stock market rises after positive earnings",
            "Earthquake hits city center, causing damage",
            "Tech company announces mass layoffs"
        ],
        "time": [time.ctime(), time.ctime(), time.ctime()]
    }
    return pd.DataFrame(data)

# -----------------------------
# Classification function
# -----------------------------
def classify_text(text):
    X = vectorizer.transform([text])
    return model.predict(X)[0]

# -----------------------------
# Fetch & classify news
# -----------------------------
if st.button("📡 Fetch Latest News"):
    df = fetch_news()
    df["sentiment"] = df["headline"].apply(classify_text)
    
    st.subheader("📰 Classified News Headlines")
    st.dataframe(df)

    # Sentiment distribution
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = df["sentiment"].value_counts()
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# -----------------------------
# User input section
# -----------------------------
st.subheader("✍️ Try Your Own Headline")
user_input = st.text_area("Enter a news headline:")
if user_input:
    pred = classify_text(user_input)
    st.success(f"**Predicted Sentiment:** {pred}")
