import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time

# ---------------------
# 1. App Title
# ---------------------
st.set_page_config(page_title="Realtime News Sentiment Classifier", layout="wide")
st.title("📰 Real-time News Sentiment Classification Dashboard")

# ---------------------
# 2. Load Model
# ---------------------
@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.pkl")   # replace with your saved model

model = load_model()

# ---------------------
# 3. News Fetching (placeholder: replace with your function)
# ---------------------
def fetch_news():
    # Example dummy data – integrate your news API/webscraper here
    data = {
        "headline": ["Stock market rises", "Earthquake hits city", "Tech company announces layoffs"],
        "time": [time.ctime(), time.ctime(), time.ctime()]
    }
    return pd.DataFrame(data)

# ---------------------
# 4. Classification
# ---------------------
def classify_text(text):
    return model.predict([text])[0]

# ---------------------
# 5. Streaming Loop
# ---------------------
if st.button("Fetch Latest News"):
    df = fetch_news()
    df["sentiment"] = df["headline"].apply(classify_text)
    st.write(df)

    # Visualization
    sentiment_counts = df["sentiment"].value_counts()
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind="bar", ax=ax)
    st.pyplot(fig)
