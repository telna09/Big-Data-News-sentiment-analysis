# app.py
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def classify(text):
    scores = analyzer.polarity_scores(text)
    c = scores["compound"]
    if c >= 0.05:
        label = "Positive"
    elif c <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return label, scores

st.set_page_config(page_title="News Sentiment Classifier", layout="centered")
st.title("News Sentiment Classifier")
st.write("Enter a news headline or short paragraph and click Analyze.")

text = st.text_area("News text", height=150)
if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        label, scores = classify(text)
        st.subheader(f"Sentiment: {label}")
        st.json(scores)

st.markdown("---")
st.write("Try examples:")
if st.button("Load sample positive"):
    st.experimental_set_query_params()
    st.experimental_rerun()
