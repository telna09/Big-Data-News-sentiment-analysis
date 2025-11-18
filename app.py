import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="News Sentiment Analyzer",
    page_icon="📰",
    layout="wide"
)

# Initialize VADER sentiment analyzer
@st.cache_resource
def load_analyzer():
    return SentimentIntensityAnalyzer()

analyzer = load_analyzer()

# Function to fetch headlines from RSS
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_headlines_from_rss(rss_url="http://feeds.bbci.co.uk/news/rss.xml", max_items=50):
    """Fetches headlines from an RSS feed URL"""
    try:
        resp = requests.get(rss_url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "xml")
        items = soup.find_all("item")
        headlines = []
        for it in items[:max_items]:
            title_tag = it.find("title")
            if title_tag:
                headlines.append(title_tag.get_text().strip())
        return headlines
    except Exception as e:
        st.error(f"RSS fetch failed: {e}")
        return []

# Function to analyze sentiment
def get_sentiment(text):
    """Analyze sentiment using VADER"""
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    
    if compound >= 0.05:
        sentiment = "Positive"
        emoji = "😊"
        color = "green"
    elif compound <= -0.05:
        sentiment = "Negative"
        emoji = "😟"
        color = "red"
    else:
        sentiment = "Neutral"
        emoji = "😐"
        color = "gray"
    
    return sentiment, scores, emoji, color

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Header
st.title("📰 Real-Time News Sentiment Analyzer")
st.markdown("### Analyze sentiment of news headlines using AI-powered sentiment analysis")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    analysis_mode = st.radio(
        "Choose Analysis Mode:",
        ["Single Headline", "Batch Analysis", "Live RSS Feed"]
    )
    
    st.divider()
    
    st.markdown("### 📊 Statistics")
    if st.session_state.analysis_history:
        total = len(st.session_state.analysis_history)
        positive = sum(1 for item in st.session_state.analysis_history if item['sentiment'] == 'Positive')
        negative = sum(1 for item in st.session_state.analysis_history if item['sentiment'] == 'Negative')
        neutral = sum(1 for item in st.session_state.analysis_history if item['sentiment'] == 'Neutral')
        
        st.metric("Total Analyzed", total)
        st.metric("Positive", positive, delta=f"{(positive/total*100):.1f}%")
        st.metric("Negative", negative, delta=f"{(negative/total*100):.1f}%")
        st.metric("Neutral", neutral, delta=f"{(neutral/total*100):.1f}%")
    else:
        st.info("No analysis yet. Start analyzing headlines!")
    
    st.divider()
    
    if st.button("🗑️ Clear History"):
        st.session_state.analysis_history = []
        st.rerun()

# Main content area
if analysis_mode == "Single Headline":
    st.subheader("🔍 Analyze Single Headline")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        headline = st.text_input(
            "Enter a news headline:",
            placeholder="e.g., Stock markets rally to record highs"
        )
    
    with col2:
        st.write("")
        st.write("")
        analyze_btn = st.button("🚀 Analyze", type="primary", use_container_width=True)
    
    if analyze_btn and headline:
        with st.spinner("Analyzing sentiment..."):
            sentiment, scores, emoji, color = get_sentiment(headline)
            
            # Add to history
            st.session_state.analysis_history.append({
                'headline': headline,
                'sentiment': sentiment,
                'compound': scores['compound'],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Display results
            st.success("Analysis Complete!")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"### {emoji} Sentiment: **:{color}[{sentiment}]**")
                st.markdown(f"**Headline:** {headline}")
            
            with col2:
                st.metric("Compound Score", f"{scores['compound']:.3f}")
            
            with col3:
                st.metric("Confidence", f"{abs(scores['compound'])*100:.1f}%")
            
            # Detailed scores
            st.subheader("📈 Detailed Sentiment Scores")
            score_df = pd.DataFrame({
                'Sentiment': ['Positive', 'Neutral', 'Negative'],
                'Score': [scores['pos'], scores['neu'], scores['neg']]
            })
            
            fig = go.Figure(data=[
    go.Bar(
        x=score_df['Sentiment'],
        y=score_df['Score'],
        marker_color=['green', 'gray', 'red'],
        text=score_df['Score'],
        texttemplate='%{text:.3f}',
        textposition='outside'
    )
])
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

elif analysis_mode == "Batch Analysis":
    st.subheader("📝 Batch Analysis")
    
    batch_input = st.text_area(
        "Enter multiple headlines (one per line):",
        height=200,
        placeholder="Enter headlines here...\nOne headline per line"
    )
    
    if st.button("🚀 Analyze All", type="primary"):
        if batch_input:
            headlines = [h.strip() for h in batch_input.split('\n') if h.strip()]
            
            if headlines:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                for i, headline in enumerate(headlines):
                    sentiment, scores, emoji, color = get_sentiment(headline)
                    results.append({
                        'Headline': headline,
                        'Sentiment': sentiment,
                        'Emoji': emoji,
                        'Compound': scores['compound'],
                        'Positive': scores['pos'],
                        'Neutral': scores['neu'],
                        'Negative': scores['neg']
                    })
                    
                    # Add to history
                    st.session_state.analysis_history.append({
                        'headline': headline,
                        'sentiment': sentiment,
                        'compound': scores['compound'],
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    progress_bar.progress((i + 1) / len(headlines))
                    status_text.text(f"Analyzing {i + 1}/{len(headlines)}...")
                
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                st.success(f"✅ Analyzed {len(results)} headlines!")
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                positive_count = sum(1 for r in results if r['Sentiment'] == 'Positive')
                negative_count = sum(1 for r in results if r['Sentiment'] == 'Negative')
                neutral_count = sum(1 for r in results if r['Sentiment'] == 'Neutral')
                
                with col1:
                    st.metric("😊 Positive", positive_count, f"{positive_count/len(results)*100:.1f}%")
                with col2:
                    st.metric("😐 Neutral", neutral_count, f"{neutral_count/len(results)*100:.1f}%")
                with col3:
                    st.metric("😟 Negative", negative_count, f"{negative_count/len(results)*100:.1f}%")
                
                # Pie chart
                st.subheader("📊 Sentiment Distribution")
                sentiment_counts = pd.DataFrame({
                    'Sentiment': ['Positive', 'Neutral', 'Negative'],
                    'Count': [positive_count, neutral_count, negative_count]
                })
                
                fig = go.Figure(data=[
                   go.Pie(
                       labels=sentiment_counts['Sentiment'],
                       values=sentiment_counts['Count'],
                       marker=dict(colors=['green', 'gray', 'red'])
                    )
                ])
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Results table
                st.subheader("📋 Detailed Results")
                results_df = pd.DataFrame(results)
                st.dataframe(
                    results_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("⚠️ Please enter at least one headline.")

elif analysis_mode == "Live RSS Feed":
    st.subheader("📡 Live RSS Feed Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        rss_url = st.text_input(
            "RSS Feed URL:",
            value="http://feeds.bbci.co.uk/news/rss.xml",
            help="Enter a valid RSS feed URL"
        )
    
    with col2:
        st.write("")
        st.write("")
        fetch_btn = st.button("🔄 Fetch & Analyze", type="primary", use_container_width=True)
    
    if fetch_btn:
        with st.spinner("Fetching headlines from RSS feed..."):
            headlines = fetch_headlines_from_rss(rss_url, max_items=30)
            
            if headlines:
                st.success(f"✅ Fetched {len(headlines)} headlines!")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                for i, headline in enumerate(headlines):
                    sentiment, scores, emoji, color = get_sentiment(headline)
                    results.append({
                        'Headline': headline,
                        'Sentiment': sentiment,
                        'Emoji': emoji,
                        'Compound': scores['compound']
                    })
                    
                    # Add to history
                    st.session_state.analysis_history.append({
                        'headline': headline,
                        'sentiment': sentiment,
                        'compound': scores['compound'],
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    progress_bar.progress((i + 1) / len(headlines))
                    status_text.text(f"Analyzing {i + 1}/{len(headlines)}...")
                
                progress_bar.empty()
                status_text.empty()
                
                # Summary
                positive_count = sum(1 for r in results if r['Sentiment'] == 'Positive')
                negative_count = sum(1 for r in results if r['Sentiment'] == 'Negative')
                neutral_count = sum(1 for r in results if r['Sentiment'] == 'Neutral')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("😊 Positive", positive_count, f"{positive_count/len(results)*100:.1f}%")
                with col2:
                    st.metric("😐 Neutral", neutral_count, f"{neutral_count/len(results)*100:.1f}%")
                with col3:
                    st.metric("😟 Negative", negative_count, f"{negative_count/len(results)*100:.1f}%")
                
                # Visualization
                st.subheader("📊 Real-Time Sentiment Overview")
                sentiment_data = pd.DataFrame({
                    'Sentiment': ['Positive', 'Neutral', 'Negative'],
                    'Count': [positive_count, neutral_count, negative_count]
                })
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=sentiment_data['Sentiment'],
                        y=sentiment_data['Count'],
                        marker_color=['green', 'gray', 'red'],
                        text=sentiment_data['Count'],
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    title="Sentiment Distribution",
                    xaxis_title="Sentiment",
                    yaxis_title="Count",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display headlines
                st.subheader("📰 Analyzed Headlines")
                for result in results:
                    with st.expander(f"{result['Emoji']} {result['Headline'][:100]}..."):
                        st.write(f"**Sentiment:** {result['Sentiment']}")
                        st.write(f"**Compound Score:** {result['Compound']:.3f}")
            else:
                st.error("❌ Failed to fetch headlines. Please check the RSS URL.")

# Analysis History Section
if st.session_state.analysis_history:
    st.divider()
    st.subheader("📜 Analysis History")
    
    history_df = pd.DataFrame(st.session_state.analysis_history)
    
    # Time series chart
    if len(history_df) > 1:
        st.line_chart(history_df.set_index('timestamp')['compound'])
    
    # Show recent analyses
    with st.expander("View Recent Analyses", expanded=False):
        st.dataframe(
            history_df.tail(10).sort_values('timestamp', ascending=False),
            use_container_width=True,
            hide_index=True
        )

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Powered by VADER Sentiment Analysis | Built with Streamlit</p>
    </div>
""", unsafe_allow_html=True)
