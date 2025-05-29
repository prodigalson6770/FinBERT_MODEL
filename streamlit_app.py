import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from collections import Counter
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Financial Sentiment Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Company name mapping
@st.cache_data
def get_company_names():
    return {
        "AAPL": ["Apple", "iPhone", "iPad", "Mac", "iOS","Tim Cook"],
        "ADBE": ["Adobe", "Photoshop", "Creative Cloud"],
        "AMD": ["AMD", "Advanced Micro Devices", "Ryzen", "EPYC"],
        "AMAT": ["Applied Materials", "AMAT"],
        "AMZN": ["Amazon", "AWS", "Prime", "Bezos"],
        "ASML": ["ASML", "lithography"],
        "AVGO": ["Broadcom", "AVGO"],
        "AXP": ["American Express", "Amex"],
        "BA": ["Boeing", "737", "787"],
        "BAC": ["Bank of America", "BofA"],
        "BABA": ["Alibaba", "Jack Ma", "Taobao"],
        "BIDU": ["Baidu", "search engine"],
        "BKNG": ["Booking", "Priceline"],
        "C": ["Citigroup", "Citi"],
        "CAT": ["Caterpillar", "construction equipment"],
        "CMCSA": ["Comcast", "NBCUniversal"],
        "CRM": ["Salesforce", "CRM"],
        "CSCO": ["Cisco", "networking"],
        "CVX": ["Chevron", "oil"],
        "DIS": ["Disney", "Marvel", "Pixar", "ESPN"],
        "F": ["Ford", "F-150", "Mustang", "automotive"],
        "GOOG": ["Google", "Alphabet", "YouTube", "Android"],
        "GS": ["Goldman Sachs", "investment bank"],
        "HD": ["Home Depot", "hardware"],
        "HON": ["Honeywell", "aerospace"],
        "IBM": ["IBM", "Watson", "Red Hat"],
        "INTC": ["Intel", "processor", "chip"],
        "JNJ": ["Johnson & Johnson", "J&J", "pharmaceutical"],
        "JPM": ["JPMorgan", "Chase"],
        "KO": ["Coca Cola", "Coke"],
        "LLY": ["Eli Lilly", "pharmaceutical"],
        "LMT": ["Lockheed Martin", "defense"],
        "MA": ["Mastercard", "payment"],
        "MCD": ["McDonald's", "fast food"],
        "MRK": ["Merck", "pharmaceutical"],
        "MS": ["Morgan Stanley", "investment"],
        "MSI": ["Motorola", "communications"],
        "MSFT": ["Microsoft", "Windows", "Office", "Azure"],
        "NFLX": ["Netflix", "streaming"],
        "NKE": ["Nike", "sportswear"],
        "NVDA": ["Nvidia", "GPU", "graphics"],
        "ORCL": ["Oracle", "database"],
        "PEP": ["PepsiCo", "Pepsi"],
        "PFE": ["Pfizer", "pharmaceutical"],
        "PYPL": ["PayPal", "payment"],
        "QCOM": ["Qualcomm", "wireless"],
        "SBUX": ["Starbucks", "coffee"],
        "WMT": ["Walmart", "retail"],
        "T": ["AT&T", "telecommunications"],
        "TSLA": ["Tesla", "electric vehicle", "Musk", "EV"]
    }

# Load model with caching
@st.cache_resource
def load_finbert_model():
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Sentiment analysis function
def get_sentiment_with_confidence(text, tokenizer, model):
    try:
        if pd.isna(text) or text.strip() == "":
            return "neutral", 0.5

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]
        pred = torch.argmax(probs).item()
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = label_map[pred]
        confidence = probs[pred].item()
        return sentiment, confidence
    except Exception as e:
        st.error(f"Error processing text: {str(e)}")
        return "neutral", 0.5

# Company relation check
def is_company_related(title, ticker, company_names):
    if pd.isna(title):
        return False
    
    title_lower = title.lower()
    if ticker in company_names:
        for keyword in company_names[ticker]:
            if keyword.lower() in title_lower:
                return True
    return False

# Filter company news
def filter_company_news(group, ticker, company_names):
    company_news = group[group["title"].apply(lambda x: is_company_related(x, ticker, company_names))]
    return company_news

# Analysis function
def analyze_ticker(group, ticker, company_names):
    company_specific_news = filter_company_news(group, ticker, company_names)

    if len(company_specific_news) == 0:
        return {
            "action": "HOLD",
            "detailed_rating": "NEUTRAL",
            "reason": f"No company-specific news found for {ticker}. Insufficient information for recommendation.",
            "confidence_score": 0.0,
            "sentiment_distribution": "No data available",
            "total_articles": len(group),
            "company_specific_articles": 0,
            "pos_pct": 0,
            "neg_pct": 0,
            "neu_pct": 0
        }

    counts = Counter(company_specific_news["sentiment"])
    total_news = len(company_specific_news)
    total_articles = len(group)

    pos_pct = counts.get("positive", 0) / total_news * 100
    neg_pct = counts.get("negative", 0) / total_news * 100
    neu_pct = counts.get("neutral", 0) / total_news * 100

    avg_confidence = company_specific_news["confidence"].mean()

    # Determine action and detailed rating
    if pos_pct >= 70 and avg_confidence > 0.8:
        action = "STRONG BUY"
        detailed_rating = "OUTPERFORM"
        sentiment_focus = "positive"
    elif pos_pct >= 60 and pos_pct > neg_pct * 2:
        action = "BUY"
        detailed_rating = "MODERATE BUY"
        sentiment_focus = "positive"
    elif pos_pct > 50 and pos_pct > neg_pct * 1.5:
        action = "BUY"
        detailed_rating = "ACCUMULATE"
        sentiment_focus = "positive"
    elif pos_pct > neg_pct and pos_pct > 40:
        action = "WEAK BUY"
        detailed_rating = "OVERWEIGHT"
        sentiment_focus = "positive"
    elif neg_pct >= 70 and avg_confidence > 0.8:
        action = "STRONG SELL"
        detailed_rating = "UNDERPERFORM"
        sentiment_focus = "negative"
    elif neg_pct >= 60 and neg_pct > pos_pct * 2:
        action = "SELL"
        detailed_rating = "MODERATE SELL"
        sentiment_focus = "negative"
    elif neg_pct > 50 and neg_pct > pos_pct * 1.5:
        action = "SELL"
        detailed_rating = "UNDERWEIGHT"
        sentiment_focus = "negative"
    elif neg_pct > pos_pct and neg_pct > 40:
        action = "WEAK SELL"
        detailed_rating = "REDUCE"
        sentiment_focus = "negative"
    elif abs(pos_pct - neg_pct) < 10:
        action = "HOLD"
        detailed_rating = "NEUTRAL"
        sentiment_focus = "mixed"
    elif pos_pct > neg_pct:
        action = "WEAK HOLD"
        detailed_rating = "MARKET PERFORM"
        sentiment_focus = "mixed"
    else:
        action = "HOLD"
        detailed_rating = "NEUTRAL"
        sentiment_focus = "mixed"

    # Get supporting news
    if sentiment_focus == "positive":
        supporting_news = company_specific_news[company_specific_news["sentiment"] == "positive"]["title"].head(3).tolist()
    elif sentiment_focus == "negative":
        supporting_news = company_specific_news[company_specific_news["sentiment"] == "negative"]["title"].head(3).tolist()
    else:
        supporting_news = company_specific_news["title"].head(3).tolist()

    reason = f"Analysis of {total_news} company-specific news articles (out of {total_articles} total):\n"
    reason += f"Positive: {pos_pct:.1f}% | Negative: {neg_pct:.1f}% | Neutral: {neu_pct:.1f}%\n"
    reason += f"Average Confidence: {avg_confidence:.2f}\n\n"
    reason += f"Rating Explanation:\n"
    reason += f"Primary Rating: {action} | Alternative Rating: {detailed_rating}\n\n"
    reason += f"Supporting headlines:\n"

    if supporting_news:
        for i, title in enumerate(supporting_news, 1):
            reason += f"{i}. {title}\n"
    else:
        reason += "No relevant headlines found."

    return {
        "action": action,
        "detailed_rating": detailed_rating,
        "reason": reason,
        "confidence_score": avg_confidence,
        "sentiment_distribution": f"Pos: {pos_pct:.1f}%, Neg: {neg_pct:.1f}%, Neu: {neu_pct:.1f}%",
        "total_articles": total_articles,
        "company_specific_articles": total_news,
        "pos_pct": pos_pct,
        "neg_pct": neg_pct,
        "neu_pct": neu_pct,
        "supporting_news": supporting_news
    }

# Main Streamlit app
def main():
    st.title("📈 Financial Sentiment Analysis Dashboard")
    st.markdown("**Analyze financial news sentiment and get investment recommendations using FinBERT**")

    # Sidebar
    st.sidebar.header("Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Financial News CSV",
        type=['csv'],
        help="Upload a CSV file with columns: ticker, title"
    )

    if uploaded_file is not None:
        # Load data
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Loaded {len(df)} rows")
            
            # Validate required columns
            required_columns = ['ticker', 'title']
            if not all(col in df.columns for col in required_columns):
                st.error(f"CSV must contain columns: {required_columns}")
                return
            
            # Remove duplicates
            original_size = len(df)
            df = df.drop_duplicates(subset=['ticker', 'title'], keep='first')
            if original_size != len(df):
                st.sidebar.info(f"Removed {original_size - len(df)} duplicate rows")

        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            return

        # Ticker selection
        available_tickers = sorted(df['ticker'].unique())
        selected_tickers = st.sidebar.multiselect(
            "Select Tickers to Analyze",
            options=available_tickers,
            default=available_tickers[:1] if available_tickers else [],
            help="Choose one or more stock tickers to analyze"
        )

        if not selected_tickers:
            st.warning("Please select at least one ticker to analyze.")
            return

        # Analysis button
        if st.sidebar.button("🚀 Run Analysis", type="primary"):
            # Load model
            with st.spinner("Loading FinBERT model..."):
                tokenizer, model = load_finbert_model()
            
            company_names = get_company_names()
            
            # Filter data
            analysis_df = df[df["ticker"].isin(selected_tickers)].copy()
            
            # Run sentiment analysis
            with st.spinner("Analyzing sentiment..."):
                progress_bar = st.progress(0)
                sentiments = []
                confidences = []
                
                for i, title in enumerate(analysis_df["title"]):
                    sentiment, confidence = get_sentiment_with_confidence(title, tokenizer, model)
                    sentiments.append(sentiment)
                    confidences.append(confidence)
                    progress_bar.progress((i + 1) / len(analysis_df))
                
                analysis_df["sentiment"] = sentiments
                analysis_df["confidence"] = confidences
                progress_bar.empty()

            # Store results in session state
            st.session_state.analysis_df = analysis_df
            st.session_state.results = []
            
            # Analyze each ticker
            for ticker in selected_tickers:
                ticker_data = analysis_df[analysis_df["ticker"] == ticker]
                if len(ticker_data) > 0:
                    analysis = analyze_ticker(ticker_data, ticker, company_names)
                    analysis["ticker"] = ticker
                    st.session_state.results.append(analysis)

    # Display results if available
    if hasattr(st.session_state, 'results') and st.session_state.results:
        st.header("📊 Analysis Results")
        
        # Create tabs for each ticker
        if len(st.session_state.results) == 1:
            display_ticker_analysis(st.session_state.results[0], st.session_state.analysis_df)
        else:
            tabs = st.tabs([result["ticker"] for result in st.session_state.results])
            for tab, result in zip(tabs, st.session_state.results):
                with tab:
                    ticker_data = st.session_state.analysis_df[st.session_state.analysis_df["ticker"] == result["ticker"]]
                    display_ticker_analysis(result, ticker_data)

def display_ticker_analysis(result, ticker_data):
    ticker = result["ticker"]
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        action_color = get_action_color(result["action"])
        st.markdown(f"""
        <div class="metric-card">
            <h4>Recommendation</h4>
            <h2 style="color: {action_color};">{result["action"]}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Rating</h4>
            <h3>{result["detailed_rating"]}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Confidence Score</h4>
            <h3>{result["confidence_score"]:.2f}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Articles Analyzed</h4>
            <h3>{result["company_specific_articles"]}/{result["total_articles"]}</h3>
        </div>
        """, unsafe_allow_html=True)

    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution pie chart
        fig_pie = px.pie(
            values=[result["pos_pct"], result["neg_pct"], result["neu_pct"]],
            names=["Positive", "Negative", "Neutral"],
            title=f"Sentiment Distribution for {ticker}",
            color_discrete_map={
                "Positive": "#00CC96",
                "Negative": "#EF553B", 
                "Neutral": "#636EFA"
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Confidence distribution
        if len(ticker_data) > 0:
            fig_hist = px.histogram(
                ticker_data,
                x="confidence",
                title=f"Confidence Score Distribution for {ticker}",
                nbins=20,
                color_discrete_sequence=["#636EFA"]
            )
            fig_hist.update_layout(xaxis_title="Confidence Score", yaxis_title="Count")
            st.plotly_chart(fig_hist, use_container_width=True)

    # Detailed analysis
    st.subheader("📋 Detailed Analysis")
    st.markdown(f"""
    <div class="recommendation-box">
    <pre>{result["reason"]}</pre>
    </div>
    """, unsafe_allow_html=True)

    # Supporting headlines
    if "supporting_news" in result and result["supporting_news"]:
        st.subheader("📰 Key Headlines")
        for i, headline in enumerate(result["supporting_news"], 1):
            st.markdown(f"**{i}.** {headline}")

    # Download results
    if st.button(f"📥 Download {ticker} Results", key=f"download_{ticker}"):
        results_df = pd.DataFrame([result])
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"sentiment_analysis_{ticker}.csv",
            mime="text/csv",
            key=f"download_csv_{ticker}"
        )

def get_action_color(action):
    if "BUY" in action:
        return "#00CC96"
    elif "SELL" in action:
        return "#EF553B"
    else:
        return "#636EFA"

if __name__ == "__main__":
    main()
