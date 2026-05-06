"""
Balti Sentiment Analysis Web App
Run with: streamlit run app.py
"""

import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack
import pandas as pd
import plotly.express as px
from datetime import datetime
import re

# Page configuration
st.set_page_config(
    page_title="Balti Sentiment Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
    }
    .main-header p {
        color: #e0e0e0;
        margin: 0;
    }
    .positive-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .negative-box {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .neutral-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .emoji-large {
        font-size: 4rem;
    }
    .confidence-bar {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model and resources
@st.cache_resource
def load_model():
    try:
        model = joblib.load('balti_best_model.pkl')
        vectorizer = joblib.load('balti_vectorizer.pkl')
        pos_lex = joblib.load('positive_lexicon.pkl')
        neg_lex = joblib.load('negative_lexicon.pkl')
        neu_lex = joblib.load('neutral_lexicon.pkl')
        model_type = joblib.load('balti_model_type.pkl')
        return model, vectorizer, pos_lex, neg_lex, neu_lex, model_type
    except FileNotFoundError:
        st.error(" Model not found! Please run model_train.py first.")
        st.stop()
    except Exception as e:
        st.error(f" Error loading model: {e}")
        st.stop()

# Lexicon feature extraction
def get_lexicon_features(text, pos_lex, neg_lex, neu_lex):
    text_lower = text.lower()
    words = text_lower.split()
    
    pos_count = sum(1 for w in words if w in pos_lex)
    neg_count = sum(1 for w in words if w in neg_lex)
    neu_count = sum(1 for w in words if w in neu_lex)
    
    for phrase in pos_lex:
        if ' ' in phrase and phrase in text_lower:
            pos_count += 1
    for phrase in neg_lex:
        if ' ' in phrase and phrase in text_lower:
            neg_count += 1
    for phrase in neu_lex:
        if ' ' in phrase and phrase in text_lower:
            neu_count += 1
    
    return [pos_count, neg_count, neu_count]

# Predict sentiment
def predict_sentiment(text, model, vectorizer, pos_lex, neg_lex, neu_lex, model_type):
    text_vec = vectorizer.transform([text])
    lex_features = get_lexicon_features(text, pos_lex, neg_lex, neu_lex)
    lex_vec = np.array([lex_features])
    
    if model_type == 'BernoulliNB':
        lex_vec = (lex_vec > 0).astype(int)
    
    combined = hstack([text_vec, lex_vec])
    prediction = model.predict(combined)[0]
    probabilities = model.predict_proba(combined)[0]
    
    return prediction, probabilities, lex_features

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/language.png", width=80)
    st.title(" About")
    st.write("""
    **Balti Sentiment Analyzer** uses Machine Learning to detect emotions in Balti language text.
    
    ### Features:
    -  Real-time sentiment detection
    -  Confidence scores
    -  Analysis history
    -  Export results
    
    ### Model Info:
    - Algorithm: Naive Bayes
    - Training data: 1500+ Balti sentences
    - Accuracy: ~75-85%
    
    ### Balti Language:
    Balti is a Tibetic language spoken in Gilgit-Baltistan, Pakistan and parts of India.
    """)
    
    st.divider()
    
    st.subheader(" Settings")
    auto_clear = st.checkbox("Auto-clear after analysis", value=False)
    show_details = st.checkbox("Show detailed analysis", value=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1> Balti Sentiment Analyzer</h1>
    <p>بلتی جذباتی تجزیہ کار | བལ་ཏི་ སེམས་ཁམས་ དབྱེ་བའི་ ལག་ཆ</p>
</div>
""", unsafe_allow_html=True)

# Load model
model, vectorizer, pos_lex, neg_lex, neu_lex, model_type = load_model()

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []
if 'analysis_count' not in st.session_state:
    st.session_state.analysis_count = 0

# Main input area
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader(" Enter Balti Text")
    
    # Example suggestions
    examples = [
        "chi liyahmo jaq chi yd",
        "na la maf bs",
        "xan ni 2 song sed",
        "na la khash yod",
        "na la dose yod",
        "na la thaday yod"
    ]
    
    selected_example = st.selectbox("Try an example:", ["Custom text"] + examples)
    
    if selected_example != "Custom text":
        user_input = st.text_area("Balti Text:", value=selected_example, height=120)
    else:
        user_input = st.text_area("Balti Text:", placeholder="Enter Balti text here...", height=120)
    
    analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

with col2:
    st.subheader(" Tips")
    st.info("""
    **Examples to try:**
    - Positive: `chi liyahmo jaq chi yd` (What a great day)
    - Negative: `na la maf bs` (I feel sorry)
    - Neutral: `xan ni 2 song sed` (Its night 2 am)
    
    Write complete sentences for best results!
    """)

# Analysis
if analyze_button and user_input:
    with st.spinner("Analyzing sentiment..."):
        # Predict
        sentiment, probabilities, lex_features = predict_sentiment(
            user_input, model, vectorizer, pos_lex, neg_lex, neu_lex, model_type
        )
        
        # Get confidence
        class_names = model.classes_
        confidence_dict = {class_names[i]: probabilities[i] for i in range(len(class_names))}
        confidence = max(probabilities)
        
        # Update history
        st.session_state.analysis_count += 1
        st.session_state.history.append({
            'id': st.session_state.analysis_count,
            'text': user_input,
            'sentiment': sentiment,
            'confidence': confidence,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Display result
        st.divider()
        st.subheader(" Analysis Result")
        
        # Result boxes
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if sentiment == 'positive':
                st.markdown(f"""
                <div class="positive-box">
                    <div class="emoji-large">😊</div>
                    <h2 style="color: #28a745;">POSITIVE</h2>
                    <p>Positive sentiment detected</p>
                </div>
                """, unsafe_allow_html=True)
            elif sentiment == 'negative':
                st.markdown(f"""
                <div class="negative-box">
                    <div class="emoji-large">😞</div>
                    <h2 style="color: #dc3545;">NEGATIVE</h2>
                    <p>Negative sentiment detected</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="neutral-box">
                    <div class="emoji-large">😐</div>
                    <h2 style="color: #ffc107;">NEUTRAL</h2>
                    <p>Neutral sentiment detected</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Confidence Score", f"{confidence:.1%}")
            st.metric("Analysis #", st.session_state.analysis_count)
        
        with col3:
            st.metric("Text Length", f"{len(user_input.split())} words")
            st.metric("Characters", len(user_input))
        
        # Confidence bars
        st.subheader(" Confidence Scores")
        
        colors = {'positive': '#28a745', 'negative': '#dc3545', 'neutral': '#ffc107'}
        
        for sent, prob in confidence_dict.items():
            st.markdown(f"**{sent.upper()}**")
            st.progress(prob, text=f"{prob:.1%}")
        
        # Detailed analysis
        if show_details:
            st.subheader(" Detailed Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Lexicon Features:**")
                st.write(f"- Positive words found: {lex_features[0]}")
                st.write(f"- Negative words found: {lex_features[1]}")
                st.write(f"- Neutral words found: {lex_features[2]}")
            
            with col2:
                st.markdown("**Text Statistics:**")
                words = user_input.lower().split()
                st.write(f"- Total words: {len(words)}")
                st.write(f"- Unique words: {len(set(words))}")
                st.write(f"- Avg word length: {sum(len(w) for w in words)/len(words):.1f}")
            
            # Highlight keywords
            st.markdown("**Detected Keywords:**")
            pos_keywords = [w for w in words if w in pos_lex]
            neg_keywords = [w for w in words if w in neg_lex]
            neu_keywords = [w for w in words if w in neu_lex]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if pos_keywords:
                    st.success(f"Positive: {', '.join(pos_keywords[:5])}")
            with col2:
                if neg_keywords:
                    st.error(f"Negative: {', '.join(neg_keywords[:5])}")
            with col3:
                if neu_keywords:
                    st.warning(f"Neutral: {', '.join(neu_keywords[:5])}")

# History section
if len(st.session_state.history) > 0:
    st.divider()
    st.subheader(" Analysis History")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create history dataframe
        history_df = pd.DataFrame(st.session_state.history)
        
        # Add color indicator
        history_df['indicator'] = history_df['sentiment'].apply(
            lambda x: '' if x == 'positive' else ('' if x == 'negative' else '')
        )
        
        # Display history
        display_df = history_df[['indicator', 'text', 'sentiment', 'confidence', 'timestamp']]
        display_df.columns = ['', 'Text', 'Sentiment', 'Confidence', 'Time']
        st.dataframe(display_df, use_container_width=True)
    
    with col2:
        # Sentiment distribution pie chart
        sentiment_counts = history_df['sentiment'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={'positive': '#28a745', 'negative': '#dc3545', 'neutral': '#ffc107'},
            title="Sentiment Distribution"
        )
        fig.update_layout(height=300, margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
    
    # Export button
    if st.button(" Export History to CSV"):
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"balti_sentiment_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Clear history button
    if st.button(" Clear History", type="secondary"):
        st.session_state.history = []
        st.rerun()

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Balti Sentiment Analyzer v1.0 | Built with Streamlit  for Balti Language Community</p>
    <p>🟢 Positive | 🔴 Negative | 🟡 Neutral</p>
</div>
""", unsafe_allow_html=True)
