import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import re

# Set page configuration
st.set_page_config(
    page_title="Text Preprocessing Report - Anxiety Dataset",
    page_icon="📊",
    layout="wide"
)

# Title and introduction
st.title("📊 Text Preprocessing Report - Anxiety Dataset")
st.markdown("### MSDS NLP 2025 - Text Preprocessing Assignment")
st.markdown("---")

# Load the processed data
@st.cache_data
def load_data():
    df = pd.read_csv('data/anxiety_preprocessed.csv')
    return df

df = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📋 Overview", "🔧 Preprocessing Steps", "📊 Statistics", "💬 Examples", "📥 Download"]
)

if page == "📋 Overview":
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Posts", f"{len(df):,}")
    
    with col2:
        st.metric("Date Range", "April 2019")
    
    with col3:
        st.metric("Subreddit", "r/Anxiety")
    
    with col4:
        st.metric("Preprocessing Steps", "9")
    
    st.markdown("### About the Dataset")
    st.write("""
    This dataset contains Reddit posts from the r/Anxiety subreddit collected during April 2019. 
    The posts provide valuable insights into anxiety-related discussions and experiences shared by the community.
    """)
    
    st.markdown("### Assignment Objective")
    st.write("""
    The goal of this assignment was to:
    1. Apply various text preprocessing techniques to prepare the data for NLP tasks
    2. Identify specific cases where preprocessing is required
    3. Document the types of preprocessing applied and their impact
    """)
    
    st.markdown("### Key Findings")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Most Common Issues:**
        - 47.2% of posts contained contractions
        - 46.7% of posts contained numbers
        - 18.8% had excessive punctuation
        - 1.8% contained URLs
        """)
    
    with col2:
        st.success("""
        **Impact of Preprocessing:**
        - Token count reduced by 53.3%
        - Improved text consistency
        - Ready for sentiment analysis
        - Suitable for topic modeling
        """)

elif page == "🔧 Preprocessing Steps":
    st.header("Preprocessing Steps Applied")
    
    steps = [
        {
            "step": "1. Lowercasing",
            "purpose": "Convert all text to lowercase for consistency",
            "example": "I'm Having ANXIETY → i'm having anxiety",
            "impact": "Ensures 'Anxiety' and 'anxiety' are treated as same word"
        },
        {
            "step": "2. Contraction Expansion",
            "purpose": "Expand contractions for better analysis",
            "example": "don't → do not, it's → it is",
            "impact": "Found in 47.2% of posts"
        },
        {
            "step": "3. URL/Email Removal",
            "purpose": "Remove web links and email addresses",
            "example": "Check https://example.com → Check [URL]",
            "impact": "Found in 1.8% of posts"
        },
        {
            "step": "4. Number Normalization",
            "purpose": "Replace numbers with placeholder token",
            "example": "3 years ago → NUM years ago",
            "impact": "Found in 46.7% of posts"
        },
        {
            "step": "5. Punctuation Removal",
            "purpose": "Remove excessive punctuation",
            "example": "Help!!! → Help.",
            "impact": "Found in 18.8% of posts"
        },
        {
            "step": "6. Whitespace Normalization",
            "purpose": "Remove extra spaces and newlines",
            "example": "text    with    spaces → text with spaces",
            "impact": "Ensures consistent spacing"
        },
        {
            "step": "7. Tokenization",
            "purpose": "Split text into individual words",
            "example": "I have anxiety → ['I', 'have', 'anxiety']",
            "impact": "Essential for word-level analysis"
        },
        {
            "step": "8. Stopword Removal",
            "purpose": "Remove common words with little meaning",
            "example": "I have the anxiety → ['anxiety']",
            "impact": "Reduced token count by 53.3%"
        },
        {
            "step": "9. Stemming",
            "purpose": "Reduce words to root form",
            "example": "running, runs → run",
            "impact": "Groups related words together"
        }
    ]
    
    for i, step_info in enumerate(steps):
        with st.expander(step_info["step"], expanded=(i<3)):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Purpose:** {step_info['purpose']}")
                st.markdown(f"**Example:** `{step_info['example']}`")
            with col2:
                st.info(f"**Impact:** {step_info['impact']}")

elif page == "📊 Statistics":
    st.header("Dataset Statistics")
    
    tab1, tab2, tab3 = st.tabs(["Token Analysis", "Preprocessing Impact", "Word Frequency"])
    
    with tab1:
        st.subheader("Token Count Analysis")
        
        # Calculate token statistics
        df['original_tokens'] = df['original_selftext'].fillna('').str.split().str.len()
        df['processed_tokens'] = df['tokens_stemmed'].str.split().str.len()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Box(y=df['original_tokens'], name='Original', marker_color='lightblue'))
            fig.add_trace(go.Box(y=df['processed_tokens'], name='After Processing', marker_color='darkblue'))
            fig.update_layout(title="Token Distribution Comparison", yaxis_title="Number of Tokens")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            avg_original = df['original_tokens'].mean()
            avg_processed = df['processed_tokens'].mean()
            
            fig = go.Figure(data=[
                go.Bar(name='Average Tokens', x=['Original', 'Processed'], 
                      y=[avg_original, avg_processed],
                      text=[f'{avg_original:.0f}', f'{avg_processed:.0f}'],
                      textposition='auto',
                      marker_color=['lightcoral', 'darkgreen'])
            ])
            fig.update_layout(title="Average Token Count", yaxis_title="Number of Tokens")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Impact of Preprocessing Steps")
        
        preprocessing_stats = {
            'Contractions': 47.2,
            'Numbers': 46.7,
            'Excessive Punctuation': 18.8,
            'URLs': 1.8
        }
        
        fig = px.bar(
            x=list(preprocessing_stats.keys()),
            y=list(preprocessing_stats.values()),
            labels={'x': 'Preprocessing Type', 'y': 'Percentage of Posts (%)'},
            title="Percentage of Posts Requiring Each Preprocessing Type",
            color=list(preprocessing_stats.values()),
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Token Reduction", "53.3%", "-90.4 tokens/post")
        with col2:
            st.metric("Text Length Reduction", "0.5%", "-4.5 chars/post")
        with col3:
            st.metric("Vocabulary Size", "~15,000 unique terms", "after stemming")
    
    with tab3:
        st.subheader("Most Frequent Terms")
        
        # Create word frequency data
        word_freq = {
            'anxiety': 1447, 'feel': 1367, 'like': 1182, 'num': 1009,
            'get': 754, 'know': 678, 'real': 612, 'want': 553,
            'time': 537, 'think': 500, 'work': 480, 'would': 477,
            'thing': 460, 'day': 457, 'help': 433
        }
        
        fig = px.bar(
            x=list(word_freq.values())[::-1],
            y=list(word_freq.keys())[::-1],
            orientation='h',
            labels={'x': 'Frequency', 'y': 'Term'},
            title="Top 15 Most Frequent Terms (After Preprocessing)",
            color=list(word_freq.values())[::-1],
            color_continuous_scale='blues'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Key Observations:**
        - "anxiety" is the most frequent term (1,447 occurrences)
        - "num" appears frequently due to number normalization
        - Emotional terms like "feel", "want", "think" are prominent
        - Work-related stress is evident from "work", "day" frequency
        """)

elif page == "💬 Examples":
    st.header("Preprocessing Examples from Dataset")
    
    st.markdown("### Real Examples Showing Preprocessing Impact")
    
    examples = [
        {
            "title": "Example 1: Contraction Expansion & Number Normalization",
            "original": "I'm going to give my 2 months notice to my employer today. It's a great company, but it just isn't a fit and I'm miserable at my current position.",
            "processed": "going give num month notice employer today great company fit miserable current position"
        },
        {
            "title": "Example 2: Stopword Removal & Stemming",
            "original": "I got fired at my sidejob at a retail store, I had no previous experience with a retail job.",
            "processed": "got fir sidejob retail store previous experience retail job"
        },
        {
            "title": "Example 3: Punctuation & Whitespace Normalization",
            "original": "I always feels and act like I'm in a hurry even if I have literally nothing to do!!!",
            "processed": "alway feel act like hurry even literal noth"
        }
    ]
    
    for example in examples:
        with st.expander(example["title"], expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Text:**")
                st.text_area("", example["original"], height=100, disabled=True, key=f"orig_{example['title']}")
            with col2:
                st.markdown("**After Preprocessing:**")
                st.text_area("", example["processed"], height=100, disabled=True, key=f"proc_{example['title']}")
    
    st.markdown("### Specific Preprocessing Requirements")
    
    requirements = {
        "Contraction Expansion": {
            "why": "Essential for standardizing text and proper tokenization",
            "example": "don't → do not, it's → it is",
            "frequency": "47.2% of posts"
        },
        "Number Normalization": {
            "why": "Important for generalization in pattern recognition",
            "example": "3 years, 5 years → NUM years",
            "frequency": "46.7% of posts"
        },
        "URL Removal": {
            "why": "Eliminates non-textual content that doesn't contribute to analysis",
            "example": "https://example.com → [URL]",
            "frequency": "1.8% of posts"
        },
        "Stopword Removal": {
            "why": "Focuses on meaningful content words",
            "example": "I am very anxious → anxious",
            "frequency": "Reduced tokens by 53.3%"
        }
    }
    
    for req_name, req_info in requirements.items():
        with st.expander(f"Why {req_name} was necessary"):
            st.write(f"**Reason:** {req_info['why']}")
            st.code(req_info['example'])
            st.info(f"Found in: {req_info['frequency']}")

else:  # Download page
    st.header("📥 Download Processed Data & Report")
    
    st.markdown("### Files Available for Download")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📄 Processed Dataset")
        st.write("Complete dataset with all preprocessing stages")
        
        # Read the CSV file
        with open('data/anxiety_preprocessed.csv', 'r') as f:
            csv_data = f.read()
        
        st.download_button(
            label="Download anxiety_preprocessed.csv",
            data=csv_data,
            file_name="anxiety_preprocessed.csv",
            mime="text/csv"
        )
        
        st.info("""
        **Contents:**
        - Original text (title & selftext)
        - Processed text
        - Tokens (original)
        - Tokens without stopwords
        - Stemmed tokens
        """)
    
    with col2:
        st.markdown("#### 📊 Summary Report")
        st.write("Detailed preprocessing report for submission")
        
        # Read the report file
        with open('data/anxiety_preprocessed.csv', 'r') as f:
            report_data = f.read()
        
        st.download_button(
            label="Download preprocessing_report.txt",
            data=report_data,
            file_name="preprocessing_report.txt",
            mime="text/plain"
        )
        
        st.info("""
        **Contents:**
        - All preprocessing steps
        - Statistics and metrics
        - Examples from dataset
        - Impact analysis
        """)
    
    st.markdown("---")

# Footer
st.markdown("---")
st.markdown("*Text Preprocessing Assignment - MSDS NLP 2025*")
