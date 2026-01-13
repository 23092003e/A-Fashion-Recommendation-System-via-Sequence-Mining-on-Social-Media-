from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

IMAGE_DIR = Path(
    r"C:\Users\ADMIN\Desktop\ITDSIU21099_HoangVanManh\Fashion-Marketing-Automation-Solutions\images\original_images"
)

from utils.data_loader import DataLoader
from utils.model_loader import ModelLoader


@st.cache_data
def get_post_image(post_id):

    img_path = IMAGE_DIR / f"{post_id}.jpg"

    if img_path.exists():
        return Image.open(img_path)
    return None

# Page config
st.set_page_config(
    page_title="Fashion Recommendation System for Social Media",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for user-friendly interface
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #f1f3f8;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    .product-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    .product-card:hover {
        transform: translateY(-5px);
    }
    .confidence-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .match-high { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .match-medium { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .tag {
        display: inline-block;
        background: #f0f0f0;
        color: #555;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    .history-item {
        background: linear-gradient(135deg, #1f2933  0%, #111827  100%);
        border-radius: 14px;
        padding: 0.9rem;
        margin: 0.7rem 0;
        border-left: 4px solid #6366f1;
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
        display: flex;
        gap: 0.75rem;
        align-items: flex-start;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        transition: all 0.25s ease;
    }

    .history-item:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 24px rgba(0,0,0,0.18);
    }

    .history-author {
        font-weight: 700;
        color: #f1f3f8;
        font-size: 0.95rem;
    }

    .history-caption {
        color: #d1d5db;
        font-size: 0.85rem;
        margin-top: 0.25rem;
        line-height: 1.4;
    }

    .history-emoji {
        font-size: 1.4rem;
        margin-right: 0.4rem;
    }

    .history-sentiment {
        font-size: 0.75rem;
        font-weight: 600;
        color: #93c5fd;
    }
    
    .history-item {
        display: flex;
        gap: 0.7rem;
        align-items: flex-start;
    }

    .history-image {
        width: 60px;
        height: 60px;
        border-radius: 12px;
        object-fit: cover;
        flex-shrink: 0;
        
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 4px 12px rgba(0,0,0,0.35);
        background: #111827;
    }

    .history-content {
        flex: 1;
    }
    
</style>
""", unsafe_allow_html=True)


def get_base_path():
    return Path(__file__).resolve().parent.parent


@st.cache_resource
def load_data():
    base_path = get_base_path()
    data_loader = DataLoader(base_path)
    success = data_loader.load_all_data()
    if not success:
        return None
    return data_loader


@st.cache_resource
def load_model(embeddings_shape, vocab_size):
    _ = (embeddings_shape, vocab_size)
    base_path = get_base_path()
    data_loader = DataLoader(base_path)
    data_loader.load_all_data()
    if data_loader.post2idx is None or data_loader.embeddings is None:
        return None
    model_loader = ModelLoader(base_path)
    model_loader.load_hybrid_model(data_loader.embeddings, data_loader.post2idx)
    return model_loader


def safe_str(val, max_len=100):
    """Safely convert value to string"""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ''
    s = str(val)
    return s[:max_len] + '...' if len(s) > max_len else s


def get_match_level(score, max_score):
    """Get match level based on score"""
    ratio = score / max_score if max_score > 0 else 0
    if ratio > 0.8:
        return "Perfect Match! 🎯", "match-high"
    elif ratio > 0.5:
        return "Great Match! ✨", "match-medium"
    else:
        return "You May Like 👍", "confidence-badge"


def render_product_card(post_info, score, rank, max_score, data_loader):
    """Render a product recommendation card"""
    
    author = safe_str(post_info.get('postUser', 'Unknown'), 30)
    caption = safe_str(post_info.get('caption', ''), 200)
    likes = post_info.get('likesCount', 0)
    comments = post_info.get('commentsCount', 0)

    # Extract fashion keywords
    cap_str = str(post_info.get('caption', '')) if post_info.get('caption') else ''
    img_str = str(post_info.get('image_description', '')) if post_info.get('image_description') else ''
    keywords = data_loader.extract_fashion_keywords(cap_str + ' ' + img_str)

    match_text, match_class = get_match_level(score, max_score)

    with st.container():
        col1, col2 = st.columns([1, 4])

        with col1:
            image = get_post_image(post_info.get("post_id"))

            if image:
                st.image(
                    image,
                    use_container_width=True
                )
            else:
                st.markdown(
                    "<div style='font-size:3rem; text-align:center;'>👗</div>",
                    unsafe_allow_html=True
                )

            st.markdown(
                f"<div class='{match_class}' style='text-align:center; margin-top:6px;'>{match_text}</div>",
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(f"### Post by @{author}")
            st.write(f"Caption:📝 {caption if caption else 'No caption available'}")

            # Fashion tags
            if keywords:
                tags_html = " ".join([f'<span class="tag">{kw}</span>' for kw in keywords[:6]])
                st.markdown(f" Hashtags:🏷️ {tags_html}", unsafe_allow_html=True)

            # Stats
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("❤️ Likes", f"{likes:,}")
            with col_b:
                st.metric("💬 Comments", f"{comments:,}")
            with col_c:
                st.metric("🎯 Match Score", f"{score:.1f}")

        st.markdown("---")


def render_history_item(post_info, polarity):
    author = safe_str(post_info.get('postUser', 'Unknown'), 20)
    caption = safe_str(post_info.get('caption', ''), 90)
    post_id = post_info.get('post_id')

    image = get_post_image(post_id)

    # Sentiment
    if polarity > 0.7:
        sentiment_emoji = "😊"
        sentiment_text = "Positive"
    elif polarity > 0.3:
        sentiment_emoji = "😐"
        sentiment_text = "Neutral"
    else:
        sentiment_emoji = "😞"
        sentiment_text = "Negative"

    img_html = ""
    if image:
        img_html = f"""
        <img src="data:image/jpeg;base64,{st.image(image, output_format='JPEG')}" />
        """

    # Streamlit không cho embed image trực tiếp trong HTML
    # => ta render bằng layout columns
    col_img, col_text = st.columns([1, 3])

    with col_img:
        if image:
            st.image(image, width=60)
        else:
            st.markdown("👗")

    with col_text:
        st.markdown(f"""
        <div class="history-author">@{author}</div>
        <div class="history-caption">{caption}</div>
        <div class="history-sentiment">
            {sentiment_emoji} {sentiment_text} interaction
        </div>
        """, unsafe_allow_html=True)



def main():
    # Header
    st.markdown('<div class="main-title">👗 Personalized Fashion Recommendation Platform</div>', unsafe_allow_html=True)

    # Load data and model
    data_loader = load_data()

    if data_loader is None or data_loader.post2idx is None:
        st.error("⚠️ Unable to load recommendation system. Please try again later.")
        st.stop()

    embeddings_shape = data_loader.embeddings.shape
    vocab_size = len(data_loader.post2idx)
    model_loader = load_model(embeddings_shape, vocab_size)

    if model_loader is None:
        st.error("⚠️ Unable to load recommendation model. Please try again later.")
        st.stop()

    # User selection (simulating login)
    st.markdown('<div class="section-title">👤 Welcome! Select your profile</div>', unsafe_allow_html=True)

    users = data_loader.get_user_list()

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_user = st.selectbox(
            "Choose your account",
            users,
            index=0,
            label_visibility="collapsed"
        )
    with col2:
        num_recommendations = st.selectbox(
            "Show recommendations",
            [5, 10, 15, 20],
            index=1,
            label_visibility="collapsed"
        )

    # Get user data
    posts, interactions = data_loader.get_user_sequence(selected_user)

    if not posts:
        st.warning("🛒 Start browsing to get personalized recommendations!")
        st.stop()

    # Layout: Recommendations (main) + History (sidebar)
    main_col, side_col = st.columns([3, 1])

    with side_col:
        st.markdown('<div class="section-title">📜 Your History</div>', unsafe_allow_html=True)

        # Show last 5 interactions
        recent_posts = posts[-5:] if len(posts) > 5 else posts
        recent_interactions = interactions[-5:] if len(interactions) > 5 else interactions

        for post_id, interaction in zip(reversed(recent_posts), reversed(recent_interactions)):
            post_info = data_loader.get_post_info(post_id)
            polarity = interaction.get('polarity', 0.5)
            render_history_item(post_info, polarity)

        # Stats
        st.markdown('<div class="section-title">📊 Your Stats</div>', unsafe_allow_html=True)
        avg_polarity = np.mean([i.get('polarity', 0.5) for i in interactions])

        st.markdown(f"""
        <div class="stats-card">
            <div class="emoji-large">{"😊" if avg_polarity > 0.5 else "😐"}</div>
            <div><strong>{len(posts)}</strong> items viewed</div>
            <div>Sentiment: <strong>{avg_polarity:.0%}</strong> positive</div>
        </div>
        """, unsafe_allow_html=True)

    with main_col:
        st.markdown('<div class="section-title">✨ Recommended For You</div>', unsafe_allow_html=True)

        # Get recommendations
        with st.spinner("Finding the best items for you..."):
            recommendations = model_loader.predict(posts, top_k=num_recommendations, use_hybrid=True)

        if not recommendations:
            st.info("🔍 We're still learning your style. Keep browsing!")
            st.stop()

        # Display recommendations
        max_score = recommendations[0]['score'] if recommendations else 1

        for i, rec in enumerate(recommendations):
            post_info = data_loader.get_post_info(rec['post_id'])
            render_product_card(post_info, rec['score'], i + 1, max_score, data_loader)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #999; padding: 1rem;">
        <small>🤖 Powered by Hybrid Neural + Collaborative Filtering | Built for Fashion Lovers</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
