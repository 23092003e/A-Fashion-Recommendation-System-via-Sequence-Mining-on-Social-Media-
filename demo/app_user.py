from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import base64
from io import BytesIO
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

IMAGE_DIR = Path(
    r"C:\Users\ADMIN\Desktop\ITDSIU21099_HoangVanManh\Fashion-Marketing-Automation-Solutions\images\original_images"
)

from utils.data_loader import DataLoader
from utils.model_loader import ModelLoader


# ==================== CONFIGURATION ====================
PAGE_CONFIG = {
    "page_title": "Fashion Recommendation System",
    "page_icon": "👗",
    "layout": "wide",
    "initial_sidebar_state": "collapsed"
}

DEFAULT_NUM_RECOMMENDATIONS = 10
HISTORY_DISPLAY_COUNT = 5


# ==================== SESSION STATE INITIALIZATION ====================
def init_session_state():
    """Initialize session state variables"""
    if 'user_interactions' not in st.session_state:
        st.session_state.user_interactions = {}
    if 'user_posts_viewed' not in st.session_state:
        st.session_state.user_posts_viewed = {}
    if 'refresh_recommendations' not in st.session_state:
        st.session_state.refresh_recommendations = False


def add_interaction(user, post_id, interaction_type, comment_text=None):
    """Add a user interaction to session state"""
    if user not in st.session_state.user_interactions:
        st.session_state.user_interactions[user] = []
    if user not in st.session_state.user_posts_viewed:
        st.session_state.user_posts_viewed[user] = []
    
    # Add post to viewed list if not already there
    if post_id not in st.session_state.user_posts_viewed[user]:
        st.session_state.user_posts_viewed[user].append(post_id)
    
    # Calculate polarity based on interaction type
    polarity = 0.8 if interaction_type == 'like' else 0.6
    
    # Add interaction
    interaction = {
        'post_id': post_id,
        'type': interaction_type,
        'polarity': polarity,
        'timestamp': datetime.now(),
        'comment': comment_text
    }
    
    st.session_state.user_interactions[user].append(interaction)
    st.session_state.refresh_recommendations = True


def get_user_session_data(user, original_posts, original_interactions):
    """Combine original data with session interactions"""
    session_posts = st.session_state.user_posts_viewed.get(user, [])
    session_interactions = st.session_state.user_interactions.get(user, [])
    
    # Combine lists
    all_posts = list(original_posts) + [i['post_id'] for i in session_interactions]
    all_interactions = list(original_interactions) + session_interactions
    
    return all_posts, all_interactions


# ==================== UTILITY FUNCTIONS ====================
@st.cache_data
def get_post_image(post_id):
    """Load and cache post image"""
    try:
        img_path = IMAGE_DIR / f"{post_id}.jpg"
        if img_path.exists():
            return Image.open(img_path)
    except Exception as e:
        st.warning(f"Error loading image {post_id}: {str(e)}")
    return None


def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    try:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()
    except:
        return None


def get_base_path():
    """Get base path of the project"""
    return Path(__file__).resolve().parent.parent


@st.cache_resource
def load_data():
    """Load and cache data"""
    base_path = get_base_path()
    data_loader = DataLoader(base_path)
    success = data_loader.load_all_data()
    return data_loader if success else None


@st.cache_resource
def load_model(embeddings_shape, vocab_size):
    """Load and cache model"""
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
    """Safely convert value to string with truncation"""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ''
    s = str(val)
    return s[:max_len] + '...' if len(s) > max_len else s


def get_match_level(score, max_score):
    """Determine match level based on score ratio"""
    ratio = score / max_score if max_score > 0 else 0
    
    if ratio > 0.8:
        return "Perfect Match! 🎯", "match-high"
    elif ratio > 0.5:
        return "Great Match! ✨", "match-medium"
    else:
        return "You May Like 👍", "match-low"


def get_sentiment_info(polarity):
    """Get sentiment emoji and text based on polarity"""
    if polarity > 0.7:
        return "😊", "Positive", "#10b981"
    elif polarity > 0.3:
        return "😐", "Neutral", "#6b7280"
    else:
        return "😞", "Negative", "#ef4444"


def get_interaction_type_display(interaction):
    """Get display info for interaction type"""
    itype = interaction.get('type', 'view')
    if itype == 'like':
        return "❤️", "Like"
    elif itype == 'comment':
        return "💬", "Comment"
    else:
        return "💬", "Comment"


# ==================== STYLING ====================
def inject_custom_css():
    """Inject optimized custom CSS"""
    st.markdown("""
    <style>
        /* Global Styles */
        .main { padding: 0 2rem; }
        
        /* Header Styles */
        .main-title {
            font-size: clamp(2rem, 5vw, 3rem);
            font-weight: 700;
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 2rem 0 1rem;
        }
        
        /* Section Titles */
        .section-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #ffffff;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 1.5rem 0 1rem;
            padding: 0.8rem 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        }
        
        /* Match Badges */
        .match-badge {
            color: white;
            padding: 0.4rem 1rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            text-align: center;
            margin-top: 0.5rem;
            display: inline-block;
        }
        
        .match-high { 
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        }
        
        .match-medium { 
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        }
        
        .match-low { 
            background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        }
        
        /* Tags */
        .tag {
            display: inline-block;
            background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
            color: #4338ca;
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            font-size: 0.8rem;
            margin: 0.2rem;
            font-weight: 500;
        }
        
        /* Interaction Buttons Container */
        .interaction-container {
            display: flex;
            gap: 0.5rem;
            margin-top: 1rem;
            align-items: center;
        }
        
        /* History Items */
        .history-item {
            background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
            border-radius: 12px;
            padding: 1rem;
            margin: 0.8rem 0;
            border-left: 4px solid #667eea;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
        }
        
        .history-item:hover {
            transform: translateX(5px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.4);
        }
        
        .history-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        }
        
        .history-author {
            font-weight: 700;
            color: #f3f4f6;
            font-size: 0.95rem;
        }
        
        .history-caption {
            color: #d1d5db;
            font-size: 0.85rem;
            line-height: 1.5;
            margin: 0.5rem 0;
        }
        
        .history-sentiment {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.2rem 0.6rem;
            border-radius: 10px;
            background: rgba(255,255,255,0.1);
        }
        
        .history-interaction-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            font-size: 0.7rem;
            font-weight: 600;
            padding: 0.2rem 0.5rem;
            border-radius: 8px;
            background: rgba(102, 126, 234, 0.3);
            color: #c7d2fe;
            margin-top: 0.3rem;
        }
        
        .history-image-container {
            width: 70px;
            height: 70px;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 0.8rem;
            border: 2px solid rgba(255,255,255,0.1);
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        
        .history-image-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        /* Stats Card */
        .stats-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            color: white;
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
            margin-top: 1rem;
        }
        
        .stats-card .emoji-large {
            font-size: 3rem;
            margin-bottom: 0.5rem;
        }
        
        .stats-card div {
            margin: 0.3rem 0;
            font-size: 0.95rem;
        }
        
        /* Product Card Enhancements */
        .product-image-container {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            margin-bottom: 1rem;
        }
        
        .product-image-container img {
            transition: transform 0.3s ease;
        }
        
        .product-image-container:hover img {
            transform: scale(1.05);
        }
        
        /* Success Message */
        .success-message {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 0.8rem 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            font-size: 0.9rem;
            font-weight: 500;
            text-align: center;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .main { padding: 0 1rem; }
            .main-title { font-size: 2rem; }
            .section-title { font-size: 1.2rem; }
        }
    </style>
    """, unsafe_allow_html=True)


# ==================== RENDERING FUNCTIONS ====================
def render_product_card(post_info, score, rank, max_score, data_loader, selected_user):
    """Render an optimized product recommendation card with interaction buttons"""
    
    # Extract data
    author = safe_str(post_info.get('postUser', 'Unknown'), 30)
    caption = safe_str(post_info.get('caption', ''), 200)
    likes = post_info.get('likesCount', 0)
    comments = post_info.get('commentsCount', 0)
    post_id = post_info.get('post_id')
    
    # Get keywords
    cap_str = str(post_info.get('caption', '')) if post_info.get('caption') else ''
    img_str = str(post_info.get('image_description', '')) if post_info.get('image_description') else ''
    keywords = data_loader.extract_fashion_keywords(cap_str + ' ' + img_str)
    
    # Get match level
    match_text, match_class = get_match_level(score, max_score)
    
    # Render
    col1, col2 = st.columns([1, 3])
    
    with col1:
        image = get_post_image(post_id)
        
        if image:
            st.markdown('<div class="product-image-container">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='font-size:4rem; text-align:center; padding:2rem; background:#f3f4f6; border-radius:12px;'>👗</div>",
                unsafe_allow_html=True
            )
        
        st.markdown(
            f'<div class="match-badge {match_class}" style="width: 100%;">{match_text}</div>',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(f"### 👤 @{author}")
        
        if caption:
            st.write(f"**📝 Caption:** {caption}")
        else:
            st.write("📝 *No caption available*")
        
        # Fashion tags
        if keywords:
            tags_html = " ".join([f'<span class="tag">#{kw}</span>' for kw in keywords[:8]])
            st.markdown(f"**🏷️ Tags:** {tags_html}", unsafe_allow_html=True)
        
        # Stats in columns
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("❤️ Likes", f"{likes:,}")
        with col_b:
            st.metric("💬 Comments", f"{comments:,}")
        with col_c:
            st.metric("🎯 Score", f"{score:.1f}")
        
        # Interaction section
        st.markdown("---")
        st.markdown("**🎬 Interact with this post:**")
        
        int_col1, int_col2, int_col3 = st.columns([1, 1, 2])
        
        with int_col1:
            if st.button("❤️ Like", key=f"like_{post_id}", use_container_width=True):
                add_interaction(selected_user, post_id, 'like')
                st.success("✨ Liked! Added to your history")
                st.rerun()
        
        with int_col2:
            show_comment = st.button("💬 Comment", key=f"comment_{post_id}", use_container_width=True)
        
        # Comment input (shown when button is clicked)
        if show_comment or f"show_comment_{post_id}" in st.session_state:
            st.session_state[f"show_comment_{post_id}"] = True
            
            with int_col3:
                pass  # Space holder
            
            comment_text = st.text_input(
                "Your comment:",
                key=f"comment_input_{post_id}",
                placeholder="Share your thoughts..."
            )
            
            if st.button("📤 Submit", key=f"submit_comment_{post_id}"):
                if comment_text.strip():
                    add_interaction(selected_user, post_id, 'comment', comment_text)
                    st.success("✨ Comment added! Added to your history")
                    del st.session_state[f"show_comment_{post_id}"]
                    st.rerun()
                else:
                    st.warning("Please enter a comment first")
    
    st.divider()


def render_history_item(post_info, interaction):
    """Render an optimized history item with interaction info"""
    
    author = safe_str(post_info.get('postUser', 'Unknown'), 25)
    caption = safe_str(post_info.get('caption', ''), 100)
    post_id = post_info.get('post_id')
    
    # Get polarity and interaction type
    polarity = interaction.get('polarity', 0.5)
    emoji_int, type_text = get_interaction_type_display(interaction)
    
    # Get sentiment
    emoji, sentiment_text, color = get_sentiment_info(polarity)
    
    # Get image
    image = get_post_image(post_id)
    
    # Create history item HTML
    st.markdown('<div class="history-item">', unsafe_allow_html=True)
    
    # Image section
    if image:
        img_base64 = image_to_base64(image)
        if img_base64:
            st.markdown(
                f'<div class="history-image-container"><img src="data:image/jpeg;base64,{img_base64}" alt="Post"></div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<div style="font-size:2rem; text-align:center; padding:0.5rem;">👗</div>',
            unsafe_allow_html=True
        )
    
    # Content
    st.markdown(
        f'<div class="history-header"><span class="history-author">@{author}</span></div>',
        unsafe_allow_html=True
    )
    
    if caption:
        st.markdown(
            f'<div class="history-caption">{caption}</div>',
            unsafe_allow_html=True
        )
    
    # Show comment if available
    if interaction.get('comment'):
        st.markdown(
            f'<div class="history-caption" style="font-style: italic; border-left: 2px solid #667eea; padding-left: 0.5rem;">"{interaction.get("comment")}"</div>',
            unsafe_allow_html=True
        )
    
    # Interaction badge and sentiment
    st.markdown(
        f'<div class="history-interaction-badge">{emoji_int} {type_text}</div>',
        unsafe_allow_html=True
    )
    
    st.markdown(
        f'<div class="history-sentiment" style="color:{color}">{emoji} {sentiment_text}</div>',
        unsafe_allow_html=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)


# ==================== MAIN APPLICATION ====================
def main():
    # Configure page
    st.set_page_config(**PAGE_CONFIG)
    
    # Initialize session state
    init_session_state()
    
    # Inject CSS
    inject_custom_css()
    
    # Header
    st.markdown(
        '<div class="main-title">👗 Personalized Fashion Recommendation Platform</div>',
        unsafe_allow_html=True
    )
    
    # Load data and model
    data_loader = load_data()
    
    if data_loader is None or data_loader.post2idx is None:
        st.error("⚠️ Unable to load recommendation system. Please check data files.")
        st.stop()
    
    # Load model
    embeddings_shape = data_loader.embeddings.shape
    vocab_size = len(data_loader.post2idx)
    model_loader = load_model(embeddings_shape, vocab_size)
    
    if model_loader is None:
        st.error("⚠️ Unable to load recommendation model. Please check model files.")
        st.stop()
    
    # User selection
    st.markdown(
        '<div class="section-title">👤 Select Your Profile</div>',
        unsafe_allow_html=True
    )
    
    users = data_loader.get_user_list()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_user = st.selectbox(
            "Choose account",
            users,
            index=0,
            label_visibility="collapsed"
        )
    with col2:
        num_recommendations = st.selectbox(
            "Number of recommendations",
            [5, 10, 15, 20],
            index=1,
            label_visibility="collapsed"
        )
    
    # Get original user data
    original_posts, original_interactions = data_loader.get_user_sequence(selected_user)
    
    # Combine with session data
    all_posts, all_interactions = get_user_session_data(
        selected_user, 
        original_posts, 
        original_interactions
    )
    
    if not all_posts:
        st.warning("🛒 Start browsing to get personalized recommendations!")
        st.stop()
    
    # Main layout
    col_main, col_side = st.columns([3, 1])
    
    # Sidebar: History & Stats
    with col_side:
        st.markdown(
            '<div class="section-title">📜 Your History</div>',
            unsafe_allow_html=True
        )
        
        # Show recent interactions
        recent_count = min(HISTORY_DISPLAY_COUNT, len(all_posts))
        recent_posts = all_posts[-recent_count:]
        recent_interactions = all_interactions[-recent_count:]
        
        for post_id, interaction in zip(reversed(recent_posts), reversed(recent_interactions)):
            # Handle both dict and post_id formats
            if isinstance(interaction, dict):
                actual_post_id = interaction.get('post_id', post_id)
            else:
                actual_post_id = post_id
                interaction = {'polarity': 0.5, 'type': 'view'}
            
            post_info = data_loader.get_post_info(actual_post_id)
            render_history_item(post_info, interaction)
        
        # User stats
        st.markdown(
            '<div class="section-title">📊 Your Stats</div>',
            unsafe_allow_html=True
        )
        
        # Calculate stats
        polarities = []
        for i in all_interactions:
            if isinstance(i, dict):
                polarities.append(i.get('polarity', 0.5))
            else:
                polarities.append(0.5)
        
        avg_polarity = np.mean(polarities) if polarities else 0.5
        sentiment_emoji = "😊" if avg_polarity > 0.5 else "😐" if avg_polarity > 0.3 else "😞"
        
        # Count interaction types
        session_interactions = st.session_state.user_interactions.get(selected_user, [])
        likes_count = sum(1 for i in session_interactions if i.get('type') == 'like')
        comments_count = sum(1 for i in session_interactions if i.get('type') == 'comment')
        
        st.markdown(f"""
        <div class="stats-card">
            <div class="emoji-large">{sentiment_emoji}</div>
            <div><strong>{len(all_posts)}</strong> posts viewed</div>
            <div><strong>{likes_count}</strong> likes • <strong>{comments_count}</strong> comments</div>
            <div>Sentiment: <strong>{avg_polarity:.0%}</strong> positive</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content: Recommendations
    with col_main:
        st.markdown(
            '<div class="section-title">✨ Recommended For You</div>',
            unsafe_allow_html=True
        )
        
        # Get recommendations
        with st.spinner("🔍 Finding perfect matches for you..."):
            recommendations = model_loader.predict(
                all_posts,
                top_k=num_recommendations,
                use_hybrid=True
            )
        
        if not recommendations:
            st.info("🔍 Keep browsing! We're learning your style preferences.")
            st.stop()
        
        # Filter out already interacted posts
        viewed_posts = set(all_posts)
        recommendations = [r for r in recommendations if r['post_id'] not in viewed_posts]
        
        if not recommendations:
            st.info("🎉 You've seen all our top recommendations! Try adjusting the count or explore more.")
            st.stop()
        
        # Display recommendations
        max_score = recommendations[0]['score'] if recommendations else 1
        
        for i, rec in enumerate(recommendations):
            post_info = data_loader.get_post_info(rec['post_id'])
            render_product_card(
                post_info,
                rec['score'],
                i + 1,
                max_score,
                data_loader,
                selected_user
            )
    
    # Reset refresh flag
    if st.session_state.refresh_recommendations:
        st.session_state.refresh_recommendations = False


if __name__ == "__main__":
    main()