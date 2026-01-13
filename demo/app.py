
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loader import DataLoader
from utils.model_loader import ModelLoader

# Page config
st.set_page_config(
    page_title="Fashion Recommendation System for Admin",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E5077;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .post-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }
    .score-badge {
        background: #667eea;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)


def get_base_path():
    """Get the base path for data files"""
    # Use absolute path to avoid issues with current working directory
    return Path(__file__).resolve().parent.parent


@st.cache_resource
def load_data():
    """Load data with caching"""
    base_path = get_base_path()
    data_loader = DataLoader(base_path)
    success = data_loader.load_all_data()
    if not success:
        return None
    return data_loader


@st.cache_resource
def load_model(embeddings_shape, vocab_size):
    """Load model with caching - uses shape/size for cache key"""
    # embeddings_shape and vocab_size are used as cache keys
    _ = (embeddings_shape, vocab_size)  # Acknowledge usage for cache invalidation

    base_path = get_base_path()

    # Reload data loader to get fresh data
    data_loader = DataLoader(base_path)
    data_loader.load_all_data()

    if data_loader.post2idx is None or data_loader.embeddings is None:
        return None

    model_loader = ModelLoader(base_path)
    model_loader.load_hybrid_model(data_loader.embeddings, data_loader.post2idx)
    return model_loader


def render_header():
    """Render main header"""
    st.markdown('<div class="main-header">👗 Fashion Recommendation System</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; color: #666; font-size: 1.1rem;">
        Hybrid Neural + Collaborative Filtering approach for personalized fashion recommendations
    </p>
    """, unsafe_allow_html=True)


def render_sidebar(data_loader):
    """Render sidebar with controls"""
    st.sidebar.title("⚙️ Settings")

    # User selection
    users = data_loader.get_user_list()
    selected_user = st.sidebar.selectbox(
        "Select User",
        users,
        index=0
    )

    # Number of recommendations
    top_k = st.sidebar.slider(
        "Number of Recommendations",
        min_value=3,
        max_value=20,
        value=10
    )

    # Model settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Settings")
    use_hybrid = st.sidebar.checkbox("Use Hybrid Model", value=True)

    # Display mode
    st.sidebar.markdown("---")
    st.sidebar.subheader("Display Options")
    show_details = st.sidebar.checkbox("Show Post Details", value=True)
    show_signals = st.sidebar.checkbox("Show Signal Analysis", value=True)

    return {
        'selected_user': selected_user,
        'top_k': top_k,
        'use_hybrid': use_hybrid,
        'show_details': show_details,
        'show_signals': show_signals
    }


def render_user_profile(data_loader, username):
    """Render user profile and interaction history"""
    st.markdown('<div class="sub-header">👤 User Profile</div>', unsafe_allow_html=True)

    posts, interactions = data_loader.get_user_sequence(username)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Username", username)
    with col2:
        st.metric("Interactions", len(posts))
    with col3:
        if interactions:
            avg_polarity = np.mean([i.get('polarity', 0.5) for i in interactions])
            sentiment = "Positive 😊" if avg_polarity > 0.5 else "Neutral 😐" if avg_polarity > 0 else "Negative 😞"
            st.metric("Avg Sentiment", sentiment)

    return posts, interactions


def render_interaction_history(data_loader, posts, interactions, show_details):
    """Render user's interaction history"""
    st.markdown('<div class="sub-header">📜 Interaction History</div>', unsafe_allow_html=True)

    if not posts:
        st.warning("No interaction history found for this user.")
        return

    # Helper function to safely get string
    def safe_str(val, max_len=50):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return 'N/A'
        s = str(val)
        return s[:max_len] + '...' if len(s) > max_len else s

    # Create timeline visualization
    if interactions:
        timeline_data = []
        for i, (post_id, interaction) in enumerate(zip(posts, interactions)):
            post_info = data_loader.get_post_info(post_id)
            timeline_data.append({
                'Order': i + 1,
                'Post ID': post_id,
                'Timestamp': interaction.get('timestamp', ''),
                'Polarity': interaction.get('polarity', 0.5),
                'Caption': safe_str(post_info.get('caption', ''), 50),
                'Author': safe_str(post_info.get('postUser', 'Unknown'), 50)
            })

        df_timeline = pd.DataFrame(timeline_data)

        # Display as expandable cards
        for i, row in df_timeline.iterrows():
            with st.expander(f"#{row['Order']} - Post {row['Post ID']} by @{row['Author']}", expanded=False):
                col1, col2 = st.columns([2, 1])
                with col1:
                    post_info = data_loader.get_post_info(row['Post ID'])
                    st.write(f"**Caption:** {safe_str(post_info.get('caption', 'N/A'), 200)}")
                    st.write(f"**Hashtags:** {safe_str(post_info.get('hashtags', 'N/A'), 100)}")

                    if show_details and post_info.get('image_description'):
                        st.write(f"**Image Description:** {safe_str(post_info.get('image_description', 'N/A'), 200)}")

                with col2:
                    st.metric("Likes", f"{post_info.get('likesCount', 0):,}")
                    st.metric("Comments", f"{post_info.get('commentsCount', 0):,}")
                    polarity = row['Polarity']
                    st.metric("Sentiment", f"{polarity:.2f}",
                             delta="Positive" if polarity > 0.5 else "Negative" if polarity < 0 else None)

        # Polarity chart
        fig = px.line(df_timeline, x='Order', y='Polarity',
                     title='Sentiment Trend Over Time',
                     markers=True,
                     color_discrete_sequence=['#667eea'])
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                     annotation_text="Neutral")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def render_recommendations(data_loader, model_loader, posts, settings):
    """Render recommendations"""
    st.markdown('<div class="sub-header">🎯 Personalized Recommendations</div>', unsafe_allow_html=True)

    if not posts:
        st.warning("Cannot generate recommendations without interaction history.")
        return

    # Get recommendations
    recommendations = model_loader.predict(posts, top_k=settings['top_k'], use_hybrid=settings['use_hybrid'])

    if not recommendations:
        st.error("Could not generate recommendations.")
        return

    # Display recommendations
    for i, rec in enumerate(recommendations):
        post_info = data_loader.get_post_info(rec['post_id'])

        with st.container():
            col1, col2, col3 = st.columns([1, 3, 1])

            with col1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                    <div style="font-size: 2rem; font-weight: bold;">#{i+1}</div>
                    <div style="font-size: 0.9rem;">Score: {rec['score']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                author = post_info.get('postUser', 'Unknown')
                if author is None or (isinstance(author, float) and pd.isna(author)):
                    author = 'Unknown'
                caption = post_info.get('caption', '')
                if caption is None or (isinstance(caption, float) and pd.isna(caption)):
                    caption = 'N/A'
                else:
                    caption = str(caption)[:150] + '...'

                st.markdown(f"**Post {rec['post_id']}** by @{author}")
                st.write(f"📝 {caption}")

                if settings['show_details']:
                    # Extract fashion keywords
                    cap_str = str(post_info.get('caption', '')) if post_info.get('caption') else ''
                    img_str = str(post_info.get('image_description', '')) if post_info.get('image_description') else ''
                    keywords = data_loader.extract_fashion_keywords(cap_str + ' ' + img_str)
                    if keywords:
                        st.write("🏷️ **Fashion Tags:** " + ", ".join(keywords[:8]))

            with col3:
                st.metric("❤️ Likes", f"{post_info.get('likesCount', 0):,}")
                st.metric("💬 Comments", f"{post_info.get('commentsCount', 0):,}")

            st.markdown("---")

    return recommendations


def render_signal_analysis(model_loader, posts, settings):
    """Render detailed signal analysis"""
    if not settings['show_signals']:
        return

    st.markdown('<div class="sub-header">📊 Signal Analysis</div>', unsafe_allow_html=True)

    analysis, weights = model_loader.get_signal_analysis(posts, top_k=settings['top_k'])

    if not analysis:
        return

    # Signal weights pie chart
    col1, col2 = st.columns([1, 2])

    with col1:
        fig_weights = px.pie(
            values=list(weights.values()),
            names=['Neural', 'Similarity', 'Co-occurrence', 'Popularity'],
            title='Signal Contribution Weights',
            color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#f5576c']
        )
        fig_weights.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_weights, use_container_width=True)

    with col2:
        # Stacked bar chart for top recommendations
        df_signals = pd.DataFrame(analysis)
        df_signals['post_label'] = df_signals['post_id'].apply(lambda x: f"Post {x}")

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name='Neural', x=df_signals['post_label'],
                                 y=df_signals['neural_score'], marker_color='#667eea'))
        fig_bar.add_trace(go.Bar(name='Similarity', x=df_signals['post_label'],
                                 y=df_signals['similarity_score'], marker_color='#764ba2'))
        fig_bar.add_trace(go.Bar(name='Co-occurrence', x=df_signals['post_label'],
                                 y=df_signals['cooccurrence_score'], marker_color='#f093fb'))
        fig_bar.add_trace(go.Bar(name='Popularity', x=df_signals['post_label'],
                                 y=df_signals['popularity_score'], marker_color='#f5576c'))

        fig_bar.update_layout(barmode='stack', title='Score Breakdown by Signal',
                             xaxis_title='Recommended Post', yaxis_title='Score')
        st.plotly_chart(fig_bar, use_container_width=True)


def render_similar_items_explorer(data_loader):
    """Render similar items explorer"""
    st.markdown('<div class="sub-header">🔍 Similar Items Explorer</div>', unsafe_allow_html=True)

    # Get all post IDs
    post_ids = list(data_loader.post2idx.keys())

    col1, col2 = st.columns([1, 2])

    with col1:
        selected_post = st.selectbox(
            "Select a post to find similar items",
            post_ids,
            format_func=lambda x: f"Post {x}"
        )

        num_similar = st.slider("Number of similar items", 3, 10, 5)

    if selected_post:
        # Get post info
        post_info = data_loader.get_post_info(selected_post)

        with col1:
            def safe_text(val, max_len=100):
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return 'N/A'
                return str(val)[:max_len] + '...' if len(str(val)) > max_len else str(val)

            st.markdown("**Selected Post:**")
            st.write(f"📝 {safe_text(post_info.get('caption', 'N/A'), 100)}")
            st.write(f"👤 @{safe_text(post_info.get('postUser', 'Unknown'), 50)}")

        # Find similar items
        similar = data_loader.get_similar_posts(selected_post, top_k=num_similar)

        with col2:
            st.markdown("**Similar Posts:**")
            for pid, sim_score in similar:
                sim_info = data_loader.get_post_info(pid)
                with st.expander(f"Post {pid} (Similarity: {sim_score:.3f})"):
                    author = sim_info.get('postUser', 'Unknown')
                    if author is None or (isinstance(author, float) and pd.isna(author)):
                        author = 'Unknown'
                    caption = sim_info.get('caption', 'N/A')
                    if caption is None or (isinstance(caption, float) and pd.isna(caption)):
                        caption = 'N/A'
                    else:
                        caption = str(caption)[:150] + '...'

                    st.write(f"**Author:** @{author}")
                    st.write(f"**Caption:** {caption}")

                    cap_str = str(sim_info.get('caption', '')) if sim_info.get('caption') else ''
                    img_str = str(sim_info.get('image_description', '')) if sim_info.get('image_description') else ''
                    keywords = data_loader.extract_fashion_keywords(cap_str + ' ' + img_str)
                    if keywords:
                        st.write("🏷️ " + ", ".join(keywords[:6]))


def render_model_comparison():
    """Render model comparison section"""
    st.markdown('<div class="sub-header">📈 Model Performance Comparison</div>', unsafe_allow_html=True)

    # Performance data from thesis
    models_data = {
        'Model': ['GRU4Rec', 'SASRec', 'BERT4Rec', 'Gated Fusion V2', 'Hybrid CF (Ours)'],
        'HR@5': [0.0231, 0.1489, 0.1870, 0.2576, 0.4681],
        'HR@10': [0.0381, 0.2121, 0.2381, 0.3177, 0.6416],
        'HR@20': [0.0606, 0.2987, 0.3420, 0.3939, 0.7612],
        'MRR': [0.0165, 0.1129, 0.1529, 0.1875, 0.3561]
    }

    df_models = pd.DataFrame(models_data)

    col1, col2 = st.columns(2)

    with col1:
        # HR@K comparison
        fig_hr = go.Figure()
        for metric in ['HR@5', 'HR@10', 'HR@20']:
            fig_hr.add_trace(go.Bar(name=metric, x=df_models['Model'], y=df_models[metric]))

        fig_hr.update_layout(
            title='Hit Rate Comparison',
            barmode='group',
            yaxis_title='Hit Rate',
            xaxis_title='Model'
        )
        st.plotly_chart(fig_hr, use_container_width=True)

    with col2:
        # MRR comparison
        fig_mrr = px.bar(df_models, x='Model', y='MRR',
                        title='Mean Reciprocal Rank Comparison',
                        color='MRR',
                        color_continuous_scale='Viridis')
        fig_mrr.update_layout(yaxis_title='MRR')
        st.plotly_chart(fig_mrr, use_container_width=True)

    # Improvement stats
    st.markdown("### 📊 Improvement Over Baselines")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("vs GRU4Rec", "+1584%", delta="HR@10 improvement")
    with col2:
        st.metric("vs BERT4Rec", "+169.5%", delta="HR@10 improvement")
    with col3:
        st.metric("vs Gated Fusion", "+102%", delta="HR@10 improvement")


def render_statistics(data_loader):
    """Render dataset statistics"""
    st.markdown('<div class="sub-header">📉 Dataset Statistics</div>', unsafe_allow_html=True)

    stats = data_loader.get_statistics()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Users", stats.get('num_users', 0))
    with col2:
        st.metric("Total Posts", stats.get('num_posts', 0))
    with col3:
        st.metric("Avg Sequence Length", f"{stats.get('avg_seq_length', 0):.1f}")
    with col4:
        st.metric("Embedding Dim", stats.get('embedding_dim', 0))

    # Sequence length distribution
    if data_loader.user_behavior is not None:
        col1, col2 = st.columns(2)

        with col1:
            fig_seq = px.histogram(
                data_loader.user_behavior,
                x='sequence_length',
                nbins=20,
                title='Sequence Length Distribution',
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(fig_seq, use_container_width=True)

        with col2:
            # Top influencers
            if data_loader.content is not None:
                top_users = data_loader.content.groupby('postUser').agg({
                    'likesCount': 'sum',
                    'commentsCount': 'sum'
                }).sort_values('likesCount', ascending=False).head(10)

                fig_top = px.bar(
                    top_users.reset_index(),
                    x='postUser',
                    y='likesCount',
                    title='Top 10 Influencers by Total Likes',
                    color_discrete_sequence=['#764ba2']
                )
                fig_top.update_layout(xaxis_title='User', yaxis_title='Total Likes')
                st.plotly_chart(fig_top, use_container_width=True)


def main():
    """Main application"""
    render_header()

    # Load data and model
    with st.spinner("Loading data and model..."):
        data_loader = load_data()

        if data_loader is None or data_loader.post2idx is None:
            st.error("Failed to load data. Please check that all required files exist.")
            st.info("Required files: input/user_behavior.csv, input/content.csv, post_embeddings_multimodal.npy")
            st.stop()

        # Pass hashable cache keys
        embeddings_shape = data_loader.embeddings.shape
        vocab_size = len(data_loader.post2idx)
        model_loader = load_model(embeddings_shape, vocab_size)

        if model_loader is None:
            st.error("Failed to load model.")
            st.stop()

    # Sidebar
    settings = render_sidebar(data_loader)

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Recommendations",
        "📈 Model Comparison",
        "🔍 Explore Similar",
        "📊 Statistics"
    ])

    with tab1:
        # User profile
        posts, interactions = render_user_profile(data_loader, settings['selected_user'])

        # Interaction history
        render_interaction_history(data_loader, posts, interactions, settings['show_details'])

        # Recommendations
        render_recommendations(data_loader, model_loader, posts, settings)

        # Signal analysis
        if posts:
            render_signal_analysis(model_loader, posts, settings)

    with tab2:
        render_model_comparison()

    with tab3:
        render_similar_items_explorer(data_loader)

    with tab4:
        render_statistics(data_loader)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🎓 Fashion Recommendation System - Thesis Project</p>
        <p>Hybrid Neural + Collaborative Filtering | HR@10: 0.6416 | MRR: 0.3561</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
