"""
Streamlit demo for sequential recommendation system.

This demo provides an interactive interface for users to:
1. View recommendations for different users
2. Search for similar items
3. Explore model performance metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.sequential_models import MarkovChainModel, GRU4RecModel, Config, set_random_seeds
from src.data.data_pipeline import DataLoader, SequenceProcessor, DataConfig
from src.utils.evaluation import SequentialEvaluator

# Set random seeds
set_random_seeds(42)

# Page configuration
st.set_page_config(
    page_title="Sequential Recommendation System",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-item {
        background-color: #e8f4fd;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load data with caching."""
    try:
        loader = DataLoader("data")
        interactions_df = loader.load_interactions()
        items_df = loader.load_items()
        users_df = loader.load_users()
        
        # Process sequences
        processor = SequenceProcessor()
        sequences = processor.interactions_to_sequences(interactions_df)
        
        return interactions_df, items_df, users_df, sequences
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None


@st.cache_resource
def load_models():
    """Load trained models with caching."""
    try:
        # Load data to determine number of items
        loader = DataLoader("data")
        interactions_df = loader.load_interactions()
        processor = SequenceProcessor()
        sequences = processor.interactions_to_sequences(interactions_df)
        
        # Determine number of unique items
        all_items = set()
        for seq in sequences:
            all_items.update(seq)
        num_items = len(all_items)
        
        # Initialize models
        models = {
            'Markov Chain': MarkovChainModel(order=1, smoothing=0.1),
            'GRU4Rec': GRU4RecModel(num_items=num_items, epochs=10, device="cpu")
        }
        
        # Train models (simplified for demo)
        train_sequences = sequences[:int(len(sequences) * 0.8)]
        for model in models.values():
            model.fit(train_sequences)
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔄 Sequential Recommendation System</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        interactions_df, items_df, users_df, sequences = load_data()
    
    if interactions_df is None:
        st.error("Failed to load data. Please ensure data files exist in the 'data' directory.")
        return
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
    
    if models is None:
        st.error("Failed to load models.")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["User Recommendations", "Item Search", "Model Performance", "Data Overview"]
    )
    
    if page == "User Recommendations":
        show_user_recommendations(interactions_df, items_df, users_df, sequences, models)
    elif page == "Item Search":
        show_item_search(items_df, models)
    elif page == "Model Performance":
        show_model_performance(sequences, models)
    elif page == "Data Overview":
        show_data_overview(interactions_df, items_df, users_df)


def show_user_recommendations(interactions_df, items_df, users_df, sequences, models):
    """Show user recommendations page."""
    st.header("👤 User Recommendations")
    
    # User selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        user_id = st.selectbox(
            "Select a user:",
            options=sorted(interactions_df['user_id'].unique()),
            format_func=lambda x: f"User {x}"
        )
    
    with col2:
        num_recommendations = st.slider("Number of recommendations:", 1, 20, 10)
    
    # Get user's interaction history
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    user_sequence = user_interactions.sort_values('timestamp')['item_id'].tolist()
    
    if len(user_sequence) < 2:
        st.warning("This user has insufficient interaction history for recommendations.")
        return
    
    # Display user's history
    st.subheader("User's Interaction History")
    history_items = []
    for item_id in user_sequence[-10:]:  # Show last 10 items
        item_info = items_df[items_df['item_id'] == item_id].iloc[0]
        history_items.append(f"{item_info['title']} ({item_info['category']})")
    
    st.write(" → ".join(history_items))
    
    # Get recommendations from each model
    st.subheader("Recommendations")
    
    input_sequence = user_sequence[:-1]  # Use all but last item for prediction
    
    for model_name, model in models.items():
        with st.expander(f"{model_name} Recommendations", expanded=True):
            try:
                recommendations = model.predict_next(input_sequence, k=num_recommendations)
                
                # Display recommendations
                for i, item_id in enumerate(recommendations, 1):
                    item_info = items_df[items_df['item_id'] == item_id].iloc[0]
                    
                    col1, col2, col3 = st.columns([1, 3, 1])
                    
                    with col1:
                        st.write(f"**#{i}**")
                    
                    with col2:
                        st.markdown(f"""
                        <div class="recommendation-item">
                            <strong>{item_info['title']}</strong><br>
                            Category: {item_info['category']}<br>
                            Price: ${item_info['price']} | Rating: {item_info['rating']}⭐
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        if st.button(f"View", key=f"{model_name}_{i}"):
                            st.session_state.selected_item = item_id
            
            except Exception as e:
                st.error(f"Error getting recommendations from {model_name}: {e}")


def show_item_search(items_df, models):
    """Show item search page."""
    st.header("🔍 Item Search & Similarity")
    
    # Item search
    search_term = st.text_input("Search for items:", placeholder="Enter item name or category...")
    
    if search_term:
        # Filter items based on search term
        filtered_items = items_df[
            items_df['title'].str.contains(search_term, case=False) |
            items_df['category'].str.contains(search_term, case=False)
        ]
        
        if len(filtered_items) > 0:
            selected_item_id = st.selectbox(
                "Select an item:",
                options=filtered_items['item_id'].tolist(),
                format_func=lambda x: f"{items_df[items_df['item_id'] == x].iloc[0]['title']} ({items_df[items_df['item_id'] == x].iloc[0]['category']})"
            )
            
            # Display item details
            item_info = items_df[items_df['item_id'] == selected_item_id].iloc[0]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"📦 {item_info['title']}")
                st.write(f"**Category:** {item_info['category']}")
                st.write(f"**Price:** ${item_info['price']}")
                st.write(f"**Rating:** {item_info['rating']}⭐")
                st.write(f"**Features:** {item_info['features']}")
            
            with col2:
                st.metric("Item ID", item_info['item_id'])
            
            # Find similar items (simplified approach)
            st.subheader("🔗 Similar Items")
            
            # Get items from same category
            similar_items = items_df[
                (items_df['category'] == item_info['category']) &
                (items_df['item_id'] != selected_item_id)
            ].head(5)
            
            for _, similar_item in similar_items.iterrows():
                st.write(f"• {similar_item['title']} (${similar_item['price']})")
        
        else:
            st.warning("No items found matching your search.")
    
    else:
        # Show random items
        st.subheader("🎲 Random Items")
        random_items = items_df.sample(n=min(5, len(items_df)))
        
        for _, item in random_items.iterrows():
            st.write(f"• {item['title']} ({item['category']}) - ${item['price']}")


def show_model_performance(sequences, models):
    """Show model performance page."""
    st.header("📊 Model Performance")
    
    # Split sequences for evaluation
    train_size = int(len(sequences) * 0.7)
    test_sequences = sequences[train_size:]
    
    if len(test_sequences) == 0:
        st.warning("No test sequences available for evaluation.")
        return
    
    # Evaluate models
    evaluator = SequentialEvaluator(k_values=[5, 10])
    
    with st.spinner("Evaluating models..."):
        try:
            results_df = evaluator.compare_models(models, test_sequences)
            
            # Display results
            st.subheader("Evaluation Results")
            st.dataframe(results_df, use_container_width=True)
            
            # Create performance charts
            st.subheader("Performance Comparison")
            
            # Select metrics to display
            metric_options = [col for col in results_df.columns if '@' in col]
            selected_metrics = st.multiselect(
                "Select metrics to display:",
                options=metric_options,
                default=metric_options[:3] if len(metric_options) >= 3 else metric_options
            )
            
            if selected_metrics:
                # Create bar chart
                fig = go.Figure()
                
                for metric in selected_metrics:
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=results_df.index,
                        y=results_df[metric],
                        text=results_df[metric].round(3),
                        textposition='auto'
                    ))
                
                fig.update_layout(
                    title="Model Performance Comparison",
                    xaxis_title="Models",
                    yaxis_title="Score",
                    barmode='group',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Model comparison table
            st.subheader("Model Comparison Summary")
            
            comparison_data = []
            for model_name in results_df.index:
                best_metrics = []
                for metric in results_df.columns:
                    if results_df.loc[model_name, metric] == results_df[metric].max():
                        best_metrics.append(metric)
                
                comparison_data.append({
                    'Model': model_name,
                    'Best Metrics': ', '.join(best_metrics) if best_metrics else 'None',
                    'Avg Score': results_df.loc[model_name].mean()
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('Avg Score', ascending=False)
            st.dataframe(comparison_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error evaluating models: {e}")


def show_data_overview(interactions_df, items_df, users_df):
    """Show data overview page."""
    st.header("📈 Data Overview")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", len(users_df))
    
    with col2:
        st.metric("Total Items", len(items_df))
    
    with col3:
        st.metric("Total Interactions", len(interactions_df))
    
    with col4:
        avg_interactions = len(interactions_df) / len(users_df) if len(users_df) > 0 else 0
        st.metric("Avg Interactions/User", f"{avg_interactions:.1f}")
    
    # Data visualizations
    st.subheader("Data Distribution")
    
    # Interactions over time
    if 'timestamp' in interactions_df.columns:
        interactions_df['date'] = pd.to_datetime(interactions_df['timestamp'], unit='s')
        daily_interactions = interactions_df.groupby(interactions_df['date'].dt.date).size()
        
        fig = px.line(
            x=daily_interactions.index,
            y=daily_interactions.values,
            title="Daily Interactions",
            labels={'x': 'Date', 'y': 'Number of Interactions'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Item category distribution
    if 'category' in items_df.columns:
        category_counts = items_df['category'].value_counts()
        
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Item Category Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # User demographics
    if 'age' in users_df.columns and 'gender' in users_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                users_df,
                x='age',
                title="User Age Distribution",
                nbins=20
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            gender_counts = users_df['gender'].value_counts()
            fig = px.bar(
                x=gender_counts.index,
                y=gender_counts.values,
                title="User Gender Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Sample data tables
    st.subheader("Sample Data")
    
    tab1, tab2, tab3 = st.tabs(["Interactions", "Items", "Users"])
    
    with tab1:
        st.dataframe(interactions_df.head(10), use_container_width=True)
    
    with tab2:
        st.dataframe(items_df.head(10), use_container_width=True)
    
    with tab3:
        st.dataframe(users_df.head(10), use_container_width=True)


if __name__ == "__main__":
    main()
