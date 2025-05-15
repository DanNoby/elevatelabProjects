import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from data_preparation import prepare_data
from collaborative_filtering import CollaborativeFiltering
from content_based_filtering import ContentBasedFiltering
from sentiment_analysis import SentimentAnalyzer
from hybrid_recommender import HybridRecommender

# Set page title and config
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

@st.cache_resource
def load_data_and_models():
    """
    Load data and models (cached for better performance)
    """
    # Load and prepare data
    movies, ratings, tags, genre_df, train_data, test_data = prepare_data()
    
    # Load or train collaborative filtering model
    cf_model_path = 'models/cf_model.pkl'
    if os.path.exists(cf_model_path):
        cf_model = CollaborativeFiltering()
        cf_model.load_model(cf_model_path)
    else:
        cf_model = CollaborativeFiltering(train_data)
        cf_model.train()
        cf_model.save_model()
        
    # Load or train content-based filtering model
    cb_model_path = 'models/cb_model.pkl'
    if os.path.exists(cb_model_path):
        cb_model = ContentBasedFiltering(movies, genre_df)
        cb_model.load_model(cb_model_path)
    else:
        cb_model = ContentBasedFiltering(movies, genre_df)
        cb_model.train()
        cb_model.save_model()
        
    # Generate sentiment data if tags are available
    sentiment_df = None
    if tags is not None:
        sentiment_analyzer = SentimentAnalyzer()
        sentiment_df = sentiment_analyzer.analyze_tags(tags)
    
    # Extract list of genres
    all_genres = []
    for genres in movies['genres'].str.split(','):
        if isinstance(genres, list):
            all_genres.extend(genres)
    unique_genres = sorted(list(set(all_genres)))
    if '(no genres listed)' in unique_genres:
        unique_genres.remove('(no genres listed)')
    
    return movies, ratings, tags, genre_df, cf_model, cb_model, sentiment_df, unique_genres

def main():
    # Title and description
    st.title("🎬 Movie Recommendation System")
    st.write("Get personalized movie recommendations based on your preferences")
    
    # Load data and models
    with st.spinner("Loading data and models..."):
        movies, ratings, tags, genre_df, cf_model, cb_model, sentiment_df, unique_genres = load_data_and_models()
    
    # Create hybrid recommender
    hybrid_recommender = HybridRecommender(
        cf_weight=0.6, 
        cb_weight=0.4, 
        sentiment_weight=0.2 if sentiment_df is not None else 0
    )
    
    hybrid_recommender.load_data(movies, cf_model, cb_model, sentiment_df)
    
    # Sidebar for user input
    st.sidebar.header("Your Preferences")
    
    # User selection
    st.sidebar.subheader("User ID")
    user_id = st.sidebar.number_input("Enter your user ID", min_value=1, max_value=int(ratings['userId'].max()), value=1)
    
    # Genre preferences
    st.sidebar.subheader("Genre Preferences")
    st.sidebar.write("Select your preferred genres and rate them from 0 to 1")
    
    genre_preferences = {}
    for genre in unique_genres:
        if st.sidebar.checkbox(genre, key=f"genre_{genre}"):
            weight = st.sidebar.slider(f"{genre} preference", 0.0, 1.0, 0.5, 0.1, key=f"weight_{genre}")
            genre_preferences[genre] = weight
    
    # Model weight adjustment
    st.sidebar.subheader("Model Weights")
    cf_weight = st.sidebar.slider("Collaborative Filtering Weight", 0.0, 1.0, 0.6, 0.1)
    cb_weight = st.sidebar.slider("Content-Based Filtering Weight", 0.0, 1.0, 0.4, 0.1)
    sentiment_weight = st.sidebar.slider("Sentiment Weight", 0.0, 1.0, 0.2, 0.1) if sentiment_df is not None else 0.0
    
    # Update model weights
    hybrid_recommender.cf_weight = cf_weight
    hybrid_recommender.cb_weight = cb_weight
    hybrid_recommender.sentiment_weight = sentiment_weight
    
    # Number of recommendations
    num_recommendations = st.sidebar.slider("Number of Recommendations", 1, 20, 5)
    
    # Submit button
    if st.sidebar.button("Get Recommendations"):
        with st.spinner("Generating recommendations..."):
            # Get recommendations
            recommendations = hybrid_recommender.get_recommendations(
                user_id, 
                genre_preferences=genre_preferences if genre_preferences else None, 
                n=num_recommendations
            )
            
            # Display recommendations
            st.subheader("Your Personalized Movie Recommendations")
            
            # Create columns for recommendations
            cols = st.columns(min(5, num_recommendations))
            
            for i, rec in enumerate(recommendations):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    st.write(f"**{i+1}. {rec['title']}**")
                    st.write(f"Genres: {rec['genres']}")
                    if 'year' in rec and rec['year'] > 0:
                        st.write(f"Year: {rec['year']}")
                    if 'avg_rating' in rec and rec['avg_rating'] > 0:
                        st.write(f"Average Rating: {rec['avg_rating']:.1f}/5.0")
                    st.write("---")
    
    # Display some popular movies if no recommendations generated yet
    else:
        st.subheader("Some Popular Movies")
        popular_movies = movies.sort_values('rating_count', ascending=False).head(10)
        
        cols = st.columns(5)
        for i, (_, movie) in enumerate(popular_movies.iterrows()):
            col_idx = i % 5
            with cols[col_idx]:
                st.write(f"**{movie['title']}**")
                st.write(f"Genres: {movie['genres']}")
                if 'year' in movie and movie['year'] > 0:
                    st.write(f"Year: {movie['year']}")
                if 'avg_rating' in movie and movie['avg_rating'] > 0:
                    st.write(f"Average Rating: {movie['avg_rating']:.1f}/5.0")
                if 'rating_count' in movie:
                    st.write(f"Ratings: {movie['rating_count']}")
                st.write("---")

if __name__ == "__main__":
    main()