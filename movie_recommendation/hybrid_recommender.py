import pandas as pd
import numpy as np
import os

class HybridRecommender:
    def __init__(self, cf_weight=0.6, cb_weight=0.4, sentiment_weight=0.1):
        """
        Initialize the hybrid recommender
        cf_weight: weight for collaborative filtering recommendations
        cb_weight: weight for content-based filtering recommendations
        sentiment_weight: weight for sentiment analysis adjustment
        """
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.sentiment_weight = sentiment_weight
        
        self.movies_df = None
        self.cf_model = None
        self.cb_model = None
        self.sentiment_df = None
        
    def load_data(self, movies_df, cf_model, cb_model, sentiment_df=None):
        """
        Load all necessary data and models
        """
        self.movies_df = movies_df
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.sentiment_df = sentiment_df
        
    def get_recommendations(self, user_id, genre_preferences=None, n=5):
        """
        Get hybrid recommendations for a user
        user_id: user ID for collaborative filtering
        genre_preferences: dict of genres and their weights for content-based filtering
        n: number of recommendations to return
        """
        if self.movies_df is None or self.cf_model is None or self.cb_model is None:
            raise ValueError("Data and models must be loaded first")
            
        # Get list of all movie IDs
        all_movie_ids = self.movies_df['movieId'].values
        
        # Get collaborative filtering recommendations
        cf_recommendations = self.cf_model.get_top_n_recommendations(
            user_id, all_movie_ids, n=n*2)
        
        # Get content-based recommendations if genre preferences are provided
        if genre_preferences:
            cb_recommendations = self.cb_model.get_recommendations_by_genres(
                genre_preferences, n=n*2)
        else:
            # Use the user's highest-rated movies to get content-based recs
            # This requires access to ratings data
            # For simplicity, use the same as CF recommendations
            cb_recommendations = cf_recommendations
        
        # Combine recommendations with weights
        movie_scores = {}
        
        # Add collaborative filtering scores
        for i, movie_id in enumerate(cf_recommendations):
            score = self.cf_weight * (1.0 - i/(n*2))
            if movie_id in movie_scores:
                movie_scores[movie_id] += score
            else:
                movie_scores[movie_id] = score
                
        # Add content-based filtering scores
        for i, movie_id in enumerate(cb_recommendations):
            score = self.cb_weight * (1.0 - i/(n*2))
            if movie_id in movie_scores:
                movie_scores[movie_id] += score
            else:
                movie_scores[movie_id] = score
                
        # Apply sentiment adjustment if available
        if self.sentiment_df is not None and self.sentiment_weight > 0:
            for movie_id in list(movie_scores.keys()):
                sentiment_info = self.sentiment_df[self.sentiment_df['movieId'] == movie_id]
                if not sentiment_info.empty:
                    sentiment_score = sentiment_info.iloc[0]['sentiment_score']
                    # Apply sentiment adjustment
                    movie_scores[movie_id] += self.sentiment_weight * sentiment_score
        
        # Sort movies by score (descending) and get top N
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        
        # Get the final list of movie IDs
        recommended_movie_ids = [movie_id for movie_id, _ in sorted_movies]
        
        # Get movie details for the recommended movies
        recommendations = []
        for movie_id in recommended_movie_ids:
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
            recommendations.append({
                'movieId': movie_id,
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'year': movie_info['year'] if 'year' in movie_info else None,
                'avg_rating': movie_info['avg_rating'] if 'avg_rating' in movie_info else None,
            })
            
        return recommendations

if __name__ == "__main__":
    import os
    import pickle
    from data_preparation import prepare_data
    from collaborative_filtering import CollaborativeFiltering
    from content_based_filtering import ContentBasedFiltering
    from sentiment_analysis import SentimentAnalyzer
    
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
        
    # Create hybrid recommender
    hybrid_recommender = HybridRecommender(
        cf_weight=0.6, 
        cb_weight=0.4, 
        sentiment_weight=0.2 if sentiment_df is not None else 0
    )
    
    hybrid_recommender.load_data(movies, cf_model, cb_model, sentiment_df)
    
    # Test with a sample user and genre preferences
    sample_user_id = 1
    sample_genre_prefs = {
        'Action': 0.8,
        'Adventure': 0.6,
        'Sci-Fi': 0.7
    }
    
    recommendations = hybrid_recommender.get_recommendations(
        sample_user_id, 
        genre_preferences=sample_genre_prefs, 
        n=5
    )
    
    print(f"\nHybrid Recommendations for User {sample_user_id}:")
    for i, rec in enumerate(recommendations):
        print(f"{i+1}. {rec['title']} ({rec['year']}) - {rec['genres']}")