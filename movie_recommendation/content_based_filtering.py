import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class ContentBasedFiltering:
    def __init__(self, movies_df=None, genre_df=None):
        self.movies_df = movies_df
        self.genre_df = genre_df
        self.similarity_matrix = None
        self.feature_names = None
    
    def prepare_data(self, movies_df=None, genre_df=None):
        """
        Prepare data for content-based filtering
        """
        if movies_df is not None:
            self.movies_df = movies_df
        if genre_df is not None:
            self.genre_df = genre_df
            
        if self.movies_df is None or self.genre_df is None:
            raise ValueError("Movies dataframe and genre dataframe are required")
        
        # Create a copy of the genre dataframe without the movieId column
        genre_features = self.genre_df.drop(columns=['movieId'])
        
        # Store feature names
        self.feature_names = genre_features.columns.tolist()
        
        # Include movie year in features (normalized)
        if 'year' in self.movies_df.columns:
            year_min = self.movies_df['year'].min()
            year_max = self.movies_df['year'].max()
            normalized_year = (self.movies_df['year'] - year_min) / (year_max - year_min)
            genre_features['year'] = normalized_year.values
            self.feature_names.append('year')
            
        # Include average rating in features (normalized)
        if 'avg_rating' in self.movies_df.columns:
            rating_min = self.movies_df['avg_rating'].min()
            rating_max = self.movies_df['avg_rating'].max()
            if rating_max > rating_min:
                normalized_rating = (self.movies_df['avg_rating'] - rating_min) / (rating_max - rating_min)
                genre_features['avg_rating'] = normalized_rating.values
                self.feature_names.append('avg_rating')
                
        return genre_features
    
    def train(self, movies_df=None, genre_df=None):
        """
        Train the content-based filtering model
        """
        genre_features = self.prepare_data(movies_df, genre_df)
        
        print("Computing similarity matrix for content-based filtering...")
        self.similarity_matrix = cosine_similarity(genre_features)
        print("Similarity matrix computation complete!")
        
    def get_similar_movies(self, movie_idx, n=10):
        """
        Get similar movies for a given movie index
        """
        if self.similarity_matrix is None:
            raise ValueError("Model is not trained yet")
            
        # Get similarity scores for the movie
        movie_similarities = self.similarity_matrix[movie_idx]
        
        # Get the indices of the top N similar movies (excluding itself)
        similar_indices = np.argsort(movie_similarities)[::-1][1:n+1]
        
        # Get the movieIds for these indices
        similar_movie_ids = self.movies_df.iloc[similar_indices]['movieId'].tolist()
        similarity_scores = movie_similarities[similar_indices].tolist()
        
        return list(zip(similar_movie_ids, similarity_scores))
    
    def get_recommendations_by_genres(self, genre_preferences, n=5):
        """
        Get movie recommendations based on genre preferences
        genre_preferences: dict of genres and their weights
        """
        if self.similarity_matrix is None:
            raise ValueError("Model is not trained yet")
            
        # If feature_names is not available, extract it from the genre_df
        if self.feature_names is None:
            # Generate feature names from genre_df
            self.feature_names = [col for col in self.genre_df.columns if col != 'movieId']
            
            # Add additional features if they exist in the movies_df
            if 'year' in self.movies_df.columns:
                self.feature_names.append('year')
            if 'avg_rating' in self.movies_df.columns:
                self.feature_names.append('avg_rating')
        
        # Get a list of genres from genre_df (excluding movieId)
        genre_columns = [col for col in self.genre_df.columns if col != 'movieId']
        
        # Create a user profile vector based on genre preferences
        genre_features = self.prepare_data()
        user_profile = np.zeros(genre_features.shape[1])
        
        # Set weights for each genre that exists in both genre_preferences and genre_columns
        for i, col in enumerate(genre_features.columns):
            if col in genre_preferences:
                user_profile[i] = genre_preferences[col]
        
        # Compute similarity between user profile and all movies
        similarities = cosine_similarity([user_profile], genre_features)[0]
        
        # Get top N movies
        movie_indices = np.argsort(similarities)[::-1][:n]
        recommended_movie_ids = self.movies_df.iloc[movie_indices]['movieId'].tolist()
        
        return recommended_movie_ids
    
    def save_model(self, filepath='models/cb_model.pkl'):
        """
        Save the model to a file
        """
        if self.similarity_matrix is None:
            raise ValueError("Cannot save untrained model")
            
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        model_data = {
            'similarity_matrix': self.similarity_matrix,
            'feature_names': self.feature_names
        }
            
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath='models/cb_model.pkl'):
        """
        Load the model from a file
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.similarity_matrix = model_data['similarity_matrix']
        
        # Try to load feature_names if available in the model file
        if 'feature_names' in model_data:
            self.feature_names = model_data['feature_names']
        # Otherwise feature_names will be generated in get_recommendations_by_genres
            
        print(f"Model loaded from {filepath}")