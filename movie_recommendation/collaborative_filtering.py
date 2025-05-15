import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class CollaborativeFiltering:
    def __init__(self, ratings_df=None):
        self.model = None
        self.ratings_df = ratings_df
        self.is_trained = False
        self.user_factors = None
        self.item_factors = None
        self.user_mapping = None
        self.item_mapping = None
        self.reverse_user_mapping = None
        self.reverse_item_mapping = None
        
    def prepare_data(self, ratings_df=None):
        """
        Prepare data for SVD model
        """
        if ratings_df is not None:
            self.ratings_df = ratings_df
            
        if self.ratings_df is None:
            raise ValueError("Ratings dataframe is required")
            
        # Create user and item mapping dictionaries
        unique_users = self.ratings_df['userId'].unique()
        unique_movies = self.ratings_df['movieId'].unique()
        
        self.user_mapping = {user_id: i for i, user_id in enumerate(unique_users)}
        self.item_mapping = {movie_id: i for i, movie_id in enumerate(unique_movies)}
        
        self.reverse_user_mapping = {i: user_id for user_id, i in self.user_mapping.items()}
        self.reverse_item_mapping = {i: movie_id for movie_id, i in self.item_mapping.items()}
        
        # Create rating matrix (users x items)
        n_users = len(unique_users)
        n_items = len(unique_movies)
        
        # Initialize rating matrix
        rating_matrix = np.zeros((n_users, n_items))
        
        # Fill rating matrix
        for _, row in self.ratings_df.iterrows():
            user_idx = self.user_mapping[row['userId']]
            item_idx = self.item_mapping[row['movieId']]
            rating_matrix[user_idx, item_idx] = row['rating']
        
        return rating_matrix
    
    def train(self, ratings_df=None, n_components=100):
        """
        Train the collaborative filtering model using SVD
        """
        rating_matrix = self.prepare_data(ratings_df)
        
        print("Training collaborative filtering model...")
        # Apply SVD
        self.model = TruncatedSVD(n_components=n_components, random_state=42)
        self.item_factors = self.model.fit_transform(rating_matrix.T)
        
        # Calculate user factors
        self.user_factors = rating_matrix @ self.item_factors @ np.linalg.pinv(
            np.diag(self.model.singular_values_)
        )
        
        self.is_trained = True
        print("Training complete!")
        
    def predict(self, user_id, movie_id):
        """
        Predict the rating for a given user and movie
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
            
        # Convert user_id and movie_id to internal indices
        if user_id not in self.user_mapping or movie_id not in self.item_mapping:
            # Return average rating if user or movie not in training data
            return 3.0
            
        user_idx = self.user_mapping[user_id]
        movie_idx = self.item_mapping[movie_id]
        
        # Get user and item factors
        user_vec = self.user_factors[user_idx]
        item_vec = self.item_factors[movie_idx]
        
        # Make prediction
        prediction = np.dot(user_vec, item_vec) / np.linalg.norm(user_vec) / np.linalg.norm(item_vec)
        
        # Scale prediction to match rating scale
        scaled_prediction = (prediction + 1) * 2.5
        
        # Clip prediction to rating range
        return max(0.5, min(5.0, scaled_prediction))
    
    def get_top_n_recommendations(self, user_id, movie_list, n=5):
        """
        Get the top N movie recommendations for a given user
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
            
        # Predict ratings for all movies in the list
        predictions = []
        for movie_id in movie_list:
            pred_rating = self.predict(user_id, movie_id)
            predictions.append((movie_id, pred_rating))
            
        # Sort predictions by rating (descending)
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N movie IDs
        top_n_movie_ids = [movie_id for movie_id, _ in predictions[:n]]
        return top_n_movie_ids
    
    def save_model(self, filepath='models/cf_model.pkl'):
        """
        Save the model to a file
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
            
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        model_data = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_mapping': self.user_mapping,
            'item_mapping': self.item_mapping,
            'reverse_user_mapping': self.reverse_user_mapping,
            'reverse_item_mapping': self.reverse_item_mapping,
            'singular_values': self.model.singular_values_ if self.model else None
        }
            
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath='models/cf_model.pkl'):
        """
        Load the model from a file
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.user_factors = model_data['user_factors']
        self.item_factors = model_data['item_factors']
        self.user_mapping = model_data['user_mapping']
        self.item_mapping = model_data['item_mapping']
        self.reverse_user_mapping = model_data['reverse_user_mapping']
        self.reverse_item_mapping = model_data['reverse_item_mapping']
        
        # Recreate the SVD model
        self.model = TruncatedSVD(n_components=self.user_factors.shape[1], random_state=42)
        if 'singular_values' in model_data and model_data['singular_values'] is not None:
            self.model.singular_values_ = model_data['singular_values']
            
        self.is_trained = True
        print(f"Model loaded from {filepath}")