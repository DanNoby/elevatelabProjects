import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import zipfile
import requests
from io import BytesIO
import os

def download_movielens_dataset(size='small'):
    """
    Download the MovieLens dataset
    size: 'small' (100K) or 'full' (25M)
    """
    if size == 'small':
        url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    else:
        url = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
    
    print(f"Downloading MovieLens {size} dataset...")
    r = requests.get(url)
    z = zipfile.ZipFile(BytesIO(r.content))
    z.extractall('data/')
    print("Dataset downloaded successfully!")
    
    if size == 'small':
        return 'data/ml-latest-small'
    else:
        return 'data/ml-latest'

def load_and_preprocess_data(dataset_path):
    """
    Load and preprocess the MovieLens dataset
    """
    print("Loading and preprocessing data...")
    
    # Load the data
    movies = pd.read_csv(f"{dataset_path}/movies.csv")
    ratings = pd.read_csv(f"{dataset_path}/ratings.csv")
    
    # Optional: Load tags if available for sentiment analysis
    try:
        tags = pd.read_csv(f"{dataset_path}/tags.csv")
    except:
        tags = None
    
    # Handle missing values in movies
    movies = movies.dropna(subset=['title', 'genres'])
    
    # Clean and preprocess movie genres
    movies['genres'] = movies['genres'].str.replace('|', ',')
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').fillna(0).astype(int)
    movies['clean_title'] = movies['title'].apply(lambda x: x.split('(')[0].strip())
    
    # Clean ratings data
    ratings = ratings.dropna()
    
    # Normalize ratings
    scaler = MinMaxScaler()
    ratings['normalized_rating'] = scaler.fit_transform(ratings[['rating']])
    
    # Create movie features dataframe with average rating and rating count
    movie_features = ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        rating_count=('rating', 'count')
    ).reset_index()
    
    # Merge with movies dataframe
    movies = movies.merge(movie_features, on='movieId', how='left')
    
    # Replace NaN values with 0 for rating metrics
    movies['avg_rating'] = movies['avg_rating'].fillna(0)
    movies['rating_count'] = movies['rating_count'].fillna(0)
    
    # Create genre matrix for content-based filtering
    genre_df = create_genre_matrix(movies)
    
    return movies, ratings, tags, genre_df

def create_genre_matrix(movies):
    """
    Create a matrix of movies and their genres
    """
    # Create a list of all genres
    genres = set()
    for genre_list in movies['genres'].str.split(','):
        if isinstance(genre_list, list):
            genres.update(genre_list)
    
    genres = sorted(list(genres))
    if '(no genres listed)' in genres:
        genres.remove('(no genres listed)')
    
    # Create a dataframe with one-hot encoding for genres
    genre_df = pd.DataFrame(0, index=movies.index, columns=genres)
    
    for i, genre_list in enumerate(movies['genres'].str.split(',')):
        if isinstance(genre_list, list):
            for genre in genre_list:
                if genre in genres:
                    genre_df.loc[i, genre] = 1
    
    # Add movieId column to the dataframe
    genre_df['movieId'] = movies['movieId'].values
    
    return genre_df

def prepare_data():
    """
    Main function to prepare the data
    """
    # Check if data is already downloaded
    if not os.path.exists('data/ml-latest-small'):
        dataset_path = download_movielens_dataset(size='small')
    else:
        dataset_path = 'data/ml-latest-small'
    
    # Load and preprocess the data
    movies, ratings, tags, genre_df = load_and_preprocess_data(dataset_path)
    
    # Split data into training and testing sets
    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
    
    print(f"Data preparation complete: {len(movies)} movies, {len(ratings)} ratings")
    return movies, ratings, tags, genre_df, train_data, test_data

if __name__ == "__main__":
    movies, ratings, tags, genre_df, train_data, test_data = prepare_data()