import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

class SentimentAnalyzer:
    def __init__(self):
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            print("Downloading NLTK resources...")
            nltk.download('vader_lexicon')
            
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze_text(self, text):
        """
        Analyze sentiment of a text and return a score
        """
        if pd.isna(text) or text == "":
            return 0
            
        sentiment = self.analyzer.polarity_scores(text)
        return sentiment['compound']  # Compound score from -1 (negative) to 1 (positive)
    
    def analyze_tags(self, tags_df):
        """
        Analyze sentiment of movie tags
        """
        if tags_df is None:
            print("No tags data available for sentiment analysis")
            return pd.DataFrame()
            
        print("Analyzing sentiment of movie tags...")
        
        # Group tags by movieId
        grouped_tags = tags_df.groupby('movieId')['tag'].apply(list).reset_index()
        
        # Calculate sentiment for each movie's tags
        sentiment_scores = []
        
        for _, row in grouped_tags.iterrows():
            movie_id = row['movieId']
            tags = row['tag']
            
            # Calculate average sentiment score for all tags
            scores = [self.analyze_text(tag) for tag in tags]
            avg_score = np.mean(scores) if scores else 0
            
            sentiment_scores.append({
                'movieId': movie_id,
                'sentiment_score': avg_score,
                'tag_count': len(tags)
            })
            
        sentiment_df = pd.DataFrame(sentiment_scores)
        print(f"Sentiment analysis complete for {len(sentiment_df)} movies")
        
        return sentiment_df

if __name__ == "__main__":
    from data_preparation import prepare_data
    
    # Load and prepare data
    movies, ratings, tags, genre_df, train_data, test_data = prepare_data()
    
    # Initialize sentiment analyzer
    sentiment_analyzer = SentimentAnalyzer()
    
    # Analyze sentiment of tags if available
    if tags is not None:
        sentiment_df = sentiment_analyzer.analyze_tags(tags)
        
        # Print top 5 movies with positive sentiment
        print("\nTop 5 Movies with Positive Sentiment:")
        top_positive = sentiment_df.sort_values('sentiment_score', ascending=False).head(5)
        for _, row in top_positive.iterrows():
            movie_info = movies[movies['movieId'] == row['movieId']].iloc[0]
            print(f"- {movie_info['title']} (sentiment: {row['sentiment_score']:.2f})")
            
        # Print top 5 movies with negative sentiment
        print("\nTop 5 Movies with Negative Sentiment:")
        top_negative = sentiment_df.sort_values('sentiment_score').head(5)
        for _, row in top_negative.iterrows():
            movie_info = movies[movies['movieId'] == row['movieId']].iloc[0]
            print(f"- {movie_info['title']} (sentiment: {row['sentiment_score']:.2f})")