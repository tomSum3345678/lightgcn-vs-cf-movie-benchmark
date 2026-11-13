# -*- coding: utf-8 -*-
"""
MovieLens 100K / 1M Conversion Script
Purpose: To convert official raw data formats to the standard movies.csv and ratings.csv formats used in our code.
"""
import os
import pandas as pd

def convert_1m(data_dir):
    print("Convert MovieLens 1M...")
    # Read Ratings (format: UserID::MovieID::Rating::Timestamp)
    ratings = pd.read_csv(
        os.path.join(data_dir, 'ratings.dat'), 
        sep='::', engine='python', header=None,
        names=['userId', 'movieId', 'rating', 'timestamp']
    )
    # Read Movies (format: MovieID::Title::Genres)
    movies = pd.read_csv(
        os.path.join(data_dir, 'movies.dat'), 
        sep='::', engine='python', header=None, encoding='latin-1',
        names=['movieId', 'title', 'genres']
    )
    
    ratings.to_csv(os.path.join(data_dir, 'ratings.csv'), index=False)
    movies.to_csv(os.path.join(data_dir, 'movies.csv'), index=False)
    print(f"✅ 1M conversion complete! A total of {len(ratings)} ratings, {len(movies)} movies.")

def convert_100k(data_dir):
    print("Converting MovieLens 100K...")
    # Read Ratings (format: user id | item id | rating | timestamp, tab separated)
    ratings = pd.read_csv(
        os.path.join(data_dir, 'u.data'), 
        sep='\t', header=None,
        names=['userId', 'movieId', 'rating', 'timestamp']
    )
    
    # Read Movies (format: movie id | movie title | release date ... | Genres One-Hot)
    genre_cols = [
        'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    cols = ['movieId', 'title', 'release_date', 'video_release_date', 'imdb_url'] + genre_cols
    
    movies_raw = pd.read_csv(
        os.path.join(data_dir, 'u.item'), 
        sep='|', header=None, encoding='latin-1', names=cols
    )
    
    # Convert One-Hot Genres back to '|' separated strings (aligned to ml-latest-small format)
    def get_genres(row):
        active_genres = [g for g in genre_cols if row[g] == 1]
        return '|'.join(active_genres) if active_genres else 'unknown'
        
    movies_raw['genres'] = movies_raw.apply(get_genres, axis=1)
    movies = movies_raw[['movieId', 'title', 'genres']]
    
    ratings.to_csv(os.path.join(data_dir, 'ratings.csv'), index=False)
    movies.to_csv(os.path.join(data_dir, 'movies.csv'), index=False)
    print(f"✅ 100K conversion complete! A total of {len(ratings)} ratings, {len(movies)} movies.")

if __name__ == '__main__':
    # Please uncomment the corresponding line number according to the downloaded dataset.
    #TARGET_DIR = './Datasets/ml-100k' # Dataset storage path
    #TARGET_DIR = './Datasets/ml-1m' Dataset storage path
    
    #convert_100k(TARGET_DIR)
    #convert_1m(TARGET_DIR)
    print("Please set TARGET_DIR and call the corresponding conversion function!")