import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Sample dataset (User ID, Movie ID, Rating)
data = {
    'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    'movie_id': [101, 102, 103, 101, 103, 104, 102, 103, 104, 101, 102, 104],
    'rating': [5, 3, 4, 4, 5, 2, 5, 4, 3, 2, 4, 5]
}

# Create DataFrame
df = pd.DataFrame(data)

# Pivot the table to create a user-movie matrix
user_movie_matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating')

# Fill NaN values with 0 (can also use mean ratings or other strategies)
user_movie_matrix.fillna(0, inplace=True)

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_movie_matrix)

# Convert it to a DataFrame for better readability
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# Function to get movie recommendations for a user based on similar users
def get_recommendations(user_id, user_movie_matrix, user_similarity_df, top_n=2):
    # Find similar users for the given user_id
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]
    
    # Get the movies rated by the similar users, and not yet rated by the current user
    user_movies = user_movie_matrix.loc[user_id]
    recommended_movies = pd.Series(dtype='float64')
    
    for similar_user in similar_users:
        similar_user_movies = user_movie_matrix.loc[similar_user]
        # Filter out movies the user has already rated
        unseen_movies = similar_user_movies[user_movies == 0]
        # Add to recommended movies with ratings
        recommended_movies = recommended_movies.append(unseen_movies)
    
    # Return the top N recommendations, sorted by rating
    return recommended_movies.sort_values(ascending=False).head(top_n)

# Example usage: Recommend 2 movies for user with user_id = 1
recommended_movies = get_recommendations(user_id=1, user_movie_matrix=user_movie_matrix, user_similarity_df=user_similarity_df, top_n=2)
print(f"Recommended movies for user 1:\n{recommended_movies}")
