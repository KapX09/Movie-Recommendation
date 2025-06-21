import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load dataset
df = pd.read_csv('imdb_top_1000.csv')

# Rename and select relevant columns for clarity and easier access
df = df.rename(columns={
    'Series_Title': 'title',
    'Genre': 'genre',
    'Overview': 'overview',
    'Star1': 'star1', 'Star2': 'star2',
    'Star3': 'star3', 'Star4': 'star4',
})
# Keep only the columns we will use for building the content-based model
df = df[['title', 'genre', 'overview', 'star1', 'star2', 'star3', 'star4']]

# Fill missing values with empty strings so TF-IDF doesn't break on NaNs
for col in ['genre', 'overview', 'star1', 'star2', 'star3', 'star4']:
    df[col] = df[col].fillna('')

# Combine genre, overview, and star names into one text string per movie
# This forms the "content" that will be vectorized for similarity comparison
df['content'] = (
    df['genre'] + ' ' +
    df['overview'] + ' ' +
    df['star1'] + ' ' +
    df['star2'] + ' ' +
    df['star3'] + ' ' +
    df['star4']
)

# Initialize TF-IDF vectorizer, remove common English stop words, and limit max features for speed
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

# Apply TF-IDF to the 'content' column, converting text to numerical feature vectors
tfidf_matrix = tfidf.fit_transform(df['content'])

# Compute the cosine similarity matrix between all movies based on their TF-IDF vectors
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a mapping from movie title to its index in the dataframe for quick lookup
# Strip whitespace to avoid key errors due to extra spaces, drop duplicates for safety
indices = pd.Series(df.index, index=df['title'].str.strip()).drop_duplicates()

# Function to get recommendations given a movie title and number of recommendations
def get_recommendations(title, num_recommendations=10):
    title = title.strip()  # Clean whitespace
    if title not in indices:
        print(f"Movie '{title}' not found.")
        return None

    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get a list of tuples (movie_index, similarity_score) for this movie compared to all others
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the list of tuples based on similarity score in descending order
    # Exclude the first one because it's the same movie (similarity = 1)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]

    # Extract just the indices of the most similar movies
    recommended_idxs = [i[0] for i in sim_scores]

    # Return the titles of the recommended movies
    return df['title'].iloc[recommended_idxs].tolist()

# Example usage: Get top 5 movies similar to 'The Shawshank Redemption'
recs = get_recommendations('The Shawshank Redemption', num_recommendations=5)

if recs:
    print("Top 5 recommendations:")
    for r in recs:
        print("-", r)