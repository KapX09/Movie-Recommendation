# Movie-Recommendation
This is _Content-Based_ ML Model with the _DATASET_: **imbd_top_1000.csv**.

#### FLow of working:
Load Dataset (CSV). 
- Select relevant columns (title, genre, overview, stars)
- Fill missing values (empty strings)
- Combine text columns into one "content" string per movie.
- Convert "content" strings into numerical vectors using TF-IDF.
- Calculate cosine similarity between all movie vectors.
- User inputs a movie title.
-  Find the movie's vector index.
-  Sort other movies by similarity score to this movie.
-  Return top-N most similar movie titles

