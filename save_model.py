"""
Build the recommendation model files from the TMDB CSV data.

Run locally:
    python save_model.py

The Streamlit app also calls build_model() automatically during deployment
when the model files are not present.
"""

import ast
import pickle
from pathlib import Path

import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ps = PorterStemmer()


def convert(obj):
    return [item["name"] for item in ast.literal_eval(obj)]


def convert3(obj):
    return [item["name"] for item in ast.literal_eval(obj)[:3]]


def fetch_director(obj):
    for item in ast.literal_eval(obj):
        if item["job"] == "Director":
            return [item["name"]]
    return []


def stem(text):
    return " ".join(ps.stem(word) for word in text.split())


def build_model(project_dir="."):
    project_path = Path(project_dir)
    model_dir = project_path / "model"
    model_dir.mkdir(exist_ok=True)

    movies_path = project_path / "tmdb_5000_movies.csv"
    credits_path = project_path / "tmdb_5000_credits.csv"

    if not movies_path.exists() or not credits_path.exists():
        missing = [
            str(path.name)
            for path in [movies_path, credits_path]
            if not path.exists()
        ]
        raise FileNotFoundError(f"Missing required data file(s): {', '.join(missing)}")

    print("Loading data...")
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    movies = movies.merge(credits, on="title")

    movies = movies[["movie_id", "genres", "title", "overview", "keywords", "cast", "crew"]]
    movies.dropna(inplace=True)

    print("Feature engineering...")
    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert3)
    movies["crew"] = movies["crew"].apply(fetch_director)
    movies["overview"] = movies["overview"].apply(lambda value: value.split())

    for column in ["genres", "keywords", "cast", "crew"]:
        movies[column] = movies[column].apply(
            lambda values: [value.replace(" ", "") for value in values]
        )

    movies["tags"] = (
        movies["overview"]
        + movies["keywords"]
        + movies["genres"]
        + movies["cast"]
        + movies["crew"]
    )

    new_df = movies[["movie_id", "title", "genres", "overview", "cast", "crew", "tags"]].copy()
    new_df["tags"] = new_df["tags"].apply(lambda values: " ".join(values))
    new_df["tags"] = new_df["tags"].apply(lambda value: value.lower())
    new_df["tags"] = new_df["tags"].apply(stem)
    new_df["overview"] = movies["overview"].apply(lambda values: " ".join(values))
    new_df = new_df.reset_index(drop=True)

    print("Vectorizing...")
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(new_df["tags"]).toarray()

    print("Computing cosine similarity...")
    similarity = cosine_similarity(vectors)

    print("Saving model files...")
    with (model_dir / "new_df.pkl").open("wb") as data_file:
        pickle.dump(new_df, data_file)
    with (model_dir / "similarity.pkl").open("wb") as similarity_file:
        pickle.dump(similarity, similarity_file)

    print("Done! Model saved to model/ folder.")
    print(f"  Movies: {len(new_df)}")
    print(f"  Similarity matrix: {similarity.shape}")

    return new_df, similarity


if __name__ == "__main__":
    build_model()
