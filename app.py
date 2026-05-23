import html
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from save_model import build_model


APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "model"
MOVIES_CSV = APP_DIR / "tmdb_5000_movies.csv"
STYLE_CSS = APP_DIR / "style.css"
MODEL_DATA = MODEL_DIR / "new_df.pkl"
MODEL_SIMILARITY = MODEL_DIR / "similarity.pkl"


st.set_page_config(
    page_title="CineMatch Movie Recommender",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css():
    with STYLE_CSS.open("r", encoding="utf-8") as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)


load_css()

def escape(value):
    return html.escape(str(value), quote=True)


def pretty_token(value):
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(value))
    return text.replace("Science Fiction", "Science Fiction").strip()


def pretty_list(values, limit=None):
    if values is None:
        return "N/A"
    if isinstance(values, float) and np.isnan(values):
        return "N/A"
    items = list(values) if isinstance(values, (list, tuple, set)) else [values]
    if limit is not None:
        items = items[:limit]
    clean = [pretty_token(item) for item in items if str(item).strip()]
    return ", ".join(clean) if clean else "N/A"


def tags_html(values, limit=4):
    if not isinstance(values, (list, tuple, set)):
        return ""
    tags = [f"<span class='tag'>{escape(pretty_token(item))}</span>" for item in list(values)[:limit]]
    return "<div class='tag-row'>" + "".join(tags) + "</div>" if tags else ""


def ensure_model_files():
    if MODEL_DATA.exists() and MODEL_SIMILARITY.exists():
        return

    if not MOVIES_CSV.exists() or not (APP_DIR / "tmdb_5000_credits.csv").exists():
        raise FileNotFoundError(
            "Model files are missing, and the CSV files needed to rebuild them are not available."
        )

    build_model(APP_DIR)


@st.cache_resource(show_spinner=False)
def load_model():
    ensure_model_files()
    with MODEL_DATA.open("rb") as data_file:
        movies = pickle.load(data_file)
    with MODEL_SIMILARITY.open("rb") as similarity_file:
        similarity = pickle.load(similarity_file)
    return movies.reset_index(drop=True), similarity


@st.cache_data(show_spinner=False)
def load_metadata():
    if not MOVIES_CSV.exists():
        return pd.DataFrame()

    columns = [
        "id",
        "title",
        "release_date",
        "runtime",
        "vote_average",
        "vote_count",
        "popularity",
        "tagline",
    ]
    metadata = pd.read_csv(MOVIES_CSV, usecols=columns)
    metadata = metadata.rename(columns={"id": "movie_id"})
    metadata["release_year"] = pd.to_datetime(
        metadata["release_date"], errors="coerce"
    ).dt.year
    return metadata


def movie_details(movie_id, metadata):
    if metadata.empty:
        return {}
    matches = metadata[metadata["movie_id"] == movie_id]
    if matches.empty:
        return {}
    return matches.iloc[0].to_dict()


def format_number(value, decimals=0):
    if pd.isna(value):
        return "N/A"
    if decimals:
        return f"{float(value):,.{decimals}f}"
    return f"{int(float(value)):,}"


def format_runtime(value):
    if pd.isna(value):
        return "N/A"
    minutes = int(float(value))
    hours, mins = divmod(minutes, 60)
    if hours and mins:
        return f"{hours}h {mins}m"
    if hours:
        return f"{hours}h"
    return f"{mins}m"


def format_year(value):
    if pd.isna(value):
        return ""
    return str(int(float(value)))


def recommend(movie_title, movies, similarity, limit):
    matches = movies.index[movies["title"] == movie_title].tolist()
    if not matches:
        return []

    selected_index = matches[0]
    scores = np.asarray(similarity[selected_index])
    ranked = np.argsort(scores)[::-1]
    ranked = [index for index in ranked if index != selected_index][:limit]

    recommendations = []
    for index in ranked:
        row = movies.iloc[index]
        recommendations.append(
            {
                "movie_id": row.get("movie_id"),
                "title": row.get("title", ""),
                "genres": row.get("genres", []),
                "overview": row.get("overview", ""),
                "cast": row.get("cast", []),
                "crew": row.get("crew", []),
                "score": round(float(scores[index]) * 100, 1),
            }
        )
    return recommendations


def render_movie_panel(row, details):
    release_year = details.get("release_year", "N/A")
    rating = format_number(details.get("vote_average"), 1)
    votes = format_number(details.get("vote_count"))
    runtime = format_runtime(details.get("runtime"))
    tagline = details.get("tagline", "")

    meta_bits = [
        f"Released {format_number(release_year)}" if not pd.isna(release_year) else "Release N/A",
        f"Runtime {runtime}",
        f"Rating {rating}/10",
        f"{votes} votes",
    ]
    tagline_html = (
        f"<div class='movie-meta'>{escape(tagline)}</div>"
        if isinstance(tagline, str) and tagline.strip()
        else ""
    )

    st.markdown(
        f"""
        <div class='movie-panel'>
            <h2>{escape(row["title"])}</h2>
            <div class='movie-meta'>{escape(" | ".join(meta_bits))}</div>
            {tags_html(row.get("genres", []), limit=6)}
            {tagline_html}
            <div class='cast-line'><b>Cast:</b> {escape(pretty_list(row.get("cast", []), limit=3))}</div>
            <div class='cast-line'><b>Director:</b> {escape(pretty_list(row.get("crew", []), limit=1))}</div>
            <div class='movie-overview'>{escape(row.get("overview", "No overview available."))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation_cards(recommendations, metadata):
    cards = []
    for rank, item in enumerate(recommendations, start=1):
        details = movie_details(item.get("movie_id"), metadata)
        year = details.get("release_year", "")
        year_text = f" ({format_year(year)})" if not pd.isna(year) else ""
        overview = str(item.get("overview", "No overview available."))
        if len(overview) > 145:
            overview = overview[:142].rstrip() + "..."

        cards.append(
            "<article class='movie-card'>"
            "<div class='poster'>"
            f"<div class='rank-number'>#{rank}</div>"
            "<div class='poster-label'>Recommended Movie</div>"
            "</div>"
            "<div class='card-body'>"
            f"<div class='match'>{escape(item['score'])}% match</div>"
            f"<h3>{escape(item['title'])}{escape(year_text)}</h3>"
            f"{tags_html(item.get('genres', []), limit=3)}"
            f"<div class='cast-line'><b>Director:</b> {escape(pretty_list(item.get('crew', []), limit=1))}</div>"
            f"<div class='overview'>{escape(overview)}</div>"
            "</div>"
            "</article>"
        )

    st.markdown("<div class='card-grid'>" + "".join(cards) + "</div>", unsafe_allow_html=True)


with st.sidebar:
    st.title("CineMatch")
    st.caption("Content-based recommendations from the TMDB 5000 movie dataset.")
    st.divider()

    result_count = 5
    st.metric("Recommendations shown", result_count)
    st.caption("The app shows the top 5 matches. The model compares genre, keywords, cast, director, and overview tags.")

    st.divider()
    st.markdown("**Project files**")
    st.caption("Notebook: MoviesRecommendation.ipynb")
    st.caption("Data: tmdb_5000_movies.csv, tmdb_5000_credits.csv")
    st.caption("Model: model/new_df.pkl, model/similarity.pkl")


try:
    with st.spinner("Preparing the recommendation model..."):
        movies, similarity = load_model()
except FileNotFoundError as error:
    st.error(str(error))
    st.info("Add both TMDB CSV files to the repository, then redeploy or run `python save_model.py` locally.")
    st.stop()

metadata = load_metadata()
titles = movies["title"].dropna().sort_values().tolist()

rated_movies = metadata.dropna(subset=["vote_average", "vote_count"]) if not metadata.empty else pd.DataFrame()
popular_count = 0 if rated_movies.empty else len(rated_movies[rated_movies["vote_count"] >= 100])
average_rating = 0 if rated_movies.empty else rated_movies["vote_average"].mean()

st.markdown(
    f"""
    <div class='topbar'>
        <section class='masthead'>
            <h1>CineMatch</h1>
            <p>Pick a film you already like and get fast recommendations from your trained content-based movie model.</p>
        </section>
        <aside class='stat-panel'>
            <div class='stat'><span>Movies</span><strong>{len(movies):,}</strong></div>
            <div class='stat'><span>Similarity</span><strong>Cosine</strong></div>
            <div class='stat'><span>Rated Titles</span><strong>{popular_count:,}</strong></div>
            <div class='stat'><span>Average Rating</span><strong>{average_rating:.1f}</strong></div>
        </aside>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='section-label'>Find a Movie</div>", unsafe_allow_html=True)
with st.form("recommendation_form", clear_on_submit=False):
    left, right = st.columns([5, 1])
    with left:
        selected_movie = st.selectbox(
            "Movie title",
            options=titles,
            index=titles.index("Avatar") if "Avatar" in titles else 0,
            label_visibility="collapsed",
        )
    with right:
        submitted = st.form_submit_button("Recommend", use_container_width=True)

selected_row = movies[movies["title"] == selected_movie].iloc[0]
selected_details = movie_details(selected_row.get("movie_id"), metadata)
render_movie_panel(selected_row, selected_details)

if submitted:
    recommendations = recommend(selected_movie, movies, similarity, result_count)
    st.markdown("<div class='section-label'>Recommended For You</div>", unsafe_allow_html=True)
    if recommendations:
        render_recommendation_cards(recommendations, metadata)
    else:
        st.markdown(
            "<div class='empty-state'>No recommendations found for this movie.</div>",
            unsafe_allow_html=True,
        )
else:
    st.markdown("<div class='section-label'>Start Here</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='empty-state'>Choose a title and press Recommend to see 5 similar movies.</div>",
        unsafe_allow_html=True,
    )
