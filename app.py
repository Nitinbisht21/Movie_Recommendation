import html
import os
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
MODEL_DATA = MODEL_DIR / "new_df.pkl"
MODEL_SIMILARITY = MODEL_DIR / "similarity.pkl"


st.set_page_config(
    page_title="CineMatch Movie Recommender",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --page: #101114;
        --panel: #17191e;
        --panel-2: #20242b;
        --line: #343944;
        --text: #f5f1e8;
        --muted: #a9b0bd;
        --soft: #737b89;
        --accent: #f3bf3c;
        --accent-2: #c84e47;
        --ok: #63b47a;
    }

    html, body, .stApp {
        background: var(--page);
        color: var(--text);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    header, footer, #MainMenu {
        visibility: hidden;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }

    [data-testid="stSidebar"] {
        background: #13151a;
        border-right: 1px solid var(--line);
    }

    h1, h2, h3, p {
        letter-spacing: 0;
    }

    .topbar {
        display: grid;
        grid-template-columns: minmax(0, 1.5fr) minmax(260px, 0.7fr);
        gap: 1rem;
        align-items: stretch;
        margin-bottom: 1.25rem;
    }

    .masthead {
        min-height: 210px;
        padding: 1.8rem;
        border: 1px solid var(--line);
        border-radius: 8px;
        background:
            linear-gradient(110deg, rgba(16, 17, 20, 0.88), rgba(16, 17, 20, 0.58)),
            repeating-linear-gradient(90deg, rgba(243, 191, 60, 0.18) 0 2px, transparent 2px 34px),
            linear-gradient(135deg, #24201a, #20242b 48%, #1d1717);
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
    }

    .masthead h1 {
        margin: 0;
        color: var(--text);
        font-size: clamp(2.2rem, 5vw, 4.2rem);
        line-height: 0.95;
        font-weight: 850;
    }

    .masthead p {
        margin: 0.75rem 0 0;
        max-width: 760px;
        color: var(--muted);
        font-size: 1.02rem;
        line-height: 1.55;
    }

    .stat-panel {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--panel);
        padding: 1rem;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.75rem;
    }

    .stat {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        background: #1c2027;
        padding: 0.9rem;
        min-height: 88px;
    }

    .stat span {
        display: block;
        color: var(--soft);
        font-size: 0.72rem;
        text-transform: uppercase;
        font-weight: 700;
    }

    .stat strong {
        display: block;
        margin-top: 0.35rem;
        color: var(--text);
        font-size: clamp(1.15rem, 2.5vw, 1.75rem);
        line-height: 1.1;
    }

    .section-label {
        margin: 1.2rem 0 0.65rem;
        color: var(--muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 800;
    }

    .movie-panel {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--panel);
        padding: 1rem;
        margin: 0.85rem 0 1.1rem;
    }

    .movie-panel h2 {
        margin: 0 0 0.45rem;
        color: var(--text);
        font-size: clamp(1.55rem, 3vw, 2.25rem);
        line-height: 1.05;
    }

    .movie-meta {
        color: var(--muted);
        font-size: 0.92rem;
        margin-bottom: 0.8rem;
    }

    .movie-overview {
        color: #d7dbe2;
        line-height: 1.55;
        margin-top: 0.8rem;
    }

    .tag-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin: 0.35rem 0;
    }

    .tag {
        display: inline-flex;
        align-items: center;
        min-height: 24px;
        border: 1px solid rgba(243, 191, 60, 0.36);
        border-radius: 999px;
        color: #f9d97d;
        background: rgba(243, 191, 60, 0.08);
        padding: 0.12rem 0.55rem;
        font-size: 0.77rem;
        font-weight: 700;
    }

    .cast-line {
        color: var(--muted);
        font-size: 0.88rem;
        margin-top: 0.45rem;
    }

    .cast-line b {
        color: var(--text);
    }

    .card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
        gap: 0.85rem;
    }

    .movie-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--panel);
        min-height: 340px;
        overflow: hidden;
    }

    .poster {
        min-height: 145px;
        background:
            linear-gradient(145deg, rgba(243, 191, 60, 0.16), rgba(200, 78, 71, 0.16)),
            linear-gradient(135deg, #20242b, #14161b);
        border-bottom: 1px solid var(--line);
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
        align-items: center;
        justify-content: center;
        color: rgba(245, 241, 232, 0.86);
        font-weight: 900;
    }

    .rank-number {
        color: var(--accent);
        font-size: 2.4rem;
        line-height: 1;
    }

    .poster-label {
        color: var(--muted);
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }

    .card-body {
        padding: 0.85rem;
    }

    .match {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        background: rgba(99, 180, 122, 0.13);
        color: #98d7a8;
        padding: 0.15rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 800;
        margin-bottom: 0.55rem;
    }

    .movie-card h3 {
        margin: 0;
        color: var(--text);
        font-size: 1.02rem;
        line-height: 1.2;
    }

    .overview {
        margin-top: 0.55rem;
        color: var(--muted);
        font-size: 0.84rem;
        line-height: 1.45;
    }

    .empty-state {
        border: 1px dashed var(--line);
        border-radius: 8px;
        background: rgba(255,255,255,0.03);
        color: var(--muted);
        padding: 2rem;
        text-align: center;
    }

    .stButton > button {
        border-radius: 8px;
        border: 1px solid rgba(243, 191, 60, 0.55);
        background: #f3bf3c;
        color: #151515;
        font-weight: 850;
        min-height: 44px;
    }

    .stButton > button:hover {
        border-color: #ffd76e;
        background: #ffd05d;
        color: #151515;
    }

    div[data-baseweb="select"] > div {
        border-radius: 8px;
        background: var(--panel);
        border-color: var(--line);
    }

    @media (max-width: 900px) {
        .topbar {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
