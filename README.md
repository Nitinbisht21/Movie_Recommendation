# 🎬 CineMatch — Content-Based Movie Recommender

*Pick a movie you like. Get 5 similar ones instantly — powered by machine learning, not guesswork.*

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![TMDB API](https://img.shields.io/badge/TMDB%20API-01D277?style=flat)
![status](https://img.shields.io/badge/status-live-brightgreen)

---

## 🔗 Quick Links

| | |
|---|---|
| 🚀 **Live App** | [movierecommendation-run.streamlit.app](https://movierecommendation-run.streamlit.app/) |
| 📊 **Dataset** | [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) (Kaggle) |
| 💻 **Source** | This repository |

---

## 📖 About

CineMatch is a **content-based movie recommendation engine** built from scratch — no external recommendation APIs, no black-box magic. It reads a movie's genres, cast, director, keywords, and plot overview, turns all of that into vectors, and finds the movies that are mathematically "closest" to whatever you pick.

Built as a hands-on project to go beyond ML theory and ship something real: a trained model, a deployed app, live poster fetching from an external API, and all the deployment headaches that come with taking something from "works on my laptop" to "works in production."

---

## ⚙️ How It Works

```
Raw movie data (genres, cast, crew, keywords, overview)
        │
        ▼
Clean + merge into a single "tags" field per movie
        │
        ▼
Vectorize tags → CountVectorizer (Bag-of-Words, top 5000 features)
        │
        ▼
Compute pairwise Cosine Similarity across all 4,800+ movies
        │
        ▼
Given a movie → rank all others by similarity score → return top 5
```

The model is trained once via `save_model.py` and cached as `.pkl` files, so the live app doesn't retrain on every request — it just looks up pre-computed similarity scores.

---

## ✨ Features

- 🎯 **Content-based filtering** — recommendations based on what a movie is actually about, not user ratings or watch history
- 🧮 **Cosine similarity** over a 5000-feature Bag-of-Words vector space
- 🖼️ **Live poster fetching** via the TMDB API — posters, ratings, runtime, and cast pulled in real time
- ⚡ **Instant results** — similarity matrix is pre-computed, so recommendations return in milliseconds
- 🎨 **Clean, custom UI** — dark-themed Streamlit interface with styled movie cards
- ☁️ **Deployed and public** — no setup needed to try it

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| ML | scikit-learn (`CountVectorizer`, cosine similarity), NLTK (stemming) |
| Frontend / Deployment | Streamlit, Streamlit Community Cloud |
| External API | TMDB API (posters ) |

---

## 🚀 Run It Locally

```bash
git clone https://github.com/Nitinbisht21/Movie_Recommendation.git
cd Movie_Recommendation
pip install -r requirements.txt
python save_model.py        
streamlit run app.py
```

### Enable movie posters (optional but recommended)

Add your TMDB API key to a local `.env` file:

```bash
TMDB_API_KEY=your_api_key_here
```

or via Streamlit secrets:

```toml
# .streamlit/secrets.toml
TMDB_API_KEY = "your_api_key_here"
```

Get a free key at [themoviedb.org](https://www.themoviedb.org/settings/api). The key is only ever read server-side — never exposed in the UI or committed to Git.

---

## 📂 Project Structure

```
├── app.py                    # Streamlit app (UI + TMDB integration)
├── save_model.py             # Builds the recommendation model
├── MoviesRecommendation.ipynb # Original notebook
├── style.css                 # Custom app styling
├── tmdb_5000_movies.csv      # Dataset — movie metadata
├── tmdb_5000_credits.csv     # Dataset — cast & crew
└── requirements.txt
```


## 🙋 About Me

Built by **Nitin Bisht** — B.Tech CSE (AI & ML) student.
Feel free to fork, star, or open an issue if you spot something worth improving.

⭐ If this project was useful or interesting, a star on the repo goes a long way!
