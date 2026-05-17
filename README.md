# CineMatch Movie Recommender

A Streamlit movie recommendation app built from the TMDB 5000 movies and credits datasets.

## Run Locally

```bash
pip install -r requirements.txt
python save_model.py
streamlit run app.py
```

The app can also build the model automatically on first startup if the `model/` folder is missing.

## Deploy

Deploy this project as a Streamlit app with:

- Repository: your GitHub repository
- Branch: `main`
- Main file path: `app.py`

The two CSV files must stay in the repository because the deployment rebuilds the ignored model files from them.

## Files

- `app.py` - Streamlit frontend and recommendation UI
- `save_model.py` - builds `model/new_df.pkl` and `model/similarity.pkl`
- `MoviesRecommendation.ipynb` - original notebook work
- `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` - source data
- `requirements.txt` - Python dependencies

