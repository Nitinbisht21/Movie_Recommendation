# Movie Recommendation App

Streamlit app that recommends 5 similar movies using the TMDB 5000 movies dataset.

## Files

- `app.py` - Streamlit frontend
- `save_model.py` - builds the recommendation model
- `style.css` - app styling
- `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` - dataset files
- `requirements.txt` - Python packages

## Run Locally

```bash
pip install -r requirements.txt
python save_model.py
python -m streamlit run app.py
```

## TMDB Posters

To show movie posters, add your TMDB API key:

```bash
# .env
TMDB_API_KEY=your_api_key_here
```

Or use Streamlit secrets:

```toml
# .streamlit/secrets.toml
TMDB_API_KEY = "your_api_key_here"
```

You can also use a TMDB read access token:

```toml
TMDB_ACCESS_TOKEN = "your_read_access_token_here"
```

Do not commit your real API key.

## Deploy

For Streamlit Community Cloud:

- Repository: this GitHub repo
- Branch: `main`
- Main file: `app.py`

Do not commit the `model/` folder. The app rebuilds the model from the CSV files when needed.
