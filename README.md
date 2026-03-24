# Music Recommendation Frontend

This project now includes a Flask frontend connected to a trained music recommendation backend.

## Run locally

```bash
pip install -r requirements.txt
python frontend_app.py
```

Then open `http://127.0.0.1:5000`.




## Connected backend files

By default the app reads:

- `c:\Users\phane\Downloads\recommendation_model.pkl`
- `c:\Users\phane\Downloads\archive (1).zip`
- `spotify_tracks.csv` inside that zip

You can override those paths with:

```bash
set RECOMMENDER_MODEL_PATH=your_model_path
set RECOMMENDER_DATASET_ZIP=your_zip_path
set RECOMMENDER_DATASET_FILE=spotify_tracks.csv
```

## API contract

The UI posts to `/api/recommend` with:

```json
{
  "song_name": "Blinding Lights"
}
```

The Flask backend returns data in this shape:

```json
{
  "query": "Blinding Lights",
  "recommendations": [
    {
      "title": "Song title",
      "artist": "Artist name",
      "album": "Album name"
    }
  ]
}
```

## Model wiring

The backend uses your saved `NearestNeighbors` model together with `spotify_tracks.csv`. It finds the requested song, standardizes these 9 audio features, and queries the model:

- `acousticness`
- `danceability`
- `energy`
- `instrumentalness`
- `liveness`
- `loudness`
- `speechiness`
- `tempo`
- `valence`
