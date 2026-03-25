# Music Recommendation Frontend

This Flask app now combines multiple recommendation engines into one UI. The backend blends:

- a Spotify-style audio-feature nearest-neighbors engine
- a Hindi song text-based KNN recommender set
- a Telugu song text-based KNN recommender set

## Run locally

From the project folder:

```bash
cd c:\Users\sivar\OneDrive\Desktop\FSD\music-recommendation
python -m pip install -r requirements.txt
python run_server.py
```

Then open `http://127.0.0.1:5050`.

## Search inputs

The UI now uses:

- `genre` or mood, required
- `song_name`, required
- `language`, optional

Examples:

- genre: `love`, song: `Samajavaragamana`
- genre: `party`, language: `english`, song: `Blinding Lights`

## Project data and models

The app reads local project files instead of hardcoded `Downloads` paths:

- `recommendation_model.pkl`
- `spotify_tracks.zip`
- `models/songs_df.pkl`
- `models/features.pkl`
- `models/vectorizer.pkl`
- `models/knn_recommender.pkl`
- `models/knn_model.pkl`
- `models/telugu_df.pkl`
- `models/telugu_features.pkl`
- `models/telugu_vectorizer.pkl`
- `models/telugu_recommender.pkl`
- `models/telugu_knn_model.pkl`
- `models/telugu_model.pkl`

## API contract

The UI posts to `/api/recommend` with:

```json
{
  "genre": "party",
  "language": "english",
  "song_name": "Blinding Lights"
}
```

The backend responds with a merged recommendation payload:

```json
{
  "query": "Blinding Lights",
  "recommendations": [
    {
      "title": "Song title",
      "artist": "Artist name",
      "album": "Album name",
      "language": "telugu",
      "genre": "love",
      "engine": "telugu-nearest-neighbors"
    }
  ],
  "source": "6 engines combined",
  "engine_count": 6,
  "filters": {
    "genre": "party",
    "language": "english"
  }
}
```
