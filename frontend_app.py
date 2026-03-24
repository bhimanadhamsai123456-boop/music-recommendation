from __future__ import annotations

import os
import zipfile
from difflib import get_close_matches
from functools import lru_cache
from typing import Any

import pandas as pd
from flask import Flask, jsonify, render_template, request
from joblib import load
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

MODEL_PATH = os.getenv(
    "RECOMMENDER_MODEL_PATH",
    r"c:\Users\phane\Downloads\recommendation_model.pkl",
)
DATASET_ZIP_PATH = os.getenv(
    "RECOMMENDER_DATASET_ZIP",
    r"c:\Users\phane\Downloads\archive (1).zip",
)
DATASET_FILE_NAME = os.getenv("RECOMMENDER_DATASET_FILE", "spotify_tracks.csv")
FEATURE_COLUMNS = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
]


@lru_cache(maxsize=1)
def load_backend_assets() -> tuple[Any, pd.DataFrame, StandardScaler]:
    model = load(MODEL_PATH)

    with zipfile.ZipFile(DATASET_ZIP_PATH) as zf:
        with zf.open(DATASET_FILE_NAME) as handle:
            songs = pd.read_csv(handle)

    scaler = StandardScaler()
    scaler.fit(songs[FEATURE_COLUMNS])
    return model, songs, scaler


def find_song_row(songs: pd.DataFrame, song_name: str) -> tuple[pd.Series | None, str]:
    normalized = song_name.casefold()
    exact_matches = songs[songs["track_name"].str.casefold() == normalized]
    if not exact_matches.empty:
        row = exact_matches.sort_values("popularity", ascending=False).iloc[0]
        return row, str(row["track_name"])

    contains_matches = songs[songs["track_name"].str.casefold().str.contains(normalized, na=False)]
    if not contains_matches.empty:
        row = contains_matches.sort_values("popularity", ascending=False).iloc[0]
        return row, str(row["track_name"])

    name_lookup = songs["track_name"].drop_duplicates().tolist()
    close = get_close_matches(song_name, name_lookup, n=1, cutoff=0.6)
    if close:
        row = songs[songs["track_name"] == close[0]].sort_values("popularity", ascending=False).iloc[0]
        return row, str(row["track_name"])

    return None, song_name


def build_recommendations(song_name: str) -> tuple[str, list[dict[str, str]]]:
    model, songs, scaler = load_backend_assets()
    row, resolved_name = find_song_row(songs, song_name)

    if row is None:
        return resolved_name, []

    song_features = row[FEATURE_COLUMNS].to_frame().T
    scaled_features = scaler.transform(song_features)
    _, indices = model.kneighbors(scaled_features)

    recommendations: list[dict[str, str]] = []
    seed_track_id = str(row["track_id"])

    for idx in indices[0]:
        rec = songs.iloc[int(idx)]
        if str(rec["track_id"]) == seed_track_id:
            continue

        recommendations.append(
            {
                "title": str(rec["track_name"]),
                "artist": str(rec["artist_name"]),
                "album": str(rec["album_name"]),
                "year": str(rec["year"]),
                "popularity": str(rec["popularity"]),
                "language": str(rec["language"]),
                "artwork_url": str(rec["artwork_url"]),
                "track_url": str(rec["track_url"]),
            }
        )

    return resolved_name, recommendations[:5]


def normalize_results(payload: Any) -> list[dict[str, str]]:
    if isinstance(payload, dict):
        items = payload.get("recommendations", [])
    else:
        items = payload

    normalized: list[dict[str, str]] = []
    for item in items or []:
        if isinstance(item, str):
            normalized.append(
                {
                    "title": item,
                    "artist": "Unknown Artist",
                    "album": "Recommended for you",
                    "year": "",
                    "popularity": "",
                    "language": "",
                    "artwork_url": "",
                    "track_url": "",
                }
            )
            continue

        if isinstance(item, dict):
            normalized.append(
                {
                    "title": str(item.get("title") or item.get("song") or "Untitled"),
                    "artist": str(item.get("artist") or "Unknown Artist"),
                    "album": str(item.get("album") or "Recommended for you"),
                    "year": str(item.get("year") or ""),
                    "popularity": str(item.get("popularity") or ""),
                    "language": str(item.get("language") or ""),
                    "artwork_url": str(item.get("artwork_url") or ""),
                    "track_url": str(item.get("track_url") or ""),
                }
            )

    return normalized


@app.get("/")
def home() -> str:
    return render_template("index.html")


@app.post("/api/recommend")
def recommend():
    data = request.get_json(silent=True) or {}
    song_name = str(data.get("song_name", "")).strip()

    if not song_name:
        return jsonify({"error": "Please enter a song name."}), 400

    try:
        resolved_query, raw_results = build_recommendations(song_name)
    except FileNotFoundError as exc:
        return jsonify({"error": f"Backend file not found: {exc.filename}"}), 500
    except Exception as exc:
        return jsonify({"error": f"Recommendation backend failed: {exc}"}), 500

    results = normalize_results(raw_results)
    return jsonify(
        {
            "query": resolved_query,
            "recommendations": results,
            "source": "trained-nearest-neighbors",
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
