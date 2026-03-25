from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, Response, jsonify, request
from joblib import load
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

app = Flask(__name__, template_folder=str(BASE_DIR), static_folder=str(BASE_DIR))

SPOTIFY_MODEL_PATH = Path(
    os.getenv("RECOMMENDER_MODEL_PATH", str(BASE_DIR / "recommendation_model.pkl"))
)
SPOTIFY_DATASET_ZIP_PATH = Path(
    os.getenv("RECOMMENDER_DATASET_ZIP", str(BASE_DIR / "spotify_tracks.zip"))
)
SPOTIFY_DATASET_FILE_NAME = os.getenv("RECOMMENDER_DATASET_FILE", "spotify_tracks.csv")
SPOTIFY_FEATURE_COLUMNS = [
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
MAX_RESULTS = 8


@dataclass(frozen=True)
class TextEngine:
    name: str
    language: str
    dataframe: pd.DataFrame
    features: Any
    model: Any
    vectorizer: Any
    title_column: str
    artist_column: str
    album_column: str
    mood_column: str
    text_column: str | None = None


def safe_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def is_match(value: str, expected: str) -> bool:
    if not expected:
        return True
    return normalize_text(value) == normalize_text(expected)


def row_value(row: pd.Series, column: str) -> str:
    if column not in row.index:
        return ""
    return safe_str(row[column])


def find_song_row(
    songs: pd.DataFrame,
    song_name: str,
    title_column: str,
    popularity_column: str | None = None,
) -> tuple[pd.Series | None, str]:
    normalized = normalize_text(song_name)
    titles = songs[title_column].fillna("").astype(str)
    exact_matches = songs[titles.map(normalize_text) == normalized]
    if not exact_matches.empty:
        return pick_preferred_row(exact_matches, title_column, popularity_column)

    contains_matches = songs[titles.map(normalize_text).str.contains(normalized, na=False)]
    if not contains_matches.empty:
        return pick_preferred_row(contains_matches, title_column, popularity_column)

    name_lookup = titles.drop_duplicates().tolist()
    close = get_close_matches(song_name, name_lookup, n=1, cutoff=0.6)
    if close:
        close_matches = songs[titles == close[0]]
        return pick_preferred_row(close_matches, title_column, popularity_column)

    return None, song_name


def pick_preferred_row(
    matches: pd.DataFrame,
    title_column: str,
    popularity_column: str | None,
) -> tuple[pd.Series, str]:
    if popularity_column and popularity_column in matches.columns:
        row = matches.sort_values(popularity_column, ascending=False).iloc[0]
    else:
        row = matches.iloc[0]
    return row, safe_str(row[title_column])


@lru_cache(maxsize=1)
def load_spotify_assets() -> tuple[Any, pd.DataFrame, StandardScaler]:
    model = load(SPOTIFY_MODEL_PATH)

    with zipfile.ZipFile(SPOTIFY_DATASET_ZIP_PATH) as archive:
        with archive.open(SPOTIFY_DATASET_FILE_NAME) as handle:
            songs = pd.read_csv(handle)

    scaler = StandardScaler()
    scaler.fit(songs[SPOTIFY_FEATURE_COLUMNS])
    return model, songs, scaler


@lru_cache(maxsize=1)
def load_text_engines() -> list[TextEngine]:
    engines: list[TextEngine] = []

    def add_engine(
        *,
        name: str,
        language: str,
        dataframe_file: str,
        features_file: str,
        model_file: str,
        vectorizer_file: str,
        title_column: str,
        artist_column: str,
        album_column: str,
        mood_column: str,
        text_column: str | None = None,
    ) -> None:
        df_path = MODELS_DIR / dataframe_file
        features_path = MODELS_DIR / features_file
        model_path = MODELS_DIR / model_file
        vectorizer_path = MODELS_DIR / vectorizer_file

        if not all(path.exists() for path in [df_path, features_path, model_path, vectorizer_path]):
            return

        dataframe = load(df_path).copy()
        features = load(features_path)
        model = load(model_path)
        vectorizer = load(vectorizer_path)

        usable_rows = min(len(dataframe), features.shape[0])
        dataframe = dataframe.iloc[:usable_rows].reset_index(drop=True)
        features = features[:usable_rows]

        if usable_rows == 0 or not hasattr(model, "kneighbors"):
            return

        engines.append(
            TextEngine(
                name=name,
                language=language,
                dataframe=dataframe,
                features=features,
                model=model,
                vectorizer=vectorizer,
                title_column=title_column,
                artist_column=artist_column,
                album_column=album_column,
                mood_column=mood_column,
                text_column=text_column,
            )
        )

    add_engine(
        name="spotify-audio-knn",
        language="multilingual",
        dataframe_file="songs_df.pkl",
        features_file="features.pkl",
        model_file="knn_recommender.pkl",
        vectorizer_file="vectorizer.pkl",
        title_column="name",
        artist_column="movie",
        album_column="movie",
        mood_column="mood",
        text_column="combined",
    )
    add_engine(
        name="spotify-audio-classifier",
        language="multilingual",
        dataframe_file="songs_df.pkl",
        features_file="features.pkl",
        model_file="knn_model.pkl",
        vectorizer_file="vectorizer.pkl",
        title_column="name",
        artist_column="movie",
        album_column="movie",
        mood_column="mood",
        text_column="combined",
    )
    add_engine(
        name="telugu-nearest-neighbors",
        language="telugu",
        dataframe_file="telugu_df.pkl",
        features_file="telugu_features.pkl",
        model_file="telugu_recommender.pkl",
        vectorizer_file="telugu_vectorizer.pkl",
        title_column="Song Name",
        artist_column="Artist",
        album_column="Movie",
        mood_column="mood",
    )
    add_engine(
        name="telugu-knn-classifier",
        language="telugu",
        dataframe_file="telugu_df.pkl",
        features_file="telugu_features.pkl",
        model_file="telugu_knn_model.pkl",
        vectorizer_file="telugu_vectorizer.pkl",
        title_column="Song Name",
        artist_column="Artist",
        album_column="Movie",
        mood_column="mood",
    )
    add_engine(
        name="telugu-knn-model",
        language="telugu",
        dataframe_file="telugu_df.pkl",
        features_file="telugu_features.pkl",
        model_file="telugu_model.pkl",
        vectorizer_file="telugu_vectorizer.pkl",
        title_column="Song Name",
        artist_column="Artist",
        album_column="Movie",
        mood_column="mood",
    )
    return engines


def build_spotify_recommendations(song_name: str) -> tuple[str, list[dict[str, str]], str]:
    model, songs, scaler = load_spotify_assets()
    row, resolved_name = find_song_row(
        songs=songs,
        song_name=song_name,
        title_column="track_name",
        popularity_column="popularity",
    )

    if row is None:
        return resolved_name, [], "spotify-feature-engine"

    song_features = row[SPOTIFY_FEATURE_COLUMNS].to_frame().T
    scaled_features = scaler.transform(song_features)
    _, indices = model.kneighbors(scaled_features, n_neighbors=min(MAX_RESULTS + 1, len(songs)))

    recommendations: list[dict[str, str]] = []
    seed_track_id = safe_str(row["track_id"])

    for idx in indices[0]:
        rec = songs.iloc[int(idx)]
        if safe_str(rec["track_id"]) == seed_track_id:
            continue

        recommendations.append(
            {
                "title": safe_str(rec["track_name"]),
                "artist": safe_str(rec["artist_name"]),
                "album": safe_str(rec["album_name"]),
                "year": safe_str(rec["year"]),
                "popularity": safe_str(rec["popularity"]),
                "language": safe_str(rec.get("language", "multilingual")),
                "artwork_url": safe_str(rec.get("artwork_url", "")),
                "track_url": safe_str(rec.get("track_url", "")),
                "engine": "spotify-feature-engine",
                "genre": "",
            }
        )

    return resolved_name, recommendations[:MAX_RESULTS], "spotify-feature-engine"


def build_text_engine_recommendations(engine: TextEngine, song_name: str) -> tuple[str, list[dict[str, str]]]:
    row, resolved_name = find_song_row(
        songs=engine.dataframe,
        song_name=song_name,
        title_column=engine.title_column,
    )

    query_vector = None
    seed_index: int | None = None

    if row is not None:
        seed_index = int(row.name)
        query_vector = engine.features[seed_index]
    elif engine.vectorizer is not None:
        text_query = song_name
        if engine.text_column and engine.text_column in engine.dataframe.columns:
            text_query = f"{song_name} {song_name}"
        query_vector = engine.vectorizer.transform([text_query])
    else:
        return resolved_name, []

    _, indices = engine.model.kneighbors(
        query_vector,
        n_neighbors=min(MAX_RESULTS + 1, len(engine.dataframe)),
    )

    recommendations: list[dict[str, str]] = []
    for idx in indices[0]:
        rec_index = int(idx)
        if seed_index is not None and rec_index == seed_index:
            continue

        rec = engine.dataframe.iloc[rec_index]
        recommendations.append(
            {
                "title": row_value(rec, engine.title_column),
                "artist": row_value(rec, engine.artist_column),
                "album": row_value(rec, engine.album_column),
                "year": "",
                "popularity": "",
                "language": engine.language,
                "artwork_url": "",
                "track_url": "",
                "engine": engine.name,
                "genre": row_value(rec, engine.mood_column),
            }
        )

    return resolved_name, recommendations[:MAX_RESULTS]


def filter_and_rank_recommendations(
    items: list[dict[str, str]],
    genre: str,
    language: str,
) -> list[dict[str, str]]:
    scored: list[tuple[int, dict[str, str]]] = []
    requested_genre = normalize_text(genre)
    requested_language = normalize_text(language)

    for item in items:
        item_language = normalize_text(safe_str(item.get("language", "")))
        item_genre = normalize_text(safe_str(item.get("genre", "")))

        score = 0
        if requested_language and item_language == requested_language:
            score += 3
        elif requested_language and item_language in {"multilingual", "unknown", ""}:
            score += 1

        if requested_genre and item_genre == requested_genre:
            score += 3

        if requested_language and item_language not in {requested_language, "multilingual", "unknown", ""}:
            continue

        scored.append((score, item))

    scored.sort(
        key=lambda entry: (
            -entry[0],
            normalize_text(safe_str(entry[1].get("title", ""))),
        )
    )
    return [item for _, item in scored[:MAX_RESULTS]]


def combine_recommendations(song_name: str, genre: str, language: str) -> tuple[str, list[dict[str, str]], str, list[str]]:
    resolved_name = song_name
    engines_used: list[str] = []
    combined: list[dict[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    query_key = normalize_text(song_name)

    spotify_name, spotify_results, spotify_engine_name = build_spotify_recommendations(song_name)
    if spotify_results:
        resolved_name = spotify_name
        engines_used.append(spotify_engine_name)
        merge_recommendations(combined, seen_keys, spotify_results, query_key)

    for engine in load_text_engines():
        engine_name, engine_results = build_text_engine_recommendations(engine, song_name)
        if engine_results:
            if resolved_name == song_name:
                resolved_name = engine_name
            engines_used.append(engine.name)
            merge_recommendations(combined, seen_keys, engine_results, query_key)

    filtered = filter_and_rank_recommendations(combined, genre, language)
    source = f"{len(engines_used)} engines combined" if engines_used else "no-engines-available"
    return resolved_name, filtered, source, engines_used


def merge_recommendations(
    target: list[dict[str, str]],
    seen_keys: set[tuple[str, str]],
    items: list[dict[str, str]],
    query_key: str,
) -> None:
    for item in items:
        key = (
            normalize_text(safe_str(item.get("title", ""))),
            normalize_text(safe_str(item.get("artist", ""))),
        )
        if not key[0] or key[0] == query_key or key in seen_keys:
            continue
        seen_keys.add(key)
        target.append(item)


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
                    "engine": "",
                    "genre": "",
                }
            )
            continue

        if isinstance(item, dict):
            normalized.append(
                {
                    "title": safe_str(item.get("title") or item.get("song") or "Untitled"),
                    "artist": safe_str(item.get("artist") or "Unknown Artist"),
                    "album": safe_str(item.get("album") or "Recommended for you"),
                    "year": safe_str(item.get("year") or ""),
                    "popularity": safe_str(item.get("popularity") or ""),
                    "language": safe_str(item.get("language") or ""),
                    "artwork_url": safe_str(item.get("artwork_url") or ""),
                    "track_url": safe_str(item.get("track_url") or ""),
                    "engine": safe_str(item.get("engine") or ""),
                    "genre": safe_str(item.get("genre") or item.get("mood") or ""),
                }
            )

    return normalized


@app.get("/")
def home() -> Response:
    html = (BASE_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace(
        "{{ url_for('static', filename='styles.css', v='20260325a') }}",
        f"{app.static_url_path}/styles.css?v=20260325a",
    )
    html = html.replace(
        "{{ url_for('static', filename='background.mp4') }}",
        f"{app.static_url_path}/background.mp4",
    )
    html = html.replace(
        "{{ url_for('static', filename='app.js', v='20260325a') }}",
        f"{app.static_url_path}/app.js?v=20260325a",
    )
    return Response(html, mimetype="text/html")


@app.post("/api/recommend")
def recommend():
    data = request.get_json(silent=True) or {}
    genre = safe_str(data.get("genre", ""))
    language = safe_str(data.get("language", ""))
    song_name = safe_str(data.get("song_name", ""))

    if not genre or not song_name:
        return jsonify({"error": "Please enter genre and song name."}), 400

    try:
        resolved_query, raw_results, source, engines_used = combine_recommendations(song_name, genre, language)
    except FileNotFoundError as exc:
        return jsonify({"error": f"Backend file not found: {exc.filename}"}), 500
    except Exception as exc:
        return jsonify({"error": f"Recommendation backend failed: {exc}"}), 500

    results = normalize_results(raw_results)
    return jsonify(
        {
            "query": resolved_query,
            "recommendations": results,
            "source": source,
            "engine_count": len(engines_used),
            "engines_used": engines_used,
            "filters": {"genre": genre, "language": language},
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
