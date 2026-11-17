"""
model.py — Spotify Vibe Space: Data Processing & Unsupervised ML Pipeline
==========================================================================
Loads the Kaggle Spotify dataset, engineers features, trains KMeans (k=6),
applies PCA for 2-D projection, and persists all artifacts so app.py loads
instantly.

Usage:
    python model.py
"""

from __future__ import annotations

import os
import sys
import glob
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR        = Path("data")
ARTIFACTS_DIR   = Path("artifacts")
N_CLUSTERS      = 6
RANDOM_STATE    = 42
N_PCA_COMPONENTS = 2

AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
]

# Column aliases — the pipeline will rename to these canonical names.
COLUMN_ALIASES: dict[str, str] = {
    # common Kaggle variants → canonical
    "track_name":    "track_name",
    "name":          "track_name",
    "song_name":     "track_name",
    "title":         "track_name",
    "track_artist":  "artist_name",
    "artist_name":   "artist_name",
    "artists":       "artist_name",
    "artist":        "artist_name",
    "track_album_name": "album_name",
    "album_name":    "album_name",
    "album":         "album_name",
    "track_popularity": "popularity",
    "popularity":    "popularity",
    "playlist_genre": "genre",
    "genre":         "genre",
    "track_genre":   "genre",
}

# Descriptive vibe labels for each cluster (assigned after training)
VIBE_LABELS = {
    0: "⚡ High Energy & Workout",
    1: "🌙 Late Night Ambient",
    2: "🎹 Melancholic & Acoustic",
    3: "🕺 Dance Floor Bangers",
    4: "🎸 Rock & Intensity",
    5: "😌 Chill & Feel-Good",
}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def find_csv(data_dir: Path) -> Path:
    """Return the first CSV found in *data_dir*."""
    candidates = sorted(data_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV file found in '{data_dir}/'. "
            "Please place your Spotify dataset CSV there and re-run."
        )
    chosen = candidates[0]
    log.info("Found dataset: %s  (%.1f MB)", chosen, chosen.stat().st_size / 1_048_576)
    return chosen


def load_and_normalize(csv_path: Path) -> pd.DataFrame:
    """Load CSV and normalise column names to our canonical set."""
    df = pd.read_csv(csv_path, low_memory=False)
    log.info("Raw shape: %s", df.shape)

    # Lowercase all column names, strip whitespace
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Rename via alias map
    rename_map = {k: v for k, v in COLUMN_ALIASES.items() if k in df.columns}
    df.rename(columns=rename_map, inplace=True)

    # Deduplicate column names that mapped to the same target
    df = df.loc[:, ~df.columns.duplicated()]

    return df


# ---------------------------------------------------------------------------
# Feature Engineering & Cleaning
# ---------------------------------------------------------------------------
def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaN in any audio feature column that is present."""
    present = [f for f in AUDIO_FEATURES if f in df.columns]
    missing = [f for f in AUDIO_FEATURES if f not in df.columns]
    if missing:
        log.warning("Audio features not in dataset: %s", missing)
    before = len(df)
    df = df.dropna(subset=present).reset_index(drop=True)
    log.info("Dropped %d rows with NaN → %d rows remain", before - len(df), len(df))
    return df, present


def ensure_meta_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create placeholder meta columns if they are missing."""
    for col, default in [
        ("track_name", "Unknown Track"),
        ("artist_name", "Unknown Artist"),
        ("album_name",  "Unknown Album"),
        ("popularity",  0),
        ("genre",       "Unknown"),
    ]:
        if col not in df.columns:
            df[col] = default
    # Coerce popularity to numeric
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0).astype(int)
    return df


# ---------------------------------------------------------------------------
# ML Pipeline
# ---------------------------------------------------------------------------
def build_pipeline(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    1. Scale features.
    2. KMeans clustering (k=N_CLUSTERS).
    3. PCA → 2 dimensions.
    Returns artifacts dict.
    """
    X = df[feature_cols].values.astype(np.float32)

    # --- Scaling ---
    log.info("Fitting StandardScaler on %d samples × %d features …", *X.shape)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- KMeans ---
    log.info("Training KMeans (k=%d) …", N_CLUSTERS)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20, max_iter=500)
    labels = kmeans.fit_predict(X_scaled)
    log.info("Cluster sizes: %s", dict(zip(*np.unique(labels, return_counts=True))))

    # Assign dynamic vibe labels (sort clusters by mean energy desc)
    if "energy" in feature_cols:
        energy_idx = feature_cols.index("energy")
        cluster_energy = {
            c: X[labels == c, energy_idx].mean()
            for c in range(N_CLUSTERS)
        }
        sorted_by_energy = sorted(cluster_energy, key=cluster_energy.get, reverse=True)
        static_labels = list(VIBE_LABELS.values())
        final_vibe_map = {cluster_id: static_labels[rank] for rank, cluster_id in enumerate(sorted_by_energy)}
    else:
        final_vibe_map = VIBE_LABELS

    # --- PCA ---
    log.info("Applying PCA (n_components=%d) …", N_PCA_COMPONENTS)
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    log.info(
        "Explained variance: %.1f%%",
        pca.explained_variance_ratio_.sum() * 100,
    )

    return {
        "scaler":    scaler,
        "kmeans":    kmeans,
        "pca":       pca,
        "X_scaled":  X_scaled,
        "labels":    labels,
        "coords":    coords,
        "vibe_map":  final_vibe_map,
    }


# ---------------------------------------------------------------------------
# Recommendation Engine
# ---------------------------------------------------------------------------
def recommend(
    seed_name: str,
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Find the cluster of *seed_name*, compute cosine similarity of its
    scaled feature vector against all tracks in the same cluster, and
    return the top *top_n* matches (excluding itself).
    """
    # Case-insensitive partial match
    mask = df["track_name"].str.lower().str.contains(seed_name.strip().lower(), na=False)
    hits = df[mask]

    if hits.empty:
        raise ValueError(f"No track found matching '{seed_name}'.")

    seed_idx   = hits.index[0]
    seed_name_ = df.loc[seed_idx, "track_name"]
    seed_cluster = df.loc[seed_idx, "cluster"]

    log.info("Seed track : '%s'  (cluster %d)", seed_name_, seed_cluster)

    # Gather all tracks in same cluster
    cluster_mask = df["cluster"] == seed_cluster
    cluster_df   = df[cluster_mask]
    cluster_idxs = cluster_df.index.tolist()

    # Cosine similarity between seed vector and all cluster vectors
    seed_vec     = X_scaled[seed_idx].reshape(1, -1)
    cluster_vecs = X_scaled[cluster_idxs]
    sims         = cosine_similarity(seed_vec, cluster_vecs)[0]

    # Attach scores, exclude seed itself, sort desc
    sim_series = pd.Series(sims, index=cluster_idxs, name="similarity")
    recs = (
        cluster_df
        .join(sim_series)
        .drop(index=seed_idx, errors="ignore")
        .sort_values("similarity", ascending=False)
        .head(top_n)
    )

    result_cols = [c for c in [
        "track_name", "artist_name", "album_name",
        "genre", "popularity", "energy", "danceability",
        "valence", "tempo", "similarity",
    ] if c in recs.columns]

    return recs[result_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_artifacts(
    scaler, kmeans, pca, df: pd.DataFrame, artifacts_dir: Path
):
    artifacts_dir.mkdir(exist_ok=True)
    joblib.dump(scaler, artifacts_dir / "scaler.pkl")
    joblib.dump(kmeans, artifacts_dir / "kmeans.pkl")
    joblib.dump(pca,    artifacts_dir / "pca.pkl")
    df.to_parquet(artifacts_dir / "processed_tracks.parquet", index=False)
    log.info("Artifacts saved to '%s/'", artifacts_dir)


def load_artifacts(artifacts_dir: Path = ARTIFACTS_DIR) -> tuple:
    """Load all saved artifacts; raises FileNotFoundError if missing."""
    scaler = joblib.load(artifacts_dir / "scaler.pkl")
    kmeans = joblib.load(artifacts_dir / "kmeans.pkl")
    pca    = joblib.load(artifacts_dir / "pca.pkl")
    df     = pd.read_parquet(artifacts_dir / "processed_tracks.parquet")
    return scaler, kmeans, pca, df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=" * 60)
    log.info("  Spotify Vibe Space — Model Builder")
    log.info("=" * 60)

    # 1. Load data
    csv_path = find_csv(DATA_DIR)
    df = load_and_normalize(csv_path)
    df = ensure_meta_columns(df)
    df, feature_cols = clean_features(df)

    if df.empty:
        log.error("Dataset is empty after cleaning. Aborting.")
        sys.exit(1)

    # 2. Train ML pipeline
    artefacts = build_pipeline(df, feature_cols)
    scaler    = artefacts["scaler"]
    kmeans    = artefacts["kmeans"]
    pca       = artefacts["pca"]
    X_scaled  = artefacts["X_scaled"]
    labels    = artefacts["labels"]
    coords    = artefacts["coords"]
    vibe_map  = artefacts["vibe_map"]

    # 3. Attach results to dataframe
    df["cluster"]    = labels
    df["vibe_label"] = df["cluster"].map(vibe_map)
    df["pca_x"]      = coords[:, 0]
    df["pca_y"]      = coords[:, 1]

    # 4. Keep feature_cols in df (already there), store metadata
    df.attrs["feature_cols"] = feature_cols
    df.attrs["vibe_map"]     = vibe_map

    # 5. Persist
    save_artifacts(scaler, kmeans, pca, df, ARTIFACTS_DIR)

    log.info("Dataset summary:")
    log.info("  Total tracks : %d", len(df))
    log.info("  Clusters     : %d", N_CLUSTERS)
    log.info("  Feature cols : %s", feature_cols)

    # Quick sanity-check: recommend from first track
    seed = df["track_name"].iloc[0]
    log.info("Sanity check — recommendations for '%s':", seed)
    try:
        recs = recommend(seed, df, X_scaled, top_n=3)
        log.info("\n%s", recs[["track_name", "artist_name", "similarity"]].to_string(index=False))
    except Exception as exc:
        log.warning("Recommendation check failed: %s", exc)

    log.info("=" * 60)
    log.info("  Model build complete! Run:  streamlit run app.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
