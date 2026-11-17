"""
app.py — Spotify Vibe Space: Planetarium UI
============================================
A soothing, dark-mode "planetarium" dashboard for exploring audio clusters and
generating smart playlists from a Spotify audio-feature dataset.

The machine-learning pipeline (StandardScaler, KMeans, PCA, cosine similarity)
and every recommendation / Spotify helper are unchanged — this module only
refactors the presentation layer.

Usage (after running model.py):
    streamlit run app.py
"""

from __future__ import annotations

import html
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — MUST be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spotify Vibe Space",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Palette & theme constants
# ---------------------------------------------------------------------------
BG_DARK        = "#0a0b10"
CARD_BG        = "#12141e"
CARD_BORDER    = "rgba(255,255,255,.08)"
TEXT_PRIMARY   = "#f4f6ff"
TEXT_SECONDARY = "#98a0b8"

# Signature palette per vibe label: (primary, secondary)
VIBE_PALETTE: dict[str, tuple[str, str]] = {
    "⚡ High Energy & Workout":   ("#22d3ee", "#0ea5e9"),   # electric cyan
    "🌙 Late Night Ambient":      ("#6366f1", "#8b5cf6"),   # midnight indigo
    "🎹 Melancholic & Acoustic":  ("#f59e0b", "#f97316"),   # warm amber
    "🕺 Dance Floor Bangers":     ("#ec4899", "#f43f5e"),   # magenta
    "🎸 Rock & Intensity":        ("#ef4444", "#f97316"),   # crimson
    "😌 Chill & Feel-Good":       ("#10b981", "#22d3ee"),   # emerald
}
DEFAULT_ACCENT = ("#8b5cf6", "#22d3ee")
CONTINUOUS_SCALE = ["#111327", "#6366f1", "#22d3ee", "#f8fafc"]


def _palette_for_vibe(vibe_label) -> tuple[str, str]:
    """Signature colour pair for a vibe label (falls back to the default)."""
    return VIBE_PALETTE.get(str(vibe_label), DEFAULT_ACCENT)


def _vibe_label_for_cluster(df: pd.DataFrame, cluster_id) -> str:
    """Resolve the human vibe label for a KMeans cluster id."""
    try:
        cid = int(cluster_id)
    except (TypeError, ValueError):
        return "All Vibes"
    sub = df[df["cluster"] == cid]
    if not sub.empty:
        return str(sub["vibe_label"].iloc[0])
    return f"Cluster {cid}"


def _resolve_accent(df: pd.DataFrame, selected_vibe: str, accent_cluster) -> tuple[str, str]:
    """Pick the ambient accent from an explicit cluster, else the selected vibe."""
    if accent_cluster is not None:
        sub = df[df["cluster"] == accent_cluster] if str(accent_cluster).lstrip("-").isdigit() else None
        if sub is not None and not sub.empty:
            return _palette_for_vibe(str(sub["vibe_label"].iloc[0]))
    if selected_vibe and selected_vibe != "All Vibes":
        return _palette_for_vibe(selected_vibe)
    return DEFAULT_ACCENT


# ---------------------------------------------------------------------------
# Theme CSS (tokens replaced per-run for dynamic colour reaction)
# ---------------------------------------------------------------------------
THEME_CSS = """
<style>
:root{
  --accent: __ACCENT__;
  --accent-2: __ACCENT2__;
  --accent-soft: __ACCENT_SOFT__;
  --accent-glow: __ACCENT_GLOW__;
  --accent-faint: __ACCENT_FAINT__;
  --bg: #0a0b10;
  --border: rgba(255,255,255,.08);
  --text: #f4f6ff;
  --muted: #98a0b8;
}

html, body, [class*="st-"]{
  font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
}

.stApp{
  background-color: var(--bg);
  background-image:
    radial-gradient(1px 1px at 25px 35px, rgba(255,255,255,.55), transparent),
    radial-gradient(1px 1px at 120px 80px, rgba(255,255,255,.35), transparent),
    radial-gradient(1.5px 1.5px at 200px 160px, rgba(255,255,255,.50), transparent),
    radial-gradient(1px 1px at 300px 240px, rgba(255,255,255,.30), transparent),
    radial-gradient(1100px 720px at 50% -12%, __ACCENT_FAINT__ 0%, transparent 62%),
    radial-gradient(900px 620px at 88% 6%, __ACCENT_FAINT__ 0%, transparent 58%),
    radial-gradient(820px 620px at 6% 92%, __ACCENT_FAINT__ 0%, transparent 60%);
  background-size: 350px 350px, 350px 350px, 350px 350px, 350px 350px, cover, cover, cover;
  background-repeat: repeat, repeat, repeat, repeat, no-repeat, no-repeat, no-repeat;
  background-attachment: fixed;
  color: var(--text);
}

#MainMenu, footer, header{ visibility: hidden; }
body{ background-color: #0a0b10; }
hr{ border-color: var(--border) !important; margin: 1.1rem 0 !important; }
a{ color: var(--accent); text-decoration: none; }
a:hover{ text-decoration: underline; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(8,9,14,.97), rgba(11,13,20,.99));
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h2{
  color: var(--accent);
  letter-spacing: .14em;
  text-transform: uppercase;
  font-size: .86rem;
  font-weight: 800;
}
[data-testid="stSidebar"] hr{ border-color: var(--border); }

/* ---------- Hero ---------- */
.hero{ padding: 1.4rem 0 .4rem 0; animation: rise .7s cubic-bezier(.2,.7,.2,1) both; }
.hero-kicker{
  font-size: .7rem; letter-spacing: .34em; text-transform: uppercase;
  color: var(--accent); font-weight: 800;
}
.hero-title{
  font-size: clamp(2.4rem, 6vw, 4.2rem);
  line-height: 1.02; font-weight: 800; letter-spacing: -.045em;
  background: linear-gradient(100deg, #ffffff 8%, var(--accent) 52%, var(--accent-2) 100%);
  background-size: 200% auto;
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: shimmer 7s linear infinite;
  margin: .3rem 0 .5rem 0;
}
.hero-sub{ color: var(--muted); font-size: 1.02rem; line-height: 1.65; max-width: 780px; margin: 0; }
.badges{ display: flex; gap: .55rem; flex-wrap: wrap; margin-top: 1.05rem; }
.badge{
  padding: .45rem .85rem; border-radius: 999px; border: 1px solid var(--border);
  background: rgba(255,255,255,.04); color: var(--muted); font-size: .78rem;
  backdrop-filter: blur(8px);
}
.badge b{ color: var(--text); font-size: .92rem; }

/* ---------- Section headers ---------- */
.section-head{ margin: 1.2rem 0 .7rem 0; }
.section-kicker{
  font-size: .66rem; letter-spacing: .3em; text-transform: uppercase;
  color: var(--accent); font-weight: 800;
}
.section-title{
  font-size: 1.4rem; font-weight: 800; letter-spacing: -.02em;
  color: var(--text); margin-top: .15rem;
}
.section-sub{ color: var(--muted); font-size: .86rem; margin-top: .2rem; line-height: 1.5; }

/* ---------- Vibe orb cards ---------- */
.vibe-orb-card{
  position: relative; display: flex; align-items: center; gap: .75rem;
  padding: .8rem .85rem; border-radius: 18px; min-height: 72px;
  border: 1px solid var(--border);
  background: linear-gradient(150deg, rgba(255,255,255,.06), rgba(255,255,255,.015));
  backdrop-filter: blur(12px);
  box-shadow: 0 10px 28px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.05);
  transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
  overflow: hidden;
}
.vibe-orb-card::after{
  content: ""; position: absolute; right: -30px; top: -40px; width: 120px; height: 120px;
  border-radius: 50%; background: radial-gradient(circle, var(--c) 0%, transparent 70%);
  opacity: .16; filter: blur(4px); pointer-events: none;
}
.vibe-orb-card:hover{
  transform: translateY(-3px); border-color: var(--c);
  box-shadow: 0 16px 38px rgba(0,0,0,.5), 0 0 24px var(--c33);
}
.vibe-orb-card.sel{
  border-color: var(--c);
  box-shadow: 0 0 0 1px var(--c), 0 0 30px var(--c33);
  background: linear-gradient(150deg, var(--c33), rgba(255,255,255,.02));
}
.orb{
  width: 38px; height: 38px; flex: 0 0 38px; border-radius: 50%; position: relative;
  background: radial-gradient(circle at 32% 28%, #ffffff 0%, var(--c) 40%, rgba(0,0,0,.9) 125%);
  box-shadow: 0 0 20px var(--c), inset -5px -6px 12px rgba(0,0,0,.55);
  animation: floaty 5s ease-in-out infinite;
}
.orb::after{
  content: ""; position: absolute; inset: -7px; border-radius: 50%;
  border: 1px solid var(--c33);
}
.vibe-orb-meta{ min-width: 0; }
.vibe-orb-name{ font-weight: 700; font-size: .82rem; color: var(--text); line-height: 1.25; }
.vibe-orb-count{ font-size: .72rem; color: var(--muted); margin-top: .12rem; }

/* ---------- Buttons ---------- */
.stButton > button{
  background: rgba(255,255,255,.05);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 999px;
  padding: .42rem .85rem;
  font-weight: 600;
  font-size: .8rem;
  transition: all .18s ease;
  backdrop-filter: blur(8px);
}
.stButton > button:hover{
  border-color: var(--accent);
  box-shadow: 0 0 18px var(--accent-glow);
  transform: translateY(-1px);
  color: #fff;
}
.stButton > button:disabled{
  opacity: .65;
  color: var(--accent);
  border-color: var(--accent);
  background: var(--accent-soft);
}

/* ---------- Inputs ---------- */
.stTextInput input{
  background: rgba(255,255,255,.05) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--text) !important;
  padding: .55rem .8rem !important;
}
.stTextInput input:focus{
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-soft) !important;
}
.stSelectbox div[data-baseweb="select"] > div{
  background: rgba(255,255,255,.05) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}
[data-testid="stSelectbox"] label,
[data-testid="stTextInput"] label,
[data-testid="stSlider"] label{
  color: var(--muted) !important; font-size: .8rem !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"]{
  background: var(--accent) !important;
  box-shadow: 0 0 12px var(--accent-glow) !important;
}
.stSlider [data-baseweb="slider"] div[data-testid*="track"]{
  background: rgba(255,255,255,.12) !important;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"]{
  background: rgba(255,255,255,.03);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: .3rem;
  gap: .3rem;
}
.stTabs [data-baseweb="tab"]{
  border-radius: 999px;
  padding: .5rem 1rem;
  color: var(--muted);
  background: transparent;
  border: none;
  font-weight: 600;
}
.stTabs [aria-selected="true"]{
  background: var(--accent-soft) !important;
  color: var(--text) !important;
  box-shadow: inset 0 0 0 1px var(--accent-glow);
}

/* ---------- Glass containers / metrics ---------- */
[data-testid="stVerticalBlockBorderWrapper"]{
  background: linear-gradient(160deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
  border: 1px solid var(--border) !important;
  border-radius: 20px;
  backdrop-filter: blur(14px);
  box-shadow: 0 18px 50px rgba(0,0,0,.4), inset 0 1px 0 rgba(255,255,255,.05);
  padding: 1rem 1.1rem;
}
[data-testid="stMetric"]{
  background: linear-gradient(160deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: .9rem 1.1rem;
  backdrop-filter: blur(12px);
  box-shadow: 0 12px 30px rgba(0,0,0,.35);
}
[data-testid="stMetricLabel"]{
  color: var(--muted) !important; font-size: .72rem !important;
  letter-spacing: .08em; text-transform: uppercase;
}
[data-testid="stMetricValue"]{ color: var(--text) !important; font-weight: 800; }
[data-testid="stMetricDelta"]{ color: var(--accent) !important; }

/* ---------- Progress (match bars) ---------- */
[data-testid="stProgress"] div[role="progressbar"]{
  height: 8px; border-radius: 999px;
  background: rgba(255,255,255,.08); overflow: hidden;
}
[data-testid="stProgress"] div[role="progressbar"] > div{
  background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
  border-radius: 999px;
  box-shadow: 0 0 12px var(--accent-glow);
}

/* ---------- Dataframe ---------- */
[data-testid="stDataFrame"]{
  border-radius: 16px;
  border: 1px solid var(--border);
  overflow: hidden;
}

/* ---------- Chips / tags / rec cards ---------- */
.tag{
  display: inline-block; padding: .16rem .55rem; border-radius: 999px;
  border: 1px solid var(--border); background: rgba(255,255,255,.04);
  color: var(--muted); font-size: .72rem; margin: .12rem .25rem .12rem 0;
}
.chip{
  display: inline-block; padding: .2rem .58rem; border-radius: 999px;
  background: rgba(255,255,255,.06); border: 1px solid var(--border);
  font-size: .72rem; color: var(--text); margin: .14rem .24rem .14rem 0;
}
.rec-title{ font-weight: 800; font-size: 1rem; color: var(--text); line-height: 1.25; }
.rec-artist{ color: var(--accent); font-size: .82rem; font-weight: 600; margin-top: .1rem; }
.rec-album{ color: var(--muted); font-size: .74rem; margin-bottom: .35rem; }
.match-row{
  display: flex; justify-content: space-between; align-items: baseline;
  font-size: .74rem; color: var(--muted); margin: .45rem 0 .25rem;
}
.match-row b{ color: var(--text); font-size: .82rem; }

.seed-orb{
  width: 100%; aspect-ratio: 1; max-height: 180px; border-radius: 20px;
  display: flex; align-items: center; justify-content: center; font-size: 3rem;
  background:
    radial-gradient(circle at 50% 42%, var(--c33), transparent 70%),
    radial-gradient(circle at 32% 28%, rgba(255,255,255,.9), var(--c) 42%, rgba(0,0,0,.9) 130%);
  box-shadow: 0 0 40px var(--c33), inset -8px -10px 24px rgba(0,0,0,.55);
  animation: floaty 6s ease-in-out infinite;
}
.badge-live{
  display: inline-block; padding: .2rem .7rem; border-radius: 999px;
  font-size: .7rem; font-weight: 800; letter-spacing: .05em;
}

/* ---------- Sidebar legend ---------- */
.legend-row{ display: flex; gap: .55rem; align-items: flex-start; margin-bottom: .6rem; }
.dot{
  flex: 0 0 10px; width: 10px; height: 10px; margin-top: .25rem;
  border-radius: 50%; background: var(--c); box-shadow: 0 0 10px var(--c);
}
.legend-name{ font-size: .8rem; font-weight: 700; color: var(--text); }
.legend-desc{ font-size: .72rem; color: var(--muted); line-height: 1.4; }

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar{ width: 8px; height: 8px; }
::-webkit-scrollbar-track{ background: transparent; }
::-webkit-scrollbar-thumb{ background: rgba(255,255,255,.14); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover{ background: var(--accent); }

/* ---------- Animations ---------- */
@keyframes shimmer{ to{ background-position: 200% center; } }
@keyframes rise{ from{ opacity: 0; transform: translateY(14px); } to{ opacity: 1; transform: none; } }
@keyframes floaty{ 0%,100%{ transform: translateY(0); } 50%{ transform: translateY(-6px); } }
</style>
"""


def inject_theme(accent: str, accent2: str) -> None:
    """Inject the global theme, tinted with the active vibe palette."""
    css = (
        THEME_CSS
        .replace("__ACCENT2__", accent2)
        .replace("__ACCENT_SOFT__", accent + "24")
        .replace("__ACCENT_GLOW__", accent + "66")
        .replace("__ACCENT_FAINT__", accent + "18")
        .replace("__ACCENT__", accent)
    )
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = Path("artifacts")

VIBE_DESCRIPTIONS: dict[str, str] = {
    "⚡ High Energy & Workout":    "Pumped-up beats to fuel your training sessions.",
    "🌙 Late Night Ambient":       "Atmospheric, dreamy soundscapes for quiet hours.",
    "🎹 Melancholic & Acoustic":   "Stripped-back, emotional tracks for introspection.",
    "🕺 Dance Floor Bangers":      "Irresistible grooves that demand you move.",
    "🎸 Rock & Intensity":         "Raw guitar energy and powerful dynamics.",
    "😌 Chill & Feel-Good":        "Warm, uplifting vibes for a relaxed mood.",
}

AUDIO_FEATURE_KEYS = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
]

N_CLUSTERS = 6
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="🪐  Charting the vibe universe …")
def load_artifacts():
    """Load pre-built ML artifacts from disk."""
    try:
        import joblib
        scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
        kmeans = joblib.load(ARTIFACTS_DIR / "kmeans.pkl")
        pca    = joblib.load(ARTIFACTS_DIR / "pca.pkl")
        df     = pd.read_parquet(ARTIFACTS_DIR / "processed_tracks.parquet")
        return scaler, kmeans, pca, df
    except FileNotFoundError:
        return None, None, None, None


@st.cache_data(show_spinner=False)
def get_scaled_features(_scaler, df: pd.DataFrame) -> np.ndarray:
    """Re-compute scaled feature matrix (cached)."""
    present = [f for f in AUDIO_FEATURE_KEYS if f in df.columns]
    return _scaler.transform(df[present].values.astype("float32")), present


# ---------------------------------------------------------------------------
# Recommendation logic (pure Python, no joblib needed here)
# ---------------------------------------------------------------------------
def recommend_tracks(
    seed_name: str,
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    top_n: int = 10,
) -> pd.DataFrame:
    from sklearn.metrics.pairwise import cosine_similarity

    mask = df["track_name"].str.lower().str.contains(
        seed_name.strip().lower(), na=False, regex=False
    )
    hits = df[mask]

    if hits.empty:
        raise ValueError(f"No track found matching **{seed_name}**.")

    seed_idx     = hits.index[0]
    seed_cluster = df.loc[seed_idx, "cluster"]

    cluster_mask = df["cluster"] == seed_cluster
    cluster_df   = df[cluster_mask]
    cluster_idxs = cluster_df.index.tolist()

    seed_vec     = X_scaled[seed_idx].reshape(1, -1)
    cluster_vecs = X_scaled[cluster_idxs]
    sims         = cosine_similarity(seed_vec, cluster_vecs)[0]

    sim_series = pd.Series(sims, index=cluster_idxs, name="similarity")
    recs = (
        cluster_df
        .join(sim_series)
        .drop(index=seed_idx, errors="ignore")
        .sort_values("similarity", ascending=False)
        .head(top_n)
    )

    display_cols = [c for c in [
        "track_name", "artist_name", "album_name",
        "genre", "popularity", "energy", "danceability",
        "valence", "tempo", "similarity",
    ] if c in recs.columns]

    return (
        recs[display_cols]
        .reset_index(drop=True)
        .assign(**{
            "similarity": lambda d: (d["similarity"] * 100).round(1)
        })
    )


# ---------------------------------------------------------------------------
# Type-ahead suggestions (alphabetical + popularity ranked)
# ---------------------------------------------------------------------------
def _build_name_index(df: pd.DataFrame) -> list[tuple[str, str, int]]:
    """Return (lowercase_name, original_name, popularity) for fast type-ahead."""
    if "popularity" in df.columns:
        pop = df.groupby("track_name")["popularity"].max()
    else:
        pop = None
    uniq = df["track_name"].dropna().astype(str).unique()
    index = [
        (n.lower(), n, int(pop.get(n, 0)) if pop is not None else 0)
        for n in uniq
    ]
    index.sort(key=lambda x: x[0])
    return index


def _suggest(
    index: list[tuple[str, str, int]], typed: str, limit: int = 12
) -> list[str]:
    """
    Type-ahead suggestions for *typed*.

    Prefix matches rank first by popularity (so hits like "Blinding Lights"
    surface early), then alphabetically; "contains" matches follow.
    """
    t = typed.strip().lower()
    if not t:
        return []
    starts = [x for x in index if x[0].startswith(t)]
    contains = [x for x in index if t in x[0] and not x[0].startswith(t)]
    starts.sort(key=lambda x: (-x[2], x[0]))
    contains.sort(key=lambda x: (-x[2], x[0]))
    ordered = starts + contains
    return [orig for _, orig, _ in ordered[:limit]]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _build_radar(feats: dict, label: str = "Audio Features", color: str = DEFAULT_ACCENT[0]) -> go.Figure:
    """Build a polar radar chart for a set of [0,1]-scaled audio features."""
    display_keys = [
        "energy", "danceability", "valence",
        "acousticness", "speechiness", "liveness",
    ]
    vals = [float(feats.get(k, 0) or 0) for k in display_keys]
    vals_circ = vals + [vals[0]]
    cats_circ = display_keys + [display_keys[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_circ, theta=cats_circ,
        fill="toself",
        fillcolor=color + "2e",
        line=dict(color=color, width=2),
        name=label,
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            angularaxis=dict(
                tickfont_color=TEXT_SECONDARY,
                linecolor="rgba(255,255,255,.10)",
                gridcolor="rgba(255,255,255,.08)",
            ),
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickfont_color=TEXT_SECONDARY,
                gridcolor="rgba(255,255,255,.08)",
                linecolor="rgba(255,255,255,.08)",
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=TEXT_PRIMARY,
        height=250,
        margin=dict(l=50, r=50, t=20, b=20),
        showlegend=False,
    )
    return fig


def _dark_layout(accent: str, height: int = 560) -> dict:
    """Shared transparent Plotly layout for the space theme."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_PRIMARY, family="Inter, sans-serif", size=12),
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        hoverlabel=dict(
            bgcolor="rgba(12,14,22,.96)",
            bordercolor=accent,
            font=dict(color=TEXT_PRIMARY, size=12),
        ),
    )


# ---------------------------------------------------------------------------
# Reusable render helpers
# ---------------------------------------------------------------------------
def render_hero(total_tracks: int, n_clusters: int, n_genres, n_artists) -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-kicker">Spotify Vibe Space</div>
          <div class="hero-title">Navigate your sound.</div>
          <p class="hero-sub">
            An interactive planetarium of your music — six luminous vibe clusters,
            a 2-D map of every track, and instant sonic recommendations powered by
            KMeans, PCA &amp; cosine similarity.
          </p>
          <div class="badges">
            <span class="badge"><b>{total_tracks:,}</b> tracks mapped</span>
            <span class="badge"><b>{n_clusters}</b> active clusters</span>
            <span class="badge"><b>{n_genres}</b> genres</span>
            <span class="badge"><b>{n_artists:,}</b> artists</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(kicker: str, title: str, sub: str = "") -> None:
    sub_html = f'<div class="section-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="section-head">
          <div class="section-kicker">{kicker}</div>
          <div class="section-title">{title}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_vibe_selector(df: pd.DataFrame, all_vibes: list[str], selected_vibe: str) -> None:
    """Interactive grid of orbiting vibe cards (replaces the dropdown)."""
    render_section(
        "Vibe Explorer",
        "Choose your orbit",
        "Each sphere is a cluster. Select one to retune the entire observatory.",
    )

    options: list[tuple[str, int, str]] = [("All Vibes", len(df), DEFAULT_ACCENT[0])]
    for vibe in all_vibes:
        sub = df[df["vibe_label"] == vibe]
        options.append((vibe, len(sub), _palette_for_vibe(vibe)[0]))

    cols = st.columns(3)
    for i, (vibe, count, color) in enumerate(options):
        with cols[i % 3]:
            is_sel = vibe == selected_vibe
            label = html.escape(vibe)
            st.markdown(
                f"""
                <div class="vibe-orb-card{' sel' if is_sel else ''}"
                     style="--c:{color}; --c33:{color}33;">
                  <div class="orb"></div>
                  <div class="vibe-orb-meta">
                    <div class="vibe-orb-name">{label}</div>
                    <div class="vibe-orb-count">{count:,} tracks</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "Selected ✓" if is_sel else "Select orbit",
                key=f"vibe_sel_{i}",
                disabled=is_sel,
                use_container_width=True,
            ):
                st.session_state["selected_vibe"] = vibe
                st.session_state.pop("accent_cluster", None)
                st.rerun()


def render_rec_card(rec: dict, accent: str) -> None:
    """A glass card for one recommendation with a sonic-match progress bar."""
    name   = html.escape(str(rec.get("track_name", "—")))
    artist = html.escape(str(rec.get("artist_name", "—")))
    album  = html.escape(str(rec.get("album_name", "—")))
    try:
        match = float(rec.get("similarity", 0) or 0)
    except (TypeError, ValueError):
        match = 0.0

    chips = []
    if pd.notna(rec.get("energy", None)):
        chips.append(f"<span class='chip'>⚡ {float(rec['energy']):.2f}</span>")
    if pd.notna(rec.get("danceability", None)):
        chips.append(f"<span class='chip'>🕺 {float(rec['danceability']):.2f}</span>")
    if pd.notna(rec.get("valence", None)):
        chips.append(f"<span class='chip'>😊 {float(rec['valence']):.2f}</span>")
    if pd.notna(rec.get("tempo", None)):
        chips.append(f"<span class='chip'>🥁 {float(rec['tempo']):.0f} BPM</span>")
    if pd.notna(rec.get("popularity", None)):
        chips.append(f"<span class='chip'>★ {int(rec['popularity'])}</span>")

    with st.container(border=True):
        st.markdown(f"<div class='rec-title'>{name}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='rec-artist'>{artist}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='rec-album'>{album}</div>", unsafe_allow_html=True)
        if chips:
            st.markdown("".join(chips), unsafe_allow_html=True)
        st.markdown(
            f"<div class='match-row'><span>Sonic match</span><b>{match:.1f}%</b></div>",
            unsafe_allow_html=True,
        )
        st.progress(max(0.0, min(1.0, match / 100.0)))


def render_rec_grid(recs: pd.DataFrame, accent: str) -> None:
    """Two-column card grid of recommendations."""
    records = recs.to_dict("records")
    for i in range(0, len(records), 2):
        row_cols = st.columns(2)
        for j, rec in enumerate(records[i:i + 2]):
            with row_cols[j]:
                render_rec_card(rec, accent)



def render_mini_scatter(
    df: pd.DataFrame,
    pca,
    seed_vec_scaled: np.ndarray,
    seed_label: str,
    cluster_df: pd.DataFrame,
    rec_names: list,
    accent: str,
) -> None:
    """Small PCA projection with the query track highlighted."""
    bg_pts = cluster_df.sample(min(500, len(cluster_df)), random_state=7)
    rec_pts = df[df["track_name"].isin(rec_names)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bg_pts["pca_x"], y=bg_pts["pca_y"],
        mode="markers",
        marker=dict(size=4, color="rgba(255,255,255,.18)"),
        name="Cluster",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=rec_pts["pca_x"], y=rec_pts["pca_y"],
        mode="markers",
        marker=dict(size=9, color=accent, opacity=0.9, line=dict(width=0)),
        name="Similar tracks",
        text=rec_pts.get("track_name"),
        hovertemplate="%{text}<extra></extra>",
    ))
    new_pca = pca.transform(seed_vec_scaled)
    fig.add_trace(go.Scatter(
        x=[new_pca[0, 0]], y=[new_pca[0, 1]],
        mode="markers+text",
        marker=dict(size=16, color="#ffffff", symbol="star"),
        name="Your track",
        text=[f"⭐ {seed_label}"],
        textposition="top center",
        textfont=dict(color=TEXT_PRIMARY, size=11),
    ))
    fig.update_layout(
        **_dark_layout(accent, height=360),
        xaxis=dict(title="PCA Dim 1", showgrid=True, gridcolor="rgba(255,255,255,.06)", zeroline=False),
        yaxis=dict(title="PCA Dim 2", showgrid=True, gridcolor="rgba(255,255,255,.06)", zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,.08)", font=dict(color=TEXT_PRIMARY)),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ===========================================================================
# APP LAYOUT
# ===========================================================================
def main():
    scaler, kmeans, pca, df = load_artifacts()

    if df is None:
        inject_theme(*DEFAULT_ACCENT)
        st.markdown(
            """
            <div class="hero">
              <div class="hero-kicker">Spotify Vibe Space</div>
              <div class="hero-title">Charting the universe…</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.error(
            "**Artifacts not found.** Run `python model.py` first to build the model, "
            "then refresh this page.",
            icon="🚫",
        )
        st.code("python model.py", language="bash")
        st.stop()

    # ── Active vibe / accent resolution ──────────────────────────────
    all_vibes = sorted(df["vibe_label"].dropna().unique().tolist())
    selected_vibe = st.session_state.get("selected_vibe", "All Vibes")
    if selected_vibe != "All Vibes" and selected_vibe not in all_vibes:
        selected_vibe = "All Vibes"
        st.session_state["selected_vibe"] = selected_vibe

    accent_cluster = st.session_state.get("accent_cluster")
    accent, accent2 = _resolve_accent(df, selected_vibe, accent_cluster)
    inject_theme(accent, accent2)

    # ── Hero + metrics ───────────────────────────────────────────────
    total_tracks = len(df)
    n_clusters = int(df["cluster"].nunique())
    n_genres = int(df["genre"].nunique()) if "genre" in df.columns else 0
    n_artists = int(df["artist_name"].nunique()) if "artist_name" in df.columns else 0

    render_hero(total_tracks, n_clusters, n_genres, n_artists)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎵 Total Tracks", f"{total_tracks:,}")
    m2.metric("🎯 Vibe Clusters", n_clusters, delta=f"k = {n_clusters}")
    m3.metric("🎸 Genres Mapped", n_genres)
    m4.metric("👤 Artists", f"{n_artists:,}")

    # ── Scaled features + name index ─────────────────────────────────
    X_scaled, feature_cols = get_scaled_features(scaler, df)
    if "track_name_index" not in st.session_state:
        st.session_state["track_name_index"] = _build_name_index(df)

    # ── Vibe selector (main grid) ────────────────────────────────────
    render_vibe_selector(df, all_vibes, selected_vibe)

    df_view = df if selected_vibe == "All Vibes" else df[df["vibe_label"] == selected_vibe].copy()

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🛰️ Mission Control")
        st.markdown("---")

        # Active orbit indicator
        acc_cid = accent_cluster
        if acc_cid is None and selected_vibe != "All Vibes":
            sub = df[df["vibe_label"] == selected_vibe]
            if not sub.empty:
                acc_cid = int(sub["cluster"].iloc[0])
        if acc_cid is not None:
            acc_name = _vibe_label_for_cluster(df, acc_cid)
        else:
            acc_name = "All Vibes"
        acc_color = _palette_for_vibe(acc_name)[0] if acc_cid is not None else accent
        st.markdown(
            f"""
            <div class="vibe-orb-card" style="--c:{acc_color}; --c33:{acc_color}33;">
              <div class="orb"></div>
              <div class="vibe-orb-meta">
                <div class="vibe-orb-name">{html.escape(acc_name)}</div>
                <div class="vibe-orb-count">ambient accent active</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Random Shuffler ───────────────────────────────────────────
        if selected_vibe != "All Vibes":
            st.markdown("---")
            st.markdown("**🎲 Random Shuffler**")
            st.caption(f"Discover a random track from **{html.escape(selected_vibe)}**")

            # Initialize or get the random track for this vibe
            shuffle_key = f"random_track_{selected_vibe}"
            if shuffle_key not in st.session_state:
                vibe_tracks = df[df["vibe_label"] == selected_vibe]
                if not vibe_tracks.empty:
                    st.session_state[shuffle_key] = vibe_tracks.sample(1).iloc[0].to_dict()

            col_card, col_btn = st.columns([3, 1])
            with col_card:
                if shuffle_key in st.session_state:
                    track = st.session_state[shuffle_key]
                    # Use a simple card display
                    st.markdown(
                        f"""
                        <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:0.75rem;">
                            <div style="font-weight:700;color:#fff;font-size:0.9rem;">{html.escape(str(track.get('track_name', '—')))}</div>
                            <div style="color:#98a0b8;font-size:0.75rem;margin-top:0.2rem;">{html.escape(str(track.get('artist_name', '—')))}</div>
                            <div style="color:#98a0b8;font-size:0.7rem;margin-top:0.1rem;">{html.escape(str(track.get('album_name', '—')))}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            with col_btn:
                if st.button("🔄", key=f"shuffle_refresh_{selected_vibe}", use_container_width=True, help="Get another random track"):
                    vibe_tracks = df[df["vibe_label"] == selected_vibe]
                    if not vibe_tracks.empty:
                        st.session_state[shuffle_key] = vibe_tracks.sample(1).iloc[0].to_dict()
                        st.rerun()

        st.markdown("---")
        st.markdown("**🎨 Scatter Colour By**")
        colour_by = st.selectbox(
            "colour_by",
            options=["vibe_label", "energy", "popularity", "danceability", "valence"],
            format_func=lambda x: x.replace("_", " ").title(),
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**📖 Vibe Guide**")
        for vibe in all_vibes:
            color = _palette_for_vibe(vibe)[0]
            desc = VIBE_DESCRIPTIONS.get(vibe, "")
            st.markdown(
                f"""
                <div class="legend-row">
                  <span class="dot" style="--c:{color};"></span>
                  <div>
                    <div class="legend-name">{html.escape(vibe)}</div>
                    <div class="legend-desc">{html.escape(desc)}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


        st.markdown("---")
        st.markdown(
            """
            <div style="font-size:.72rem;color:#98a0b8;line-height:1.6;">
              Built with ❤️ using<br/>
              <b>scikit-learn</b> · <b>Streamlit</b> · <b>Plotly</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Tabs ─────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌌  Vibe Space Explorer",
        "🎧  Smart Playlist Generator",
        "📊  Analysis Dashboard",
        "🎛️  Manual Classifier",
    ])

    # ================================================================
    # TAB 1 — VIBE SPACE EXPLORER
    # ================================================================
    with tab1:
        render_section(
            "Vibe Space Explorer",
            "The shape of your sound",
            f"Each dot is a track projected into 2-D via PCA, coloured by "
            f"<b>{colour_by.replace('_', ' ').title()}</b>. Hover for details; zoom &amp; pan to explore.",
        )

        if df_view.empty:
            st.warning("No tracks for the selected vibe. Try 'All Vibes'.")
        else:
            plot_df = df_view.sample(min(len(df_view), 8_000), random_state=42)
            # keep row index for click-to-select
            plot_df = plot_df.reset_index(drop=False).rename(columns={"index": "_idx"})
            custom_cols = ["track_name", "_idx"]
            if colour_by == "vibe_label":
                color_map = {vibe: _palette_for_vibe(vibe)[0] for vibe in all_vibes}
                fig = px.scatter(
                    plot_df, x="pca_x", y="pca_y",
                    color="vibe_label", color_discrete_map=color_map,
                    custom_data=custom_cols, opacity=0.78, size_max=10,
                )
            else:
                fig = px.scatter(
                    plot_df, x="pca_x", y="pca_y",
                    color=colour_by, color_continuous_scale=CONTINUOUS_SCALE,
                    custom_data=custom_cols, opacity=0.78, size_max=10,
                )

            fig.update_traces(
                marker=dict(size=5, opacity=0.78, line=dict(width=0)),
                hovertemplate="%{customdata[0]}<extra></extra>",
            )
            fig.update_layout(
                **_dark_layout(accent, height=560),
                xaxis=dict(title="PCA Dim 1", showgrid=True, gridcolor="rgba(255,255,255,.06)", zeroline=False, ticks="outside", tickcolor="rgba(255,255,255,.15)"),
                yaxis=dict(title="PCA Dim 2", showgrid=True, gridcolor="rgba(255,255,255,.06)", zeroline=False, ticks="outside", tickcolor="rgba(255,255,255,.15)"),
                legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,.08)", borderwidth=1, font=dict(color=TEXT_PRIMARY)),
            )
            sel = st.plotly_chart(
                fig, use_container_width=True,
                config={"displayModeBar": False},
                on_select="rerun", selection_mode="points",
                key="vibe_scatter",
            )
            picked_idx = None
            try:
                pts = (sel or {}).get("selection", {}).get("points", [])
                if pts:
                    cd = pts[0].get("customdata") or []
                    if len(cd) > 1:
                        picked_idx = cd[1]
            except Exception:
                picked_idx = None
            if picked_idx is not None and picked_idx in df.index:
                row = df.loc[picked_idx]
                st.markdown("#### Selected track")
                render_rec_card(row.to_dict(), accent)

        st.markdown("#### 📊 Cluster audio fingerprints")
        stat_cols = [c for c in [
            "vibe_label", "energy", "danceability", "valence",
            "acousticness", "tempo", "speechiness",
        ] if c in df.columns]
        if stat_cols:
            stats = (
                df[stat_cols]
                .groupby("vibe_label")
                .agg(["mean", "std"])
                .round(3)
            )
            stats.columns = [" / ".join(c).strip() for c in stats.columns]
            with st.container(border=True):
                st.dataframe(stats, use_container_width=True)

    # ================================================================
    # TAB 2 — SMART PLAYLIST GENERATOR
    # ================================================================
    with tab2:
        render_section(
            "Synesthesia Search Hub",
            "Find your next sonic neighbour",
            "Type a seed track. The engine locates its vibe cluster and surfaces the most "
            "sonically similar tracks using cosine similarity on scaled audio features.",
        )

        col_input, col_slider = st.columns([3, 1])

        with col_input:
            if "seed_pick" in st.session_state:
                st.session_state["seed_track_input"] = st.session_state.pop("seed_pick")

            seed_input = st.text_input(
                "🔍 Seed track",
                placeholder="e.g. Blinding Lights, Shape of You, Levitating …",
                label_visibility="collapsed",
                key="seed_track_input",
            )

            typed = (seed_input or "").strip()
            if typed:
                matches = _suggest(st.session_state.get("track_name_index", []), typed, limit=10)
                if matches:
                    with st.container(border=True):
                        st.caption("💡 Suggestions from your dataset")
                        for i, name in enumerate(matches):
                            if st.button(name, key=f"seed_sug_{i}", use_container_width=True):
                                st.session_state["seed_pick"] = name
                                hit = df[df["track_name"].str.lower() == name.strip().lower()]
                                if not hit.empty:
                                    st.session_state["accent_cluster"] = int(hit.iloc[0]["cluster"])
                                st.rerun()

        with col_slider:
            top_n = st.slider("Tracks", min_value=5, max_value=30, value=10, step=1)

        generate_btn = st.button("✨  Generate Playlist", use_container_width=False)

        if generate_btn or seed_input:
            if not seed_input.strip():
                st.info("👆 Enter a seed track name above and hit **Generate Playlist**.")
            else:
                with st.spinner("Finding your vibe matches …"):
                    try:
                        recs = recommend_tracks(seed_input, df, X_scaled, top_n=top_n)

                        mask = df["track_name"].str.lower().str.contains(
                            seed_input.strip().lower(), na=False, regex=False
                        )
                        seed_row = df[mask].iloc[0]
                        seed_cluster = int(seed_row["cluster"])
                        seed_color = _palette_for_vibe(seed_row.get("vibe_label", ""))[0]
                        seed_pos = df.index.get_loc(seed_row.name)
                        seed_vec_scaled = X_scaled[seed_pos].reshape(1, -1)

                        # Shift the whole observatory to the seed's vibe on explicit generate
                        if generate_btn and st.session_state.get("accent_cluster") != seed_cluster:
                            st.session_state["accent_cluster"] = seed_cluster
                            st.rerun()

                        # ── Seed card ───────────────────────────────
                        with st.container(border=True):
                            s1, s2, s3 = st.columns([1, 2, 2])
                            with s1:
                                st.markdown(
                                    f'<div class="seed-orb" style="--c:{seed_color}; --c33:{seed_color}33;">🎵</div>',
                                    unsafe_allow_html=True,
                                )
                            with s2:
                                st.markdown(f"### {html.escape(str(seed_row.get('track_name', '—')))}")
                                st.markdown(
                                    f"<span class='tag'>{html.escape(str(seed_row.get('artist_name', '—')))}</span>"
                                    f"<span class='tag' style='border-color:{seed_color};color:{seed_color};'>"
                                    f"{html.escape(str(seed_row.get('vibe_label', '—')))}</span>",
                                    unsafe_allow_html=True,
                                )
                                feat_map = {
                                    "energy": seed_row.get("energy", 0),
                                    "danceability": seed_row.get("danceability", 0),
                                    "valence": seed_row.get("valence", 0),
                                    "acousticness": seed_row.get("acousticness", 0),
                                    "speechiness": seed_row.get("speechiness", 0),
                                    "liveness": seed_row.get("liveness", 0),
                                }
                                chips = []
                                if pd.notna(feat_map.get("energy")):
                                    chips.append(f"<span class='chip'>⚡ {float(feat_map['energy']):.2f}</span>")
                                if pd.notna(feat_map.get("danceability")):
                                    chips.append(f"<span class='chip'>🕺 {float(feat_map['danceability']):.2f}</span>")
                                if pd.notna(feat_map.get("valence")):
                                    chips.append(f"<span class='chip'>😊 {float(feat_map['valence']):.2f}</span>")
                                if pd.notna(seed_row.get("tempo", None)):
                                    chips.append(f"<span class='chip'>🥁 {float(seed_row['tempo']):.0f} BPM</span>")
                                st.markdown("".join(chips), unsafe_allow_html=True)
                            with s3:
                                radar = _build_radar(feat_map, label="Seed", color=seed_color)
                                st.plotly_chart(radar, use_container_width=True, config={"displayModeBar": False})

                        # ── Recommendations ─────────────────────────
                        st.markdown(
                            f"#### 🎵 Top {len(recs)} matches for "
                            f"**{html.escape(str(seed_row.get('track_name', seed_input)))}**"
                        )
                        render_rec_grid(recs, accent)

                        st.markdown("#### Recommendations in vibe space")
                        rec_names = recs["track_name"].tolist() if "track_name" in recs.columns else []
                        render_mini_scatter(
                            df, pca, seed_vec_scaled,
                            str(seed_row.get("track_name", "Seed")),
                            df[df["cluster"] == seed_cluster],
                            rec_names, accent,
                        )

                    except ValueError as exc:
                        st.error(f"❌  {exc}", icon="🔍")
                        st.markdown("_Tip: Try a partial name, e.g. `Shape` instead of `Shape of You`._")
                    except Exception as exc:
                        st.error(f"Unexpected error: {exc}")


    # ================================================================
    # TAB 3 — ANALYSIS DASHBOARD
    # ================================================================
    with tab3:
        render_section(
            "Analysis Dashboard",
            "Dataset at a glance",
            "A consolidated view of the audio features, cluster structure, and track distribution in this dataset.",
        )

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Tracks", f"{len(df):,}")
        d2.metric("Clusters", f"{df['cluster'].nunique()}")
        d3.metric("Genres", f"{df['genre'].nunique() if 'genre' in df.columns else 0}")
        d4.metric("Artists", f"{df['artist_name'].nunique() if 'artist_name' in df.columns else 0:,}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Tracks per vibe cluster")
            vc = (
                df.groupby(["vibe_label"], as_index=False).size()
                .rename(columns={"size": "tracks"})
                .sort_values("tracks", ascending=True)
            )
            fig_vc = px.bar(
                vc, x="tracks", y="vibe_label", orientation="h",
                color="vibe_label",
                color_discrete_map={v: _palette_for_vibe(v)[0] for v in vc["vibe_label"]},
            )
            fig_vc.update_layout(**_dark_layout(accent, height=360), showlegend=False,
                                 xaxis_title="Tracks", yaxis_title="")
            st.plotly_chart(fig_vc, use_container_width=True, config={"displayModeBar": False})

        with c2:
            st.markdown("#### Average audio profile by vibe")
            feat_cols = [c for c in ["danceability","energy","valence","acousticness","speechiness","liveness"] if c in df.columns]
            if feat_cols:
                prof = df.groupby("vibe_label")[feat_cols].mean().reset_index()
                melted = prof.melt(id_vars="vibe_label", var_name="feature", value_name="mean")
                fig_prof = px.bar(
                    melted, x="feature", y="mean", color="vibe_label", barmode="group",
                    color_discrete_map={v: _palette_for_vibe(v)[0] for v in prof["vibe_label"]},
                )
                fig_prof.update_layout(**_dark_layout(accent, height=360), yaxis_range=[0, 1],
                                       xaxis_title="", yaxis_title="Mean value")
                st.plotly_chart(fig_prof, use_container_width=True, config={"displayModeBar": False})

        st.markdown("#### Vibe space (PCA projection)")
        sample_dash = df.sample(min(len(df), 8000), random_state=42)
        color_map_dash = {vibe: _palette_for_vibe(vibe)[0] for vibe in all_vibes}
        fig_map = px.scatter(
            sample_dash, x="pca_x", y="pca_y", color="vibe_label",
            color_discrete_map=color_map_dash, custom_data=["track_name"],
            opacity=0.7,
        )
        fig_map.update_traces(
            marker=dict(size=4, line=dict(width=0)),
            hovertemplate="%{customdata[0]}<extra></extra>",
        )
        fig_map.update_layout(
            **_dark_layout(accent, height=520),
            xaxis=dict(title="PCA Dim 1", showgrid=True, gridcolor="rgba(255,255,255,.06)", zeroline=False),
            yaxis=dict(title="PCA Dim 2", showgrid=True, gridcolor="rgba(255,255,255,.06)", zeroline=False),
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,.08)", font=dict(color=TEXT_PRIMARY)),
        )
        st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

        c3 = st.columns(1)[0]
        with c3:
            st.markdown("#### Feature distributions")
            dist_feat = st.selectbox(
                "Feature", [c for c in ["energy","danceability","valence","acousticness","tempo","popularity","loudness"] if c in df.columns],
                key="dash_dist_feat",
            )
            fig_hist = px.histogram(
                df, x=dist_feat, color="vibe_label", nbins=40, barmode="overlay", opacity=0.55,
                color_discrete_map={v: _palette_for_vibe(v)[0] for v in all_vibes},
            )
            fig_hist.update_layout(**_dark_layout(accent, height=360), yaxis_title="Count")
            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

        st.markdown("#### Cluster audio fingerprints (mean / std)")
        stat_cols = [c for c in ["vibe_label", "energy", "danceability", "valence", "acousticness", "tempo", "speechiness"] if c in df.columns]
        if stat_cols:
            stats = df[stat_cols].groupby("vibe_label").agg(["mean", "std"]).round(3)
            stats.columns = [" / ".join(c).strip() for c in stats.columns]
            with st.container(border=True):
                st.dataframe(stats, use_container_width=True)

    # ================================================================
    # TAB 4 — MANUAL CLASSIFIER
    # ================================================================
    with tab4:
        render_section(
            "Manual Vibe Classifier",
            "Enter audio features to predict the vibe",
            "Input any track's audio characteristics and the model will classify it into one of the 6 vibe clusters.",
        )

        # Default values (approximate dataset means)
        defaults = {
            "danceability": 0.55,
            "energy": 0.55,
            "key": 5,
            "loudness": -8.0,
            "mode": 1,
            "speechiness": 0.05,
            "acousticness": 0.3,
            "instrumentalness": 0.0,
            "liveness": 0.15,
            "valence": 0.45,
            "tempo": 120.0,
        }

        st.markdown("**Audio Features**")
        
        # Song name input (optional, for display only)
        song_name = st.text_input(
            "🎵 Song Name (optional)",
            placeholder="e.g., My Custom Track",
            key="manual_song_name",
            help="Just for display in results — does not affect classification"
        )
        
        c1, c2, c3 = st.columns(3)
        inputs = {}
        feature_info = [
            ("danceability", "Danceability", 0.0, 1.0, 0.01, "How suitable for dancing"),
            ("energy", "Energy", 0.0, 1.0, 0.01, "Intensity and activity level"),
            ("key", "Key (0-11)", 0, 11, 1, "Pitch class (C=0, C#=1, ...)"),
            ("loudness", "Loudness (dB)", -60.0, 5.0, 0.1, "Overall loudness in decibels"),
            ("mode", "Mode (0/1)", 0, 1, 1, "Major (1) or Minor (0)"),
            ("speechiness", "Speechiness", 0.0, 1.0, 0.01, "Presence of spoken words"),
            ("acousticness", "Acousticness", 0.0, 1.0, 0.01, "Confidence track is acoustic"),
            ("instrumentalness", "Instrumentalness", 0.0, 1.0, 0.01, "Confidence track is instrumental"),
            ("liveness", "Liveness", 0.0, 1.0, 0.01, "Presence of audience/live feel"),
            ("valence", "Valence", 0.0, 1.0, 0.01, "Musical positiveness/mood"),
            ("tempo", "Tempo (BPM)", 40.0, 220.0, 0.5, "Beats per minute"),
        ]

        for idx, (feat, label, min_v, max_v, step, help_text) in enumerate(feature_info):
            col = [c1, c2, c3][idx % 3]
            with col:
                if feat in ("key", "mode"):
                    inputs[feat] = st.number_input(
                        label, min_value=min_v, max_value=max_v, step=step,
                        value=int(defaults[feat]), key=f"manual_{feat}", help=help_text
                    )
                else:
                    inputs[feat] = st.number_input(
                        label, min_value=min_v, max_value=max_v, step=step,
                        value=defaults[feat], key=f"manual_{feat}", help=help_text, format="%.2f"
                    )

        classify_btn = st.button("🔮 Classify Vibe", use_container_width=True, type="primary")

        if classify_btn:
            # Build feature vector in the same order as training
            feature_cols = [f for f in AUDIO_FEATURE_KEYS if f in df.columns]
            vec = np.array([[inputs[f] for f in feature_cols]], dtype=np.float32)
            
            # Scale and predict
            vec_scaled = scaler.transform(vec)
            cluster_id = int(kmeans.predict(vec_scaled)[0])
            vibe_label = df[df["cluster"] == cluster_id]["vibe_label"].iloc[0]
            vibe_color = _palette_for_vibe(vibe_label)[0]
            
            # Get distances to all centroids for confidence
            distances = np.linalg.norm(kmeans.cluster_centers_ - vec_scaled, axis=1)
            closest_dist = distances[cluster_id]
            
            # Better confidence: inverse of normalized distance to closest centroid
            # Use softmax-like approach: exp(-d) / sum(exp(-d))
            inv_distances = np.exp(-distances)
            confidence = inv_distances[cluster_id] / inv_distances.sum()
            confidence = max(0.0, min(1.0, confidence))

            # Display result
            st.markdown("---")
            res_left, res_right = st.columns([1, 1])
            
            with res_left:
                display_name = html.escape(song_name.strip()) if song_name.strip() else "Unnamed Track"
                st.markdown(
                    f"""
                    <div style="background:linear-gradient(135deg, {vibe_color}22, {vibe_color}08);border:1px solid {vibe_color}44;border-radius:16px;padding:1.5rem;text-align:center;">
                        <div style="font-size:0.7rem;letter-spacing:.2em;text-transform:uppercase;color:{vibe_color};font-weight:700;margin-bottom:.5rem;">Predicted Vibe</div>
                        <div style="font-size:1.1rem;color:#98a0b8;margin-bottom:.3rem;">{display_name}</div>
                        <div style="font-size:1.8rem;font-weight:800;color:#fff;margin-bottom:.3rem;">{html.escape(vibe_label)}</div>
                        <div style="font-size:0.9rem;color:#98a0b8;">Cluster {cluster_id}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                # Show nearest centroid distances - horizontally extended
                st.markdown("**Distance to cluster centroids**")
                dist_df = pd.DataFrame({
                    "Vibe": [df[df["cluster"] == i]["vibe_label"].iloc[0] for i in range(N_CLUSTERS)],
                    "Distance": distances,
                    "Is Closest": [i == cluster_id for i in range(N_CLUSTERS)]
                }).sort_values("Distance")
                
                # Horizontal bar chart style display
                for _, row in dist_df.iterrows():
                    badge = " ✓" if row["Is Closest"] else ""
                    color = _palette_for_vibe(row["Vibe"])[0] if row["Is Closest"] else "#98a0b8"
                    # Normalize distance for bar width (invert so closer = wider)
                    max_dist = distances.max()
                    bar_width = int((1 - row["Distance"] / max_dist) * 100) if max_dist > 0 else 100
                    st.markdown(
                        f"""
                        <div style='margin-bottom:.4rem;'>
                            <div style='display:flex;align-items:center;gap:.5rem;margin-bottom:.15rem;'>
                                <span style='color:{color};font-size:.8rem;font-weight:600;min-width:160px;'>{html.escape(row['Vibe'])}</span>
                                <span style='color:#98a0b8;font-size:.75rem;font-family:monospace;min-width:50px;'>{row['Distance']:.3f}</span>
                            </div>
                            <div style='background:rgba(255,255,255,.08);border-radius:4px;height:6px;overflow:hidden;'>
                                <div style='background:{color};width:{bar_width}%;height:100%;transition:width .3s;'></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with res_right:
                # Radar chart for the input features
                radar_feats = {
                    "danceability": inputs["danceability"],
                    "energy": inputs["energy"],
                    "valence": inputs["valence"],
                    "acousticness": inputs["acousticness"],
                    "speechiness": inputs["speechiness"],
                    "liveness": inputs["liveness"],
                }
                radar = _build_radar(radar_feats, label="Input Features", color=vibe_color)
                st.plotly_chart(radar, use_container_width=True, config={"displayModeBar": False})

            # PCA projection of the input
            st.markdown("#### Position in Vibe Space")
            pca_coords = pca.transform(vec_scaled)
            
            fig = go.Figure()
            # Background cluster points
            for vibe in all_vibes:
                sub = df[df["vibe_label"] == vibe].sample(min(500, len(df[df["vibe_label"] == vibe])), random_state=42)
                color = _palette_for_vibe(vibe)[0]
                fig.add_trace(go.Scatter(
                    x=sub["pca_x"], y=sub["pca_y"],
                    mode="markers",
                    marker=dict(size=4, color=color, opacity=0.3, line=dict(width=0)),
                    name=vibe,
                    hoverinfo="skip",
                    showlegend=True,
                ))
            # Input point
            fig.add_trace(go.Scatter(
                x=[pca_coords[0, 0]], y=[pca_coords[0, 1]],
                mode="markers+text",
                marker=dict(size=18, color="#ffffff", symbol="star", line=dict(width=2, color=vibe_color)),
                name="Your Track",
                text=[f"★ Your Track"],
                textposition="top center",
                textfont=dict(color=vibe_color, size=11, family="Inter"),
                showlegend=True,
            ))
            fig.update_layout(
                **_dark_layout(accent, height=400),
                xaxis=dict(title="PCA Dim 1", showgrid=True, gridcolor="rgba(255,255,255,.06)", zeroline=False),
                yaxis=dict(title="PCA Dim 2", showgrid=True, gridcolor="rgba(255,255,255,.06)", zeroline=False),
                legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,.08)", font=dict(color=TEXT_PRIMARY)),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


if __name__ == "__main__":
    main()