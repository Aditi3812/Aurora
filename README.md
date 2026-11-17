# 🎵 Spotify Vibe Space

> **Unsupervised Audio Clustering & Smart Playlist Generation**  
> KMeans · PCA · Cosine Similarity · Streamlit · Plotly

---

## ✨ What It Does

Spotify Vibe Space ingests real Spotify audio-feature data, groups every track into one of **6 latent vibe clusters** using unsupervised KMeans, projects the full feature space down to 2-D with PCA for a gorgeous interactive scatter plot, and lets you type any song name to instantly surface the **most sonically similar tracks** in that cluster via cosine similarity.

---

## 🗂️ Project Structure

```
Aurora/
├── data/
│   └── dataset.csv          ← Place your Kaggle Spotify CSV here
├── artifacts/               ← Auto-created by model.py
│   ├── scaler.pkl
│   ├── kmeans.pkl
│   ├── pca.pkl
│   └── processed_tracks.parquet
├── model.py                 ← ML pipeline (run once)
├── app.py                   ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Create & Activate Virtual Environment (Recommended)

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> The `venv/` folder is ignored by git (see `.gitignore`).  
> Re-create it on any new machine with the steps above.

### 2. Install Dependencies (if not done above)

```bash
pip install -r requirements.txt
```

### 2. Add Your Dataset

Place the Kaggle Spotify dataset CSV inside the `data/` folder.  
The file can be named anything — `model.py` auto-detects it.

> **Recommended dataset**: [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)  
> Expected key columns: `track_name`, `artists`/`track_artist`, `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `popularity`.

### 3. Build the Model

```bash
python model.py
```

This will:
- Auto-detect and load the CSV from `data/`
- Clean rows with missing audio features
- Fit a `StandardScaler` and `KMeans` (k=6)
- Apply `PCA` for 2-D projection
- Save all artifacts to `artifacts/`

Typical runtime: **15–60 seconds** depending on dataset size.

### 4. Launch the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🎛️ Feature Overview

| Feature | Details |
|---------|---------|
| **Vibe Clusters** | 6 unsupervised clusters, labelled by mean audio signature |
| **PCA Scatter** | Interactive 2-D projection, coloured by vibe / energy / popularity / danceability / valence |
| **Hover Cards** | Track name, artist, BPM, energy on hover |
| **Recommendation Engine** | Cosine similarity within cluster, top-N configurable via slider |
| **Radar Chart** | Per-track audio fingerprint for the seed song |
| **Cluster Stats** | Mean ± std of all audio features per vibe |
| **Filter Sidebar** | Filter entire app by selected vibe label |

---

## 🎨 Vibe Labels

| Cluster | Vibe | Character |
|---------|------|-----------|
| 0 | ⚡ High Energy & Workout | Highest energy, strong beat |
| 1 | 🌙 Late Night Ambient | Low energy, atmospheric |
| 2 | 🎹 Melancholic & Acoustic | High acousticness, low valence |
| 3 | 🕺 Dance Floor Bangers | High danceability, positive valence |
| 4 | 🎸 Rock & Intensity | High loudness, mid energy |
| 5 | 😌 Chill & Feel-Good | Moderate energy, high valence |

> Cluster-to-vibe assignment is dynamic — clusters are sorted by mean energy and labelled accordingly, so the mapping may shift slightly with different datasets.

---

## 🧠 ML Architecture

```
Raw CSV
  │
  ▼
pandas (load + clean)
  │
  ▼
StandardScaler          ← Normalise 11 audio features
  │
  ├──► KMeans (k=6)     ← Cluster assignment
  │
  └──► PCA (2D)         ← Scatter coordinates

At query time:
  Seed track → its cluster → cosine_similarity vs all cluster members → top-N recs
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **pandas / numpy** — data wrangling
- **scikit-learn** — StandardScaler, KMeans, PCA, cosine_similarity
- **joblib** — artifact serialisation
- **streamlit** — interactive web UI
- **plotly** — interactive scatter & radar charts
- **pyarrow** — fast parquet I/O

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| `Artifacts not found` | Run `python model.py` first |
| `No CSV found in data/` | Ensure your CSV is inside the `data/` folder |
| `No track found matching …` | Try a partial name (e.g. `Shape` → `Shape of You`) |
| Slow first load | Only the first load is slow; Streamlit caches everything after |
| Missing column errors | Make sure your CSV has the standard Spotify audio feature columns |

---

## 📜 License

MIT — free to use, modify, and distribute.
