# %% [markdown]
# # Hierarchical Clustering of Songs into Genre-like Groups
# Uses audio features: danceability, energy, key, loudness, speechiness,
# acousticness, instrumentalness, liveness, valence, tempo.
#
# Note: clustering is unsupervised, so clusters are "genre-like" groups based on
# sound. We label them by inspecting each cluster's average feature profile.

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sns.set_theme(style="whitegrid")

# %% Configuration
CSV_PATH = "./data/song_genres_copy.csv"          # <-- change to your file
MAX_SONGS = 5000                # hierarchical clustering is O(n^2) in memory; sample if larger
LINKAGE_METHOD = "ward"         # 'ward', 'average', 'complete', 'single'
K_RANGE = range(2, 16)          # cluster counts to evaluate
N_CLUSTERS = None               # set an int to override the automatically chosen k
RANDOM_STATE = 42

FEATURES = [
    "danceability", "energy", "key", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

# %% Load data
df = pd.read_csv(CSV_PATH)
print("Shape:", df.shape)
missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# --- Make sure every feature is numeric -------------------------------------
# 'key' is often stored as text (e.g. "C#", "A minor", "5"), so convert it to
# pitch-class integers 0-11 (C=0, C#=1, ... B=11).
NOTE_TO_PC = {
    "C": 0, "B#": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "FB": 4,
    "F": 5, "E#": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9,
    "A#": 10, "BB": 10, "B": 11, "CB": 11,
}

def key_to_pitch_class(value):
    """Convert numbers or note names ('C#', 'Db', 'A minor', 'C#/Db') to 0-11."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip().upper()
    try:
        return float(s) % 12
    except ValueError:
        pass
    first = s.replace("♯", "#").replace("♭", "B").split()[0].split("/")[0]
    return NOTE_TO_PC.get(first, np.nan)

df["key"] = df["key"].apply(key_to_pitch_class)

for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

bad = df[FEATURES].isna().sum()
if bad.any():
    print("Rows with unparseable/missing values (will be dropped):")
    print(bad[bad > 0])

df = df.dropna(subset=FEATURES).drop_duplicates().reset_index(drop=True)

if len(df) > MAX_SONGS:
    df = df.sample(MAX_SONGS, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Sampled down to {MAX_SONGS} songs")

df[FEATURES].describe().T

# %% Preprocess
# 'key' is cyclic (0-11, where 11 is adjacent to 0), so encode it as sin/cos
# rather than treating it as a plain number.
X = df[FEATURES].copy()
X["key_sin"] = np.sin(2 * np.pi * X["key"] / 12)
X["key_cos"] = np.cos(2 * np.pi * X["key"] / 12)
X = X.drop(columns="key")

feature_names = list(X.columns)
X_scaled = StandardScaler().fit_transform(X)

# %% Build the linkage matrix
Z = linkage(X_scaled, method=LINKAGE_METHOD)  # ward implies euclidean distance

# Cophenetic correlation: how faithfully the dendrogram preserves pairwise distances
coph_corr, _ = cophenet(Z, pdist(X_scaled))
print(f"Cophenetic correlation ({LINKAGE_METHOD}): {coph_corr:.3f}")

# %% Dendrogram (truncated so it stays readable)
plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90,
           leaf_font_size=9, show_contracted=True)
plt.title(f"Hierarchical Clustering Dendrogram ({LINKAGE_METHOD} linkage, truncated)")
plt.xlabel("Cluster (size in parentheses if merged)")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# %% Choose number of clusters via silhouette score
scores = {}
for k in K_RANGE:
    labels_k = fcluster(Z, t=k, criterion="maxclust")
    if len(np.unique(labels_k)) > 1:
        scores[k] = silhouette_score(X_scaled, labels_k)

plt.figure(figsize=(8, 4))
plt.plot(list(scores.keys()), list(scores.values()), marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title("Silhouette score by k")
plt.show()

best_k = max(scores, key=scores.get)
k = N_CLUSTERS or best_k
print(f"Best k by silhouette: {best_k} (score {scores[best_k]:.3f}); using k = {k}")

# %% Assign clusters
df["cluster"] = fcluster(Z, t=k, criterion="maxclust")
print(df["cluster"].value_counts().sort_index())

# %% Cluster profiles (mean of the original features per cluster)
profile = df.groupby("cluster")[FEATURES].mean()
profile["n_songs"] = df["cluster"].value_counts().sort_index()
display(profile.round(3)) if "display" in globals() else print(profile.round(3))

# Heatmap of z-scored cluster means shows what makes each cluster distinctive
z_profile = (profile[FEATURES] - profile[FEATURES].mean()) / profile[FEATURES].std(ddof=0)
plt.figure(figsize=(11, max(3, 0.6 * k + 2)))
sns.heatmap(z_profile, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Cluster feature profiles (z-scored across clusters)")
plt.ylabel("Cluster")
plt.show()

# %% Heuristic genre-style labels from each cluster's profile
def describe_cluster(row):
    """Rough, human-readable label from relative feature strengths."""
    tags = []
    if row["speechiness"] > 0.5:
        tags.append("Hip-Hop / Rap-like")
    if row["acousticness"] > 0.6 and row["energy"] < 0:
        tags.append("Acoustic / Folk-like")
    if row["instrumentalness"] > 0.7:
        tags.append("Instrumental / Ambient / Classical-like")
    if row["energy"] > 0.7 and row["loudness"] > 0.5:
        tags.append("Rock / Metal-like")
    if row["danceability"] > 0.7 and row["valence"] > 0.5:
        tags.append("Pop / Dance-like")
    if row["danceability"] > 0.7 and row["energy"] > 0.7:
        tags.append("Electronic / EDM-like")
    if row["liveness"] > 1.0:
        tags.append("Live recordings")
    return ", ".join(tags) if tags else "Mixed / Other"

# Use z-scores across clusters for relative thresholds on most features,
# but raw values for features that are already bounded and interpretable.
label_input = z_profile.copy()
for col in ["speechiness", "acousticness", "instrumentalness", "danceability", "valence"]:
    label_input[col] = profile[col]
label_input["energy"] = z_profile["energy"] * 0 + profile["energy"].values
label_input["loudness"] = z_profile["loudness"]

cluster_labels = {c: describe_cluster(label_input.loc[c]) for c in label_input.index}
for c, lab in cluster_labels.items():
    print(f"Cluster {c}: {lab}")

df["genre_guess"] = df["cluster"].map(cluster_labels)
print("\nTreat these labels as starting points and adjust after listening to samples.")

# %% Visualize clusters in 2D with PCA
pca = PCA(n_components=2, random_state=RANDOM_STATE)
coords = pca.fit_transform(X_scaled)

plt.figure(figsize=(9, 7))
sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=df["cluster"],
                palette="tab10", s=15, alpha=0.7, legend="full")
plt.title(f"Song clusters in PCA space "
          f"({pca.explained_variance_ratio_.sum():.0%} variance explained)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

# %% Clustermap (rows are clustered too; uses a sample for speed)
sample_idx = np.random.RandomState(RANDOM_STATE).choice(
    len(df), size=min(300, len(df)), replace=False)
lut = dict(zip(sorted(df["cluster"].unique()),
               sns.color_palette("tab10", df["cluster"].nunique())))
row_colors = df.loc[sample_idx, "cluster"].map(lut).values

sns.clustermap(pd.DataFrame(X_scaled[sample_idx], columns=feature_names),
               method=LINKAGE_METHOD, row_colors=row_colors,
               cmap="vlag", center=0, figsize=(10, 10))
plt.show()

# %% Optional: compare against a known genre column, if your CSV has one
GENRE_COL = None  # e.g. "genre" or "playlist_genre"
if GENRE_COL and GENRE_COL in df.columns:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    print("Adjusted Rand Index:", adjusted_rand_score(df[GENRE_COL], df["cluster"]))
    print("Normalized Mutual Info:", normalized_mutual_info_score(df[GENRE_COL], df["cluster"]))
    display(pd.crosstab(df["cluster"], df[GENRE_COL], normalize="index").round(2))

# %% Sample songs per cluster (add name/artist columns if you have them)
name_cols = [c for c in ["track_name", "name", "title", "artist", "artists"] if c in df.columns]
for c in sorted(df["cluster"].unique()):
    print(f"\n=== Cluster {c}: {cluster_labels[c]} ===")
    sample = df[df["cluster"] == c].head(5)
    print(sample[name_cols + ["danceability", "energy", "valence", "tempo"]].to_string(index=False)
          if name_cols else sample[FEATURES].to_string(index=False))

# %% Save results
df.to_csv("songs_with_clusters.csv", index=False)
print("Saved songs_with_clusters.csv")
