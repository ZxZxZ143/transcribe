from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT_CSV = Path("./../data/api/final_transcripts.csv")
DEFAULT_OUTPUT_CSV = Path("./../data/clustering/clustered_transcripts.csv")

DEFAULT_MODEL_NAME = "BAAI/bge-m3"

DEFAULT_ALGORITHM = "agglomerative"
DEFAULT_N_CLUSTERS = 9

DEFAULT_REDUCE_DIM = "pca"
DEFAULT_PCA_COMPONENTS = 20
DEFAULT_RANDOM_STATE = 0

DEFAULT_MIN_CLUSTER_SIZE = 4
DEFAULT_MIN_SAMPLES = 4

TEXT_COLUMNS_PRIORITY = [
    "semantic_text",
    "cleaned_transcript",
    "raw_transcript",
]


def now() -> float:
    return time.perf_counter()


def elapsed_seconds(started_at: float) -> float:
    return round(time.perf_counter() - started_at, 4)


def load_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
) -> Any:
    from sentence_transformers import SentenceTransformer

    if device:
        return SentenceTransformer(model_name, device=device)

    return SentenceTransformer(model_name)


def read_input_csv(input_csv: str | Path) -> pd.DataFrame:
    input_csv = Path(input_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"CSV не найден: {input_csv}")

    return pd.read_csv(input_csv, encoding="utf-8-sig")


def save_csv(df: pd.DataFrame, output_csv: str | Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")


def as_clean_string(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    return str(value).strip()


def build_text_for_clustering(row: pd.Series) -> str:
    for column in TEXT_COLUMNS_PRIORITY:
        if column in row.index:
            text = as_clean_string(row.get(column))

            if text:
                return text

    return ""


def add_text_for_clustering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_for_clustering"] = df.apply(build_text_for_clustering, axis=1)
    return df


def build_embeddings(
    texts: list[str],
    embedding_model: Any | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    batch_size: int = 16,
    show_progress_bar: bool = True,
    performance: dict[str, Any] | None = None,
) -> np.ndarray:
    if performance is None:
        performance = {}

    if embedding_model is None:
        model_load_started_at = now()

        embedding_model = load_embedding_model(
            model_name=model_name,
            device=device,
        )

        performance["model_load_seconds"] = elapsed_seconds(model_load_started_at)
        performance["model_loaded_inside_function"] = True
    else:
        performance["model_load_seconds"] = 0.0
        performance["model_loaded_inside_function"] = False

    embeddings_started_at = now()

    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    )

    performance["embeddings_seconds"] = elapsed_seconds(embeddings_started_at)
    performance["embedding_texts_count"] = len(texts)

    if len(texts) > 0:
        performance["seconds_per_embedded_text"] = round(
            performance["embeddings_seconds"] / len(texts),
            6,
        )
    else:
        performance["seconds_per_embedded_text"] = 0.0

    return np.asarray(embeddings, dtype=np.float32)


def reduce_embeddings_pca(
    embeddings: np.ndarray,
    n_components: int = DEFAULT_PCA_COMPONENTS,
    random_state: int = DEFAULT_RANDOM_STATE,
    performance: dict[str, Any] | None = None,
) -> np.ndarray:
    if performance is None:
        performance = {}

    pca_started_at = now()

    if embeddings.ndim != 2:
        raise ValueError("embeddings должен быть двумерным массивом: rows x dimensions")

    n_samples, n_features = embeddings.shape

    performance["original_embedding_dim"] = int(n_features)
    performance["pca_requested_components"] = int(n_components)

    if n_samples < 2:
        performance["pca_seconds"] = elapsed_seconds(pca_started_at)
        performance["pca_used"] = False
        performance["pca_skip_reason"] = "n_samples < 2"
        performance["reduced_embedding_dim"] = int(n_features)
        return embeddings

    max_components = min(n_samples, n_features)

    if n_components <= 0:
        performance["pca_seconds"] = elapsed_seconds(pca_started_at)
        performance["pca_used"] = False
        performance["pca_skip_reason"] = "pca_components <= 0"
        performance["reduced_embedding_dim"] = int(n_features)
        return embeddings

    effective_components = min(int(n_components), int(max_components))

    if effective_components >= n_features:
        performance["pca_seconds"] = elapsed_seconds(pca_started_at)
        performance["pca_used"] = False
        performance["pca_skip_reason"] = "pca_components >= original_embedding_dim"
        performance["reduced_embedding_dim"] = int(n_features)
        return embeddings

    from sklearn.decomposition import PCA

    pca = PCA(
        n_components=effective_components,
        random_state=random_state,
    )

    reduced_embeddings = pca.fit_transform(embeddings)

    performance["pca_seconds"] = elapsed_seconds(pca_started_at)
    performance["pca_used"] = True
    performance["pca_effective_components"] = int(effective_components)
    performance["reduced_embedding_dim"] = int(reduced_embeddings.shape[1])
    performance["pca_explained_variance_sum"] = round(
        float(pca.explained_variance_ratio_.sum()),
        4,
    )

    return np.asarray(reduced_embeddings, dtype=np.float32)


def reduce_embeddings(
    embeddings: np.ndarray,
    reduce_dim: str = DEFAULT_REDUCE_DIM,
    pca_components: int = DEFAULT_PCA_COMPONENTS,
    random_state: int = DEFAULT_RANDOM_STATE,
    performance: dict[str, Any] | None = None,
) -> np.ndarray:
    if performance is None:
        performance = {}

    reduce_dim = (reduce_dim or "none").lower().strip()

    performance["reduce_dim"] = reduce_dim

    if reduce_dim == "none":
        performance["pca_seconds"] = 0.0
        performance["pca_used"] = False
        performance["original_embedding_dim"] = int(embeddings.shape[1]) if embeddings.ndim == 2 else None
        performance["reduced_embedding_dim"] = int(embeddings.shape[1]) if embeddings.ndim == 2 else None
        return embeddings

    if reduce_dim == "pca":
        return reduce_embeddings_pca(
            embeddings=embeddings,
            n_components=pca_components,
            random_state=random_state,
            performance=performance,
        )

    raise ValueError("reduce_dim должен быть одним из: none, pca")


def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int | None = DEFAULT_MIN_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        prediction_data=False,
    )

    labels = clusterer.fit_predict(embeddings)
    probabilities = clusterer.probabilities_

    return labels.astype(int), probabilities.astype(float)


def run_agglomerative(
    embeddings: np.ndarray,
    n_clusters: int = DEFAULT_N_CLUSTERS,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import AgglomerativeClustering

    if embeddings.ndim != 2:
        raise ValueError("embeddings должен быть двумерным массивом: rows x dimensions")

    n_samples = embeddings.shape[0]

    if n_samples == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    if n_clusters <= 0:
        raise ValueError("n_clusters должен быть больше 0")

    effective_clusters = min(int(n_clusters), int(n_samples))

    clusterer = AgglomerativeClustering(
        n_clusters=effective_clusters,
        linkage="ward",
    )

    labels = clusterer.fit_predict(embeddings)

    probabilities = calculate_cluster_probabilities(
        embeddings=embeddings,
        labels=labels,
    )

    return labels.astype(int), probabilities.astype(float)


def run_kmeans(
    embeddings: np.ndarray,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import KMeans

    if embeddings.ndim != 2:
        raise ValueError("embeddings должен быть двумерным массивом: rows x dimensions")

    n_samples = embeddings.shape[0]

    if n_samples == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    if n_clusters <= 0:
        raise ValueError("n_clusters должен быть больше 0")

    effective_clusters = min(int(n_clusters), int(n_samples))

    clusterer = KMeans(
        n_clusters=effective_clusters,
        n_init=10,
        random_state=random_state,
    )

    labels = clusterer.fit_predict(embeddings)

    probabilities = calculate_cluster_probabilities(
        embeddings=embeddings,
        labels=labels,
    )

    return labels.astype(int), probabilities.astype(float)


def calculate_cluster_probabilities(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    if len(labels) == 0:
        return np.array([], dtype=float)

    probabilities = np.zeros(len(labels), dtype=float)

    for cluster_id in sorted(set(labels.tolist())):
        if cluster_id == -1:
            probabilities[labels == cluster_id] = 0.0
            continue

        mask = labels == cluster_id
        cluster_embeddings = embeddings[mask]

        if len(cluster_embeddings) == 1:
            probabilities[mask] = 1.0
            continue

        centroid = cluster_embeddings.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)

        if centroid_norm == 0:
            probabilities[mask] = 0.5
            continue

        embedding_norms = np.linalg.norm(cluster_embeddings, axis=1)

        similarities = []
        for vector, vector_norm in zip(cluster_embeddings, embedding_norms):
            if vector_norm == 0:
                similarities.append(0.0)
            else:
                similarities.append(float(np.dot(vector, centroid) / (vector_norm * centroid_norm)))

        similarities_array = np.asarray(similarities, dtype=float)

        min_similarity = float(similarities_array.min())
        max_similarity = float(similarities_array.max())

        if max_similarity == min_similarity:
            normalized = np.ones_like(similarities_array)
        else:
            normalized = (similarities_array - min_similarity) / (max_similarity - min_similarity)

        probabilities[mask] = 0.5 + normalized * 0.5

    return np.round(probabilities, 4)


def run_clustering_algorithm(
    embeddings: np.ndarray,
    algorithm: str = DEFAULT_ALGORITHM,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int | None = DEFAULT_MIN_SAMPLES,
    random_state: int = DEFAULT_RANDOM_STATE,
    performance: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if performance is None:
        performance = {}

    algorithm = (algorithm or DEFAULT_ALGORITHM).lower().strip()
    performance["algorithm"] = algorithm

    clustering_started_at = now()

    if algorithm == "agglomerative":
        labels, probabilities = run_agglomerative(
            embeddings=embeddings,
            n_clusters=n_clusters,
        )

        performance["clustering_seconds"] = elapsed_seconds(clustering_started_at)
        performance["agglomerative_seconds"] = performance["clustering_seconds"]
        performance["hdbscan_seconds"] = 0.0
        performance["kmeans_seconds"] = 0.0
        return labels, probabilities

    if algorithm == "kmeans":
        labels, probabilities = run_kmeans(
            embeddings=embeddings,
            n_clusters=n_clusters,
            random_state=random_state,
        )

        performance["clustering_seconds"] = elapsed_seconds(clustering_started_at)
        performance["kmeans_seconds"] = performance["clustering_seconds"]
        performance["agglomerative_seconds"] = 0.0
        performance["hdbscan_seconds"] = 0.0
        return labels, probabilities

    if algorithm == "hdbscan":
        labels, probabilities = run_hdbscan(
            embeddings=embeddings,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )

        performance["clustering_seconds"] = elapsed_seconds(clustering_started_at)
        performance["hdbscan_seconds"] = performance["clustering_seconds"]
        performance["agglomerative_seconds"] = 0.0
        performance["kmeans_seconds"] = 0.0
        return labels, probabilities

    raise ValueError("algorithm должен быть одним из: agglomerative, kmeans, hdbscan")


def add_cluster_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cluster_sizes = df["cluster_id"].value_counts().to_dict()

    df["cluster_size"] = df["cluster_id"].map(cluster_sizes).fillna(0).astype(int)
    df["cluster_status"] = np.where(df["cluster_id"].eq(-1), "noise", "clustered")

    return df


def build_empty_clustered_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    clustered_df = df.copy()
    clustered_df["text_for_clustering"] = pd.Series(dtype="str")
    clustered_df["cluster_id"] = pd.Series(dtype="int")
    clustered_df["cluster_probability"] = pd.Series(dtype="float")
    clustered_df["cluster_size"] = pd.Series(dtype="int")
    clustered_df["cluster_status"] = pd.Series(dtype="str")
    return clustered_df


def cluster_dataframe(
    df: pd.DataFrame,
    embedding_model: Any | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    batch_size: int = 16,
    algorithm: str = DEFAULT_ALGORITHM,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int | None = DEFAULT_MIN_SAMPLES,
    reduce_dim: str = DEFAULT_REDUCE_DIM,
    pca_components: int = DEFAULT_PCA_COMPONENTS,
    random_state: int = DEFAULT_RANDOM_STATE,
    show_progress_bar: bool = True,
    performance: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if performance is None:
        performance = {}

    text_preparation_started_at = now()

    df = add_text_for_clustering(df)

    performance["text_preparation_seconds"] = elapsed_seconds(text_preparation_started_at)

    df["cluster_id"] = -1
    df["cluster_probability"] = 0.0

    valid_mask = df["text_for_clustering"].astype(str).str.strip().ne("")
    valid_count = int(valid_mask.sum())

    performance["total_rows"] = int(len(df))
    performance["valid_text_rows"] = valid_count
    performance["empty_text_rows"] = int(len(df) - valid_count)

    if valid_count == 0:
        performance["model_load_seconds"] = 0.0
        performance["embeddings_seconds"] = 0.0
        performance["pca_seconds"] = 0.0
        performance["pca_used"] = False
        performance["reduce_dim"] = reduce_dim
        performance["original_embedding_dim"] = None
        performance["reduced_embedding_dim"] = None
        performance["embedding_texts_count"] = 0
        performance["seconds_per_embedded_text"] = 0.0
        performance["clustering_seconds"] = 0.0
        performance["hdbscan_seconds"] = 0.0
        performance["agglomerative_seconds"] = 0.0
        performance["kmeans_seconds"] = 0.0
        performance["skipped_clustering_reason"] = "valid_count == 0"

        metadata_started_at = now()
        result_df = add_cluster_metadata(df)
        performance["cluster_metadata_seconds"] = elapsed_seconds(metadata_started_at)

        return result_df

    valid_texts = df.loc[valid_mask, "text_for_clustering"].astype(str).tolist()

    embeddings = build_embeddings(
        texts=valid_texts,
        embedding_model=embedding_model,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        performance=performance,
    )

    clustering_embeddings = reduce_embeddings(
        embeddings=embeddings,
        reduce_dim=reduce_dim,
        pca_components=pca_components,
        random_state=random_state,
        performance=performance,
    )

    labels, probabilities = run_clustering_algorithm(
        embeddings=clustering_embeddings,
        algorithm=algorithm,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        random_state=random_state,
        performance=performance,
    )

    valid_indexes = df.index[valid_mask]

    df.loc[valid_indexes, "cluster_id"] = labels
    df.loc[valid_indexes, "cluster_probability"] = np.round(probabilities, 4)

    metadata_started_at = now()
    result_df = add_cluster_metadata(df)
    performance["cluster_metadata_seconds"] = elapsed_seconds(metadata_started_at)

    return result_df


def run_clustering(
    input_csv: str | Path = DEFAULT_INPUT_CSV,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    embedding_model: Any | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    batch_size: int = 16,
    algorithm: str = DEFAULT_ALGORITHM,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int | None = DEFAULT_MIN_SAMPLES,
    reduce_dim: str = DEFAULT_REDUCE_DIM,
    pca_components: int = DEFAULT_PCA_COMPONENTS,
    random_state: int = DEFAULT_RANDOM_STATE,
    show_progress_bar: bool = True,
) -> dict[str, Any]:
    total_started_at = now()
    performance: dict[str, Any] = {}

    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    csv_read_started_at = now()
    df = read_input_csv(input_csv)
    performance["csv_read_seconds"] = elapsed_seconds(csv_read_started_at)

    if df.empty:
        clustered_df = build_empty_clustered_dataframe(df)

        performance["total_rows"] = 0
        performance["valid_text_rows"] = 0
        performance["empty_text_rows"] = 0
        performance["text_preparation_seconds"] = 0.0
        performance["model_load_seconds"] = 0.0
        performance["embeddings_seconds"] = 0.0
        performance["pca_seconds"] = 0.0
        performance["pca_used"] = False
        performance["reduce_dim"] = reduce_dim
        performance["original_embedding_dim"] = None
        performance["reduced_embedding_dim"] = None
        performance["clustering_seconds"] = 0.0
        performance["hdbscan_seconds"] = 0.0
        performance["agglomerative_seconds"] = 0.0
        performance["kmeans_seconds"] = 0.0
        performance["cluster_metadata_seconds"] = 0.0
        performance["embedding_texts_count"] = 0
        performance["seconds_per_embedded_text"] = 0.0
        performance["algorithm"] = algorithm
        performance["skipped_clustering_reason"] = "empty_dataframe"
    else:
        clustered_df = cluster_dataframe(
            df=df,
            embedding_model=embedding_model,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            algorithm=algorithm,
            n_clusters=n_clusters,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            reduce_dim=reduce_dim,
            pca_components=pca_components,
            random_state=random_state,
            show_progress_bar=show_progress_bar,
            performance=performance,
        )

    csv_save_started_at = now()
    save_csv(clustered_df, output_csv)
    performance["csv_save_seconds"] = elapsed_seconds(csv_save_started_at)

    performance["total_seconds"] = elapsed_seconds(total_started_at)

    total_rows = int(len(clustered_df))
    clustered_rows = int((clustered_df["cluster_id"] != -1).sum()) if total_rows else 0
    noise_rows = int((clustered_df["cluster_id"] == -1).sum()) if total_rows else 0
    clusters_count = int(
        clustered_df.loc[clustered_df["cluster_id"] != -1, "cluster_id"].nunique()
    ) if total_rows else 0

    if total_rows > 0:
        performance["seconds_per_row"] = round(
            performance["total_seconds"] / total_rows,
            6,
        )
    else:
        performance["seconds_per_row"] = 0.0

    return {
        "status": "done",
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "model_name": model_name,
        "algorithm": algorithm,
        "n_clusters": n_clusters,
        "total_rows": total_rows,
        "clustered_rows": clustered_rows,
        "noise_rows": noise_rows,
        "clusters_count": clusters_count,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "reduce_dim": reduce_dim,
        "pca_components": pca_components,
        "elapsed_seconds": performance["total_seconds"],
        "performance": performance,
    }


def print_performance(performance: dict[str, Any]) -> None:
    print("Performance:")
    print(f"  CSV read: {performance.get('csv_read_seconds', 0.0)}s")
    print(f"  Text preparation: {performance.get('text_preparation_seconds', 0.0)}s")
    print(f"  Model load: {performance.get('model_load_seconds', 0.0)}s")
    print(f"  Embeddings: {performance.get('embeddings_seconds', 0.0)}s")
    print(f"  PCA: {performance.get('pca_seconds', 0.0)}s | used: {performance.get('pca_used', False)}")
    print(f"  Clustering: {performance.get('clustering_seconds', 0.0)}s | algorithm: {performance.get('algorithm')}")
    print(f"  Cluster metadata: {performance.get('cluster_metadata_seconds', 0.0)}s")
    print(f"  CSV save: {performance.get('csv_save_seconds', 0.0)}s")
    print(f"  Total: {performance.get('total_seconds', 0.0)}s")
    print(f"  Seconds per row: {performance.get('seconds_per_row', 0.0)}s")
    print(f"  Seconds per embedded text: {performance.get('seconds_per_embedded_text', 0.0)}s")
    print(f"  Embedding dim: {performance.get('original_embedding_dim')} -> {performance.get('reduced_embedding_dim')}")

    if performance.get("pca_used"):
        print(f"  PCA explained variance sum: {performance.get('pca_explained_variance_sum')}")

    skipped_reason = performance.get("skipped_clustering_reason")
    if skipped_reason:
        print(f"  Skipped clustering reason: {skipped_reason}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster transcripts with BGE-M3 embeddings and selected clustering algorithm."
    )

    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)

    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument(
        "--algorithm",
        choices=["agglomerative", "kmeans", "hdbscan"],
        default=DEFAULT_ALGORITHM,
    )

    parser.add_argument("--n-clusters", type=int, default=DEFAULT_N_CLUSTERS)

    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)

    parser.add_argument("--reduce-dim", choices=["none", "pca"], default=DEFAULT_REDUCE_DIM)
    parser.add_argument("--pca-components", type=int, default=DEFAULT_PCA_COMPONENTS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = run_clustering(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        algorithm=args.algorithm,
        n_clusters=args.n_clusters,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        reduce_dim=args.reduce_dim,
        pca_components=args.pca_components,
        random_state=args.random_state,
        show_progress_bar=True,
    )

    print("Clustering finished")
    print(f"Input: {result['input_csv']}")
    print(f"Output: {result['output_csv']}")
    print(f"Algorithm: {result['algorithm']}")
    print(f"Requested clusters: {result['n_clusters']}")
    print(f"Total rows: {result['total_rows']}")
    print(f"Clusters: {result['clusters_count']}")
    print(f"Clustered rows: {result['clustered_rows']}")
    print(f"Noise rows: {result['noise_rows']}")
    print(f"Reduce dim: {result['reduce_dim']} | PCA components: {result['pca_components']}")
    print(f"Time: {result['elapsed_seconds']}s")
    print_performance(result["performance"])


if __name__ == "__main__":
    main()
