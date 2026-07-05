from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CLUSTERED_CSV = Path("data/clustering/clustered_transcripts.csv")
DEFAULT_REPORT_CSV = Path("data/clustering/clusters_summary.csv")

LANGUAGE_COLUMN = "selected_language"
PROBLEM_KEYWORDS_COLUMN = "problem_keywords_found"
TEXT_COLUMN = "text_for_clustering"

STOPWORDS = {
    "это", "что", "как", "для", "или", "там", "тут", "уже", "еще", "ещё",
    "меня", "мне", "мой", "моя", "мои", "ваш", "ваша", "вам", "вас",
    "они", "она", "оно", "его", "ему", "нее", "ней", "нас", "нам",
    "есть", "нет", "да", "ну", "вот", "просто", "сейчас", "потом",
    "здравствуйте", "спасибо", "пожалуйста", "хорошо", "добрый",
    "банк", "банка", "клиент", "оператор", "девушка",
    "мен", "сіз", "біз", "бұл", "сол", "осы", "және", "үшін", "бар",
    "жоқ", "иә", "ия", "рахмет", "жақсы", "қазір", "кейін", "керек",
}


def read_clustered_csv(clustered_csv: str | Path) -> pd.DataFrame:
    clustered_csv = Path(clustered_csv)

    if not clustered_csv.exists():
        raise FileNotFoundError(f"CSV с кластерами не найден: {clustered_csv}")

    return pd.read_csv(clustered_csv, encoding="utf-8-sig")


def save_report(df: pd.DataFrame, report_csv: str | Path) -> None:
    report_csv = Path(report_csv)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(report_csv, index=False, encoding="utf-8-sig")


def as_clean_string(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    return str(value).strip()


def tokenize_for_keywords(text: str) -> list[str]:
    text = as_clean_string(text).lower()
    tokens = re.findall(r"[a-zа-яёәғқңөұүһі0-9_]+", text, flags=re.IGNORECASE)

    return [
        token
        for token in tokens
        if len(token) >= 4 and token not in STOPWORDS
    ]


def top_terms_for_texts(texts: list[str], limit: int = 12) -> str:
    counter: Counter[str] = Counter()

    for text in texts:
        counter.update(tokenize_for_keywords(text))

    return ", ".join(term for term, _ in counter.most_common(limit))


def split_keywords(value: Any) -> list[str]:
    text = as_clean_string(value)

    if not text:
        return []

    return [
        item.strip()
        for item in text.split(",")
        if item.strip()
    ]


def top_problem_keywords(values: list[Any], limit: int = 12) -> str:
    counter: Counter[str] = Counter()

    for value in values:
        counter.update(split_keywords(value))

    return ", ".join(keyword for keyword, _ in counter.most_common(limit))


def short_text(text: Any, limit: int = 320) -> str:
    text = as_clean_string(text)

    if len(text) <= limit:
        return text

    return text[:limit].rstrip() + "..."


def get_languages(group: pd.DataFrame) -> str:
    if LANGUAGE_COLUMN not in group.columns:
        return ""

    values = [
        str(value).strip()
        for value in group[LANGUAGE_COLUMN].dropna().unique()
        if str(value).strip()
    ]

    return ", ".join(sorted(values))


def build_cluster_report(
    clustered_df: pd.DataFrame,
    examples_per_cluster: int = 5,
    top_terms_limit: int = 12,
    text_preview_limit: int = 320,
) -> pd.DataFrame:
    required_columns = {"cluster_id", "cluster_probability"}

    missing_columns = required_columns - set(clustered_df.columns)
    if missing_columns:
        raise ValueError(
            "В CSV нет обязательных колонок для отчёта: "
            + ", ".join(sorted(missing_columns))
        )

    if TEXT_COLUMN not in clustered_df.columns:
        raise ValueError(f"В CSV нет колонки с текстом: {TEXT_COLUMN}")

    summary_rows: list[dict[str, Any]] = []

    for cluster_id, group in clustered_df.groupby("cluster_id", dropna=False):
        group_sorted = group.sort_values(
            by="cluster_probability",
            ascending=False,
            kind="stable",
        )

        texts = group_sorted[TEXT_COLUMN].fillna("").astype(str).tolist()

        problem_keywords = ""
        if PROBLEM_KEYWORDS_COLUMN in group_sorted.columns:
            problem_keywords = top_problem_keywords(
                group_sorted[PROBLEM_KEYWORDS_COLUMN].tolist(),
                limit=top_terms_limit,
            )

        row = {
            "cluster_id": int(cluster_id),
            "cluster_status": "noise" if int(cluster_id) == -1 else "clustered",
            "cluster_size": int(len(group_sorted)),
            "avg_probability": round(float(group_sorted["cluster_probability"].mean()), 4),
            "languages": get_languages(group_sorted),
            "top_problem_keywords": problem_keywords,
            "top_terms": top_terms_for_texts(
                texts=texts,
                limit=top_terms_limit,
            ),
        }

        examples = [
            short_text(text, limit=text_preview_limit)
            for text in texts[:examples_per_cluster]
        ]

        for index in range(examples_per_cluster):
            row[f"example_{index + 1}"] = examples[index] if index < len(examples) else ""

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    if summary_df.empty:
        return summary_df

    return summary_df.sort_values(
        by=["cluster_status", "cluster_size", "cluster_id"],
        ascending=[True, False, True],
        kind="stable",
    )


def create_cluster_report(
    clustered_csv: str | Path = DEFAULT_CLUSTERED_CSV,
    report_csv: str | Path = DEFAULT_REPORT_CSV,
    examples_per_cluster: int = 5,
    top_terms_limit: int = 12,
    text_preview_limit: int = 320,
) -> dict[str, Any]:
    clustered_csv = Path(clustered_csv)
    report_csv = Path(report_csv)

    clustered_df = read_clustered_csv(clustered_csv)

    report_df = build_cluster_report(
        clustered_df=clustered_df,
        examples_per_cluster=examples_per_cluster,
        top_terms_limit=top_terms_limit,
        text_preview_limit=text_preview_limit,
    )

    save_report(report_df, report_csv)

    total_rows = int(len(clustered_df))
    clusters_count = int(
        clustered_df.loc[clustered_df["cluster_id"] != -1, "cluster_id"].nunique()
    ) if total_rows and "cluster_id" in clustered_df.columns else 0
    noise_rows = int((clustered_df["cluster_id"] == -1).sum()) if total_rows else 0

    return {
        "status": "done",
        "clustered_csv": str(clustered_csv),
        "report_csv": str(report_csv),
        "total_rows": total_rows,
        "clusters_count": clusters_count,
        "noise_rows": noise_rows,
        "report_rows": int(len(report_df)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a readable cluster summary report."
    )

    parser.add_argument("--clustered-csv", type=Path, default=DEFAULT_CLUSTERED_CSV)
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    parser.add_argument("--examples-per-cluster", type=int, default=5)
    parser.add_argument("--top-terms-limit", type=int, default=12)
    parser.add_argument("--text-preview-limit", type=int, default=320)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = create_cluster_report(
        clustered_csv=args.clustered_csv,
        report_csv=args.report_csv,
        examples_per_cluster=args.examples_per_cluster,
        top_terms_limit=args.top_terms_limit,
        text_preview_limit=args.text_preview_limit,
    )

    print("Cluster report created")
    print(f"Clustered CSV: {result['clustered_csv']}")
    print(f"Report CSV: {result['report_csv']}")
    print(f"Total rows: {result['total_rows']}")
    print(f"Clusters: {result['clusters_count']}")
    print(f"Noise rows: {result['noise_rows']}")
    print(f"Report rows: {result['report_rows']}")


if __name__ == "__main__":
    main()
