"""Download and index BEIR benchmark corpora.

Usage:
    python -m siare.benchmarks.scripts.download_beir_corpus --dataset nfcorpus
    python -m siare.benchmarks.scripts.download_beir_corpus --dataset scifact --persist-dir ./data/beir
"""

import argparse
import json
import logging
from pathlib import Path


logger = logging.getLogger(__name__)

# BEIR datasets with sizes (for user guidance)
BEIR_DATASETS = {
    "nfcorpus": {"docs": 3633, "queries": 323, "size_mb": 5},
    "scifact": {"docs": 5183, "queries": 300, "size_mb": 10},
    "arguana": {"docs": 8674, "queries": 1406, "size_mb": 15},
    "fiqa": {"docs": 57638, "queries": 648, "size_mb": 50},
    "hotpotqa": {"docs": 5233329, "queries": 7405, "size_mb": 2000},
    "nq": {"docs": 2681468, "queries": 3452, "size_mb": 1500},
}

# Small datasets suitable for quick testing
SMALL_DATASETS = ["nfcorpus", "scifact", "arguana"]


def download_beir_dataset(
    dataset_name: str,
    output_dir: Path,
) -> tuple[dict, dict, dict]:
    """Download BEIR dataset from official source.

    Args:
        dataset_name: Name of BEIR dataset
        output_dir: Directory to save data

    Returns:
        Tuple of (corpus, queries, qrels)
    """
    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
    except ImportError as e:
        raise ImportError("beir library required. Install with: pip install beir") from e

    # Download and extract
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, str(output_dir))

    # Load data
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    logger.info(f"Loaded {dataset_name}: {len(corpus)} docs, {len(queries)} queries")

    return corpus, queries, qrels


def index_beir_corpus(
    corpus: dict,
    dataset_name: str,
    persist_dir: str,
) -> int:
    """Index BEIR corpus into vector store.

    Args:
        corpus: BEIR corpus dict
        dataset_name: Name for the index
        persist_dir: Directory for persistent storage

    Returns:
        Number of vectors indexed
    """
    from siare.benchmarks.corpus.loader import CorpusLoader

    loader = CorpusLoader(persist_dir=persist_dir)
    return loader.load_beir_corpus(corpus, index_name=f"beir_{dataset_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and index BEIR benchmark corpora"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(BEIR_DATASETS.keys()),
        help="BEIR dataset to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/beir",
        help="Directory for downloaded data",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="data/vector_store",
        help="Directory for vector store",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip vector indexing (download only)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = BEIR_DATASETS[args.dataset]
    print(f"\nDownloading {args.dataset}:")
    print(f"  Documents: {dataset_info['docs']:,}")
    print(f"  Queries: {dataset_info['queries']:,}")
    print(f"  Size: ~{dataset_info['size_mb']} MB")
    print()

    # Download
    corpus, queries, qrels = download_beir_dataset(args.dataset, output_dir)

    # Index
    if not args.skip_index:
        print("\nIndexing corpus to vector store...")
        count = index_beir_corpus(corpus, args.dataset, args.persist_dir)
        print(f"Indexed {count:,} vectors")

    # Save queries and qrels for benchmark use
    queries_path = output_dir / f"{args.dataset}_queries.json"
    qrels_path = output_dir / f"{args.dataset}_qrels.json"

    with queries_path.open("w") as f:
        json.dump(queries, f)
    with qrels_path.open("w") as f:
        json.dump(qrels, f)

    print(f"\nSaved queries to: {queries_path}")
    print(f"Saved qrels to: {qrels_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
