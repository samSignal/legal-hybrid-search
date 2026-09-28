from pathlib import Path

import pytest

from legalsearch.corpus import load_dataset
from legalsearch.embeddings import WordLlamaEncoder
from legalsearch.search import HybridSearcher

DATA = Path(__file__).resolve().parent.parent / "data" / "norland"


@pytest.fixture(scope="session")
def encoder():
    return WordLlamaEncoder()


@pytest.fixture(scope="session")
def dataset():
    docs, _, qrels = load_dataset(DATA)
    _, train, _ = load_dataset(DATA, "train")
    _, test, _ = load_dataset(DATA, "test")
    return docs, train, test, qrels


@pytest.fixture(scope="session")
def searcher(encoder, dataset):
    s = HybridSearcher(encoder)
    s.index(dataset[0])
    return s
