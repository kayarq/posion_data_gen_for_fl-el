"""Data distribution to FL clients (Section 5.2).

IID  – each client gets an equal, randomly shuffled share so that
       all classes appear in roughly the same proportions.
Non-IID – classes are sorted and split into shards, then shards are
          assigned to clients so each client sees only a subset of classes.
"""

from __future__ import annotations

import numpy as np


def distribute_iid(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int = 5,
    random_state: int = 42,
) -> list[dict]:
    """Split data equally among *n_clients* in an IID fashion.

    Returns a list of dicts, one per client, each containing
    ``X``, ``y``, and ``client_id``.
    """
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(y))
    splits = np.array_split(indices, n_clients)

    clients = []
    for i, idx in enumerate(splits):
        clients.append({
            "client_id": i,
            "X": X[idx],
            "y": y[idx],
        })
    return clients


def distribute_non_iid(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int = 5,
    shards_per_client: int = 2,
    random_state: int = 42,
) -> list[dict]:
    """Split data among *n_clients* in a non-IID fashion.

    Data is sorted by label then divided into ``n_clients * shards_per_client``
    shards. Each client receives *shards_per_client* shards, so it sees only a
    subset of the label space.
    """
    rng = np.random.default_rng(random_state)

    n_shards = n_clients * shards_per_client
    sorted_idx = np.argsort(y)
    shards = np.array_split(sorted_idx, n_shards)

    shard_ids = rng.permutation(n_shards)

    clients = []
    for i in range(n_clients):
        assigned = shard_ids[i * shards_per_client : (i + 1) * shards_per_client]
        idx = np.concatenate([shards[s] for s in assigned])
        clients.append({
            "client_id": i,
            "X": X[idx],
            "y": y[idx],
        })
    return clients
