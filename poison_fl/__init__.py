"""poison_fl – Poisoning data generation for Federated Learning.

Implements the data-poisoning strategy from:
    Al Dalaien et al., "A dual-aggregation approach to fortify federated
    learning against poisoning attacks in IoTs", Array 28 (2025) 100520.
"""

from poison_fl.poisoning import poison_data
from poison_fl.preprocessing import preprocess
from poison_fl.distribution import distribute_iid, distribute_non_iid
from poison_fl.pipeline import run_pipeline

__version__ = "0.1.0"

__all__ = [
    "poison_data",
    "preprocess",
    "distribute_iid",
    "distribute_non_iid",
    "run_pipeline",
]
