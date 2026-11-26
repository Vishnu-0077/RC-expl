"""Benchmark datasets: chaotic dynamical systems and classification tasks."""
from .chaotic import (SYSTEMS, TimeSeries, henon_map, load_system, logistic_map, lorenz,
                      mackey_glass, narma, rossler, tent_map)
from .classification import CLASSIFICATION_DATASETS, load_classification

__all__ = ["SYSTEMS", "TimeSeries", "henon_map", "load_system", "logistic_map", "lorenz",
           "mackey_glass", "narma", "rossler", "tent_map", "CLASSIFICATION_DATASETS",
           "load_classification"]
