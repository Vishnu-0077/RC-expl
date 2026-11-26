"""Reservoir computing: echo state networks, deep and ensemble variants, NG-RC, topologies and diagnostics."""
from .diagnostics import (effective_dimension, esp_convergence, memory_capacity,
                          quadratic_capacity)
from .esn import ESN, ESNEnsemble, ridge_regression
from .ngrc import NGRC
from .topology import TOPOLOGIES, build_topology, scale_spectral_radius, spectral_radius

__all__ = ["ESN", "ESNEnsemble", "NGRC", "ridge_regression", "TOPOLOGIES", "build_topology",
           "scale_spectral_radius", "spectral_radius", "memory_capacity", "quadratic_capacity",
           "effective_dimension", "esp_convergence"]
