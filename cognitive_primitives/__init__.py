"""
Cognitive Primitives for RWKV-AtomSpace Integration

This module provides the foundational cognitive primitives and tensor encodings
for translating between RWKV neural state representations and AtomSpace
hypergraph patterns.
"""

from .tensor_primitives import TensorPrimitive, CognitiveState
from .hypergraph_encoding import HypergraphEncoder, AtomSpaceNode
from .ko6ml_adapter import Ko6mlAdapter

__version__ = "0.1.0"
__all__ = [
    "TensorPrimitive",
    "CognitiveState", 
    "HypergraphEncoder",
    "AtomSpaceNode",
    "Ko6mlAdapter"
]