"""
Tensor Primitives Module

Defines the foundational tensor representations for cognitive states
following the specification: [modality, depth, context, salience, autonomy_index]
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

class ModalityType(Enum):
    """Supported modality types for cognitive processing"""
    LINGUISTIC = 0
    VISUAL = 1
    AUDITORY = 2
    SPATIAL = 3
    TEMPORAL = 4
    ABSTRACT = 5
    SENSORIMOTOR = 6

@dataclass
class TensorPrimitive:
    """
    Core tensor primitive with cognitive dimensions:
    [modality, depth, context, salience, autonomy_index]
    """
    modality: int  # ModalityType value
    depth: int     # Processing depth level (0-N layers)
    context: int   # Context window size or position
    salience: float # Attention weight [0.0, 1.0]
    autonomy_index: float # Autonomy level [0.0, 1.0]
    
    def to_tensor(self) -> torch.Tensor:
        """Convert primitive to tensor representation"""
        return torch.tensor([
            self.modality,
            self.depth,
            self.context,
            self.salience,
            self.autonomy_index
        ], dtype=torch.float32)
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'TensorPrimitive':
        """Create primitive from tensor representation"""
        if tensor.shape[-1] != 5:
            raise ValueError(f"Expected tensor shape [..., 5], got {tensor.shape}")
        
        values = tensor.flatten()[-5:].tolist()
        return cls(
            modality=int(values[0]),
            depth=int(values[1]),
            context=int(values[2]),
            salience=float(values[3]),
            autonomy_index=float(values[4])
        )

class CognitiveState:
    """
    Manages cognitive state as a collection of tensor primitives
    with bidirectional mapping to RWKV internal states
    """
    
    def __init__(self, n_layer: int, n_embd: int):
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.primitives: Dict[str, TensorPrimitive] = {}
        self.rwkv_state: Optional[torch.Tensor] = None
        
    def add_primitive(self, name: str, primitive: TensorPrimitive):
        """Add a named cognitive primitive"""
        self.primitives[name] = primitive
        
    def extract_from_rwkv_state(self, rwkv_state: torch.Tensor) -> Dict[str, TensorPrimitive]:
        """
        Extract cognitive primitives from RWKV state tensor
        
        RWKV state format: [n_layer * (2 + head_size), n_embd]
        Maps to cognitive dimensions through factorization
        """
        self.rwkv_state = rwkv_state
        extracted_primitives = {}
        
        # Extract layer-wise cognitive dimensions
        for layer_idx in range(self.n_layer):
            # Extract attention state (corresponds to salience/context)
            att_state_idx = layer_idx * 3 + 1  # Attention state index
            if att_state_idx < rwkv_state.shape[0]:
                att_state = rwkv_state[att_state_idx]
                
                # Map attention state to cognitive dimensions
                modality = self._extract_modality(att_state)
                depth = layer_idx
                context = self._extract_context(att_state)
                salience = self._extract_salience(att_state)
                autonomy = self._extract_autonomy(att_state, layer_idx)
                
                primitive = TensorPrimitive(
                    modality=modality,
                    depth=depth,
                    context=context,
                    salience=salience,
                    autonomy_index=autonomy
                )
                
                extracted_primitives[f"layer_{layer_idx}"] = primitive
                
        return extracted_primitives
    
    def _extract_modality(self, state_tensor: torch.Tensor) -> int:
        """Extract modality from state tensor using prime factorization approach"""
        # Use mean activation patterns to determine modality
        mean_activation = torch.mean(state_tensor).item()
        # Map to modality types based on activation patterns
        modality_idx = int(abs(mean_activation * 7)) % len(ModalityType)
        return modality_idx
    
    def _extract_context(self, state_tensor: torch.Tensor) -> int:
        """Extract context size from state tensor"""
        # Use variance as proxy for context complexity
        variance = torch.var(state_tensor).item()
        return min(int(variance * 1000), 2048)  # Cap at reasonable context size
    
    def _extract_salience(self, state_tensor: torch.Tensor) -> float:
        """Extract salience (attention weight) from state tensor"""
        # Use L2 norm as salience measure
        l2_norm = torch.norm(state_tensor, p=2).item()
        # Normalize to [0, 1] range
        return min(l2_norm / (self.n_embd ** 0.5), 1.0)
    
    def _extract_autonomy(self, state_tensor: torch.Tensor, layer_idx: int) -> float:
        """Extract autonomy index based on layer depth and state complexity"""
        # Higher layers have more autonomy
        depth_factor = layer_idx / max(self.n_layer - 1, 1)
        
        # State complexity as autonomy measure
        entropy = self._calculate_entropy(state_tensor)
        complexity_factor = entropy / 10.0  # Normalize entropy
        
        return min((depth_factor + complexity_factor) / 2.0, 1.0)
    
    def _calculate_entropy(self, tensor: torch.Tensor) -> float:
        """Calculate entropy of tensor values as complexity measure"""
        # Convert to probabilities
        probs = torch.softmax(tensor.flatten(), dim=0)
        # Calculate entropy
        log_probs = torch.log(probs + 1e-12)  # Add small epsilon
        entropy = -torch.sum(probs * log_probs).item()
        return entropy
    
    def to_rwkv_state(self) -> torch.Tensor:
        """
        Convert cognitive primitives back to RWKV state format
        """
        if self.rwkv_state is None:
            # Initialize state if not available
            state_size = self.n_layer * 3  # Simplified for demo
            self.rwkv_state = torch.zeros(state_size, self.n_embd)
        
        # Update state based on primitives
        for name, primitive in self.primitives.items():
            if name.startswith("layer_"):
                layer_idx = int(name.split("_")[1])
                if layer_idx < self.n_layer:
                    # Update corresponding state slice
                    state_idx = layer_idx * 3 + 1  # Attention state
                    if state_idx < self.rwkv_state.shape[0]:
                        # Encode primitive back to state tensor
                        encoded_state = self._encode_primitive_to_state(primitive)
                        self.rwkv_state[state_idx] = encoded_state
        
        return self.rwkv_state
    
    def _encode_primitive_to_state(self, primitive: TensorPrimitive) -> torch.Tensor:
        """Encode cognitive primitive back to state tensor"""
        # Create state tensor from primitive dimensions
        state = torch.zeros(self.n_embd)
        
        # Encode modality as periodic pattern
        modality_pattern = torch.sin(torch.arange(self.n_embd, dtype=torch.float32) * 
                                   (primitive.modality + 1) * 0.1)
        
        # Encode depth as amplitude scaling
        depth_scale = (primitive.depth + 1) / self.n_layer
        
        # Encode context as frequency modulation
        context_freq = primitive.context * 0.01
        context_pattern = torch.cos(torch.arange(self.n_embd, dtype=torch.float32) * context_freq)
        
        # Combine with salience and autonomy
        state = (modality_pattern * depth_scale + 
                context_pattern * primitive.salience) * primitive.autonomy_index
        
        return state
    
    def get_prime_factorization_mapping(self) -> Dict[str, Any]:
        """
        Document the prime factorization mapping between tensor dimensions
        and RWKV state components
        """
        return {
            "modality_primes": [2, 3, 5, 7, 11, 13, 17],  # Maps to ModalityType
            "depth_factor": "layer_index / n_layer",
            "context_encoding": "variance_based_quantization", 
            "salience_mapping": "l2_norm_normalized",
            "autonomy_calculation": "(depth_factor + entropy_factor) / 2",
            "state_reconstruction": {
                "modality": "sin_pattern_with_prime_frequency",
                "depth": "amplitude_scaling", 
                "context": "frequency_modulation",
                "salience": "pattern_weighting",
                "autonomy": "global_scaling"
            }
        }