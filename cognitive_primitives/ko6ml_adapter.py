"""
Ko6ml Adapter Module

Provides bidirectional translation between ko6ml primitives and 
AtomSpace hypergraph patterns for agentic grammar processing.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json

from .tensor_primitives import TensorPrimitive, CognitiveState
from .hypergraph_encoding import HypergraphEncoder, AtomSpaceNode, AtomType

@dataclass 
class Ko6mlPrimitive:
    """
    Represents a ko6ml primitive with semantic and syntactic properties
    """
    symbol: str
    semantic_role: str
    syntactic_category: str
    tensor_binding: torch.Tensor
    cognitive_features: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "semantic_role": self.semantic_role,
            "syntactic_category": self.syntactic_category,
            "tensor_binding": self.tensor_binding.tolist(),
            "cognitive_features": self.cognitive_features
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Ko6mlPrimitive':
        return cls(
            symbol=data["symbol"],
            semantic_role=data["semantic_role"],
            syntactic_category=data["syntactic_category"],
            tensor_binding=torch.tensor(data["tensor_binding"]),
            cognitive_features=data["cognitive_features"]
        )

class Ko6mlAdapter:
    """
    Bidirectional adapter between ko6ml primitives and AtomSpace hypergraphs
    """
    
    def __init__(self):
        self.hypergraph_encoder = HypergraphEncoder()
        self.primitive_registry: Dict[str, Ko6mlPrimitive] = {}
        self.translation_cache: Dict[str, Any] = {}
        
        # Initialize common ko6ml primitive patterns
        self._initialize_base_primitives()
    
    def _initialize_base_primitives(self):
        """Initialize basic ko6ml primitives for agentic grammar"""
        base_primitives = [
            {
                "symbol": "AGENT",
                "semantic_role": "actor",
                "syntactic_category": "noun_phrase",
                "features": {"autonomy": 0.8, "agency": 0.9, "intentionality": 0.7}
            },
            {
                "symbol": "ACTION", 
                "semantic_role": "predicate",
                "syntactic_category": "verb_phrase",
                "features": {"dynamism": 0.9, "transitivity": 0.6, "completion": 0.5}
            },
            {
                "symbol": "GOAL",
                "semantic_role": "target",
                "syntactic_category": "noun_phrase", 
                "features": {"desirability": 0.8, "achievability": 0.6, "specificity": 0.7}
            },
            {
                "symbol": "CONTEXT",
                "semantic_role": "circumstance",
                "syntactic_category": "adverbial",
                "features": {"relevance": 0.7, "temporality": 0.5, "locality": 0.6}
            },
            {
                "symbol": "BELIEF",
                "semantic_role": "epistemic",
                "syntactic_category": "modal",
                "features": {"certainty": 0.6, "evidence": 0.5, "consistency": 0.7}
            }
        ]
        
        for primitive_def in base_primitives:
            # Create tensor binding based on features
            feature_vector = torch.tensor(list(primitive_def["features"].values()))
            # Expand to 5D cognitive tensor format [modality, depth, context, salience, autonomy]
            tensor_binding = torch.zeros(5)
            tensor_binding[0] = 0  # Linguistic modality
            tensor_binding[1] = 1  # Base depth
            tensor_binding[2] = len(primitive_def["symbol"])  # Context from symbol length
            tensor_binding[3] = torch.mean(feature_vector).item()  # Salience from features
            tensor_binding[4] = primitive_def["features"].get("autonomy", 0.5)  # Autonomy
            
            primitive = Ko6mlPrimitive(
                symbol=primitive_def["symbol"],
                semantic_role=primitive_def["semantic_role"],
                syntactic_category=primitive_def["syntactic_category"],
                tensor_binding=tensor_binding,
                cognitive_features=primitive_def["features"]
            )
            
            self.primitive_registry[primitive_def["symbol"]] = primitive
    
    def ko6ml_to_atomspace(self, ko6ml_expression: List[str]) -> Dict[str, Any]:
        """
        Translate ko6ml expression to AtomSpace hypergraph
        
        Args:
            ko6ml_expression: List of ko6ml symbols/primitives
            
        Returns:
            Hypergraph representation
        """
        cache_key = "_".join(ko6ml_expression)
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        # Clear previous state
        self.hypergraph_encoder = HypergraphEncoder()
        
        atomspace_nodes = []
        
        # Convert each ko6ml symbol to AtomSpace node
        for symbol in ko6ml_expression:
            if symbol in self.primitive_registry:
                primitive = self.primitive_registry[symbol]
                
                # Create ConceptNode for the symbol
                concept_node = AtomSpaceNode(
                    atom_type=AtomType.CONCEPT_NODE,
                    name=f"concept_{symbol.lower()}",
                    tensor_embedding=primitive.tensor_binding,
                    strength=torch.mean(torch.tensor(list(primitive.cognitive_features.values()))).item(),
                    confidence=primitive.cognitive_features.get("certainty", 0.7),
                    metadata={
                        "symbol": symbol,
                        "semantic_role": primitive.semantic_role,
                        "syntactic_category": primitive.syntactic_category,
                        "cognitive_features": primitive.cognitive_features
                    }
                )
                
                # Add concept node to encoder
                self.hypergraph_encoder.nodes[concept_node.name] = concept_node
                
                # Create TensorNode for the primitive
                tensor_node = self.hypergraph_encoder.encode_tensor_primitive(
                    primitive.tensor_binding,
                    name=f"tensor_{symbol.lower()}"
                )
                
                # Create InheritanceLink connecting concept to tensor
                inheritance_link = self.hypergraph_encoder.create_inheritance_link(
                    tensor_node, concept_node
                )
                
                atomspace_nodes.append(concept_node)
                atomspace_nodes.append(tensor_node)
        
        # Create relationships between nodes based on co-occurrence
        for i in range(len(atomspace_nodes) - 1):
            for j in range(i + 1, len(atomspace_nodes)):
                if atomspace_nodes[i].atom_type == AtomType.CONCEPT_NODE and \
                   atomspace_nodes[j].atom_type == AtomType.CONCEPT_NODE:
                    # Create similarity link between concepts
                    self.hypergraph_encoder.create_similarity_link(
                        atomspace_nodes[i], atomspace_nodes[j]
                    )
        
        # Create evaluation links for semantic relationships
        if len(ko6ml_expression) >= 2:
            self._create_semantic_evaluation_links(ko6ml_expression, atomspace_nodes)
        
        hypergraph = self.hypergraph_encoder.get_hypergraph_fragment()
        self.translation_cache[cache_key] = hypergraph
        
        return hypergraph
    
    def _create_semantic_evaluation_links(self, expression: List[str], nodes: List[AtomSpaceNode]):
        """Create EvaluationLinks for semantic relationships in the expression"""
        concept_nodes = [node for node in nodes if node.atom_type == AtomType.CONCEPT_NODE]
        
        # Look for AGENT-ACTION-GOAL patterns
        agent_nodes = [node for node in concept_nodes if "agent" in node.name]
        action_nodes = [node for node in concept_nodes if "action" in node.name]
        goal_nodes = [node for node in concept_nodes if "goal" in node.name]
        
        # Create evaluation links for agent-action relationships
        for agent in agent_nodes:
            for action in action_nodes:
                predicate_name = f"performs_{agent.name}_{action.name}"
                
                # Create predicate node
                predicate_node = AtomSpaceNode(
                    atom_type=AtomType.PREDICATE_NODE,
                    name=predicate_name,
                    tensor_embedding=torch.cat([agent.tensor_embedding, action.tensor_embedding]),
                    strength=(agent.strength + action.strength) / 2,
                    confidence=(agent.confidence + action.confidence) / 2,
                    metadata={
                        "predicate_type": "performs",
                        "subject": agent.name,
                        "object": action.name
                    }
                )
                
                # Create evaluation link
                evaluation_link = AtomSpaceNode(
                    atom_type=AtomType.EVALUATION_LINK,
                    name=f"eval_{predicate_name}",
                    tensor_embedding=predicate_node.tensor_embedding,
                    strength=predicate_node.strength,
                    confidence=predicate_node.confidence,
                    children=[predicate_node, agent, action],
                    metadata={
                        "evaluation_type": "agent_action",
                        "predicate": predicate_name
                    }
                )
                
                self.hypergraph_encoder.links.append(evaluation_link)
    
    def atomspace_to_ko6ml(self, hypergraph: Dict[str, Any]) -> List[str]:
        """
        Translate AtomSpace hypergraph back to ko6ml expression
        
        Args:
            hypergraph: Hypergraph representation
            
        Returns:
            List of ko6ml symbols
        """
        ko6ml_symbols = []
        
        # Extract ConceptNodes and map back to ko6ml symbols
        for node_data in hypergraph.get("nodes", []):
            if node_data.get("atom_type") == AtomType.CONCEPT_NODE.value:
                symbol = node_data.get("metadata", {}).get("symbol")
                if symbol:
                    ko6ml_symbols.append(symbol)
        
        # Sort by semantic roles for proper ordering
        role_order = ["actor", "predicate", "target", "circumstance", "epistemic"]
        ko6ml_symbols.sort(key=lambda s: self._get_role_priority(s, role_order))
        
        return ko6ml_symbols
    
    def _get_role_priority(self, symbol: str, role_order: List[str]) -> int:
        """Get priority for symbol based on semantic role"""
        if symbol in self.primitive_registry:
            role = self.primitive_registry[symbol].semantic_role
            try:
                return role_order.index(role)
            except ValueError:
                return len(role_order)
        return len(role_order)
    
    def round_trip_test(self, ko6ml_expression: List[str]) -> Dict[str, Any]:
        """
        Perform round-trip translation test: ko6ml -> AtomSpace -> ko6ml
        
        Args:
            ko6ml_expression: Original ko6ml expression
            
        Returns:
            Test results with fidelity metrics
        """
        # Forward translation
        hypergraph = self.ko6ml_to_atomspace(ko6ml_expression)
        
        # Backward translation  
        reconstructed_ko6ml = self.atomspace_to_ko6ml(hypergraph)
        
        # Calculate fidelity metrics
        original_set = set(ko6ml_expression)
        reconstructed_set = set(reconstructed_ko6ml)
        
        precision = len(original_set & reconstructed_set) / len(reconstructed_set) if reconstructed_set else 0
        recall = len(original_set & reconstructed_set) / len(original_set) if original_set else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Order preservation score
        order_score = self._calculate_order_preservation(ko6ml_expression, reconstructed_ko6ml)
        
        return {
            "original_expression": ko6ml_expression,
            "hypergraph_nodes": len(hypergraph.get("nodes", [])),
            "hypergraph_links": len(hypergraph.get("links", [])),
            "reconstructed_expression": reconstructed_ko6ml,
            "fidelity_metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "order_preservation": order_score
            },
            "tensor_consistency": self._check_tensor_consistency(hypergraph)
        }
    
    def _calculate_order_preservation(self, original: List[str], reconstructed: List[str]) -> float:
        """Calculate how well order is preserved in reconstruction"""
        if not original or not reconstructed:
            return 0.0
        
        # Count matching adjacent pairs
        original_pairs = set(zip(original[:-1], original[1:]))
        reconstructed_pairs = set(zip(reconstructed[:-1], reconstructed[1:]))
        
        if not original_pairs:
            return 1.0
        
        matching_pairs = len(original_pairs & reconstructed_pairs)
        return matching_pairs / len(original_pairs)
    
    def _check_tensor_consistency(self, hypergraph: Dict[str, Any]) -> Dict[str, float]:
        """Check consistency of tensor representations in hypergraph"""
        tensor_nodes = [node for node in hypergraph.get("nodes", []) 
                       if node.get("atom_type") == AtomType.TENSOR_NODE.value]
        
        if not tensor_nodes:
            return {"mean_magnitude": 0.0, "variance": 0.0, "validity": 0.0}
        
        magnitudes = []
        valid_tensors = 0
        
        for node in tensor_nodes:
            tensor_data = node.get("tensor_data", [])
            if len(tensor_data) == 5:  # Valid 5D cognitive tensor
                valid_tensors += 1
                magnitude = np.linalg.norm(tensor_data)
                magnitudes.append(magnitude)
        
        return {
            "mean_magnitude": float(np.mean(magnitudes)) if magnitudes else 0.0,
            "variance": float(np.var(magnitudes)) if magnitudes else 0.0, 
            "validity": valid_tensors / len(tensor_nodes) if tensor_nodes else 0.0
        }
    
    def get_primitive_registry(self) -> Dict[str, Dict[str, Any]]:
        """Get current primitive registry as dictionary"""
        return {symbol: primitive.to_dict() 
                for symbol, primitive in self.primitive_registry.items()}
    
    def add_custom_primitive(self, symbol: str, semantic_role: str, 
                           syntactic_category: str, cognitive_features: Dict[str, float]):
        """Add a custom ko6ml primitive to the registry"""
        # Create tensor binding from features
        feature_values = list(cognitive_features.values())
        tensor_binding = torch.zeros(5)
        tensor_binding[0] = 0  # Default to linguistic modality
        tensor_binding[1] = 2  # Custom primitives at depth 2
        tensor_binding[2] = len(symbol)  # Context from symbol length
        tensor_binding[3] = np.mean(feature_values) if feature_values else 0.5  # Salience
        tensor_binding[4] = cognitive_features.get("autonomy", 0.5)  # Autonomy
        
        primitive = Ko6mlPrimitive(
            symbol=symbol,
            semantic_role=semantic_role,
            syntactic_category=syntactic_category,
            tensor_binding=tensor_binding,
            cognitive_features=cognitive_features
        )
        
        self.primitive_registry[symbol] = primitive