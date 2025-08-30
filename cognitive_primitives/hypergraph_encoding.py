"""
Hypergraph Encoding Module

Provides encoding and decoding between cognitive primitives and AtomSpace
hypergraph representations for agentic grammar processing.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json

class AtomType(Enum):
    """AtomSpace atom types for cognitive representation"""
    CONCEPT_NODE = "ConceptNode"
    PREDICATE_NODE = "PredicateNode"
    EVALUATION_LINK = "EvaluationLink"
    INHERITANCE_LINK = "InheritanceLink"
    SIMILARITY_LINK = "SimilarityLink"
    IMPLICATION_LINK = "ImplicationLink"
    TENSOR_NODE = "TensorNode"
    COGNITIVE_STATE_LINK = "CognitiveStateLink"

@dataclass
class AtomSpaceNode:
    """
    Represents an AtomSpace node/link with tensor embedding
    """
    atom_type: AtomType
    name: str
    tensor_embedding: torch.Tensor
    strength: float = 1.0
    confidence: float = 1.0
    children: List['AtomSpaceNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.tensor_embedding.dim() == 0:
            # Convert scalar to 1D tensor
            self.tensor_embedding = self.tensor_embedding.unsqueeze(0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "atom_type": self.atom_type.value,
            "name": self.name,
            "tensor_shape": list(self.tensor_embedding.shape),
            "tensor_data": self.tensor_embedding.tolist(),
            "strength": self.strength,
            "confidence": self.confidence,
            "children": [child.to_dict() for child in self.children],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AtomSpaceNode':
        """Create from dictionary representation"""
        node = cls(
            atom_type=AtomType(data["atom_type"]),
            name=data["name"],
            tensor_embedding=torch.tensor(data["tensor_data"]),
            strength=data["strength"],
            confidence=data["confidence"],
            metadata=data["metadata"]
        )
        node.children = [cls.from_dict(child) for child in data["children"]]
        return node

class HypergraphEncoder:
    """
    Encodes cognitive primitives as hypergraph nodes and links
    with tensor shape: [modality, depth, context, salience, autonomy_index]
    """
    
    def __init__(self):
        self.nodes: Dict[str, AtomSpaceNode] = {}
        self.links: List[AtomSpaceNode] = []
        self.node_id_counter = 0
        
    def encode_tensor_primitive(self, primitive_tensor: torch.Tensor, 
                              name: Optional[str] = None) -> AtomSpaceNode:
        """
        Encode a tensor primitive as an AtomSpace TensorNode
        
        Args:
            primitive_tensor: Tensor with shape [modality, depth, context, salience, autonomy_index]
            name: Optional name for the node
        
        Returns:
            AtomSpaceNode representing the primitive
        """
        if name is None:
            name = f"tensor_primitive_{self.node_id_counter}"
            self.node_id_counter += 1
            
        # Extract dimensions
        if primitive_tensor.dim() == 1 and primitive_tensor.shape[0] == 5:
            modality, depth, context, salience, autonomy = primitive_tensor.tolist()
        else:
            raise ValueError(f"Expected tensor shape [5], got {primitive_tensor.shape}")
        
        # Create TensorNode
        tensor_node = AtomSpaceNode(
            atom_type=AtomType.TENSOR_NODE,
            name=name,
            tensor_embedding=primitive_tensor,
            strength=float(salience),
            confidence=float(autonomy),
            metadata={
                "modality": int(modality),
                "depth": int(depth), 
                "context": int(context),
                "salience": float(salience),
                "autonomy_index": float(autonomy)
            }
        )
        
        self.nodes[name] = tensor_node
        return tensor_node
    
    def create_cognitive_state_link(self, agent_primitives: List[AtomSpaceNode],
                                  name: Optional[str] = None) -> AtomSpaceNode:
        """
        Create a CognitiveStateLink connecting multiple tensor primitives
        representing an agent's state
        """
        if name is None:
            name = f"cognitive_state_{self.node_id_counter}"
            self.node_id_counter += 1
            
        # Aggregate tensor embeddings
        if agent_primitives:
            aggregated_tensor = torch.stack([node.tensor_embedding for node in agent_primitives])
            state_embedding = torch.mean(aggregated_tensor, dim=0)
        else:
            state_embedding = torch.zeros(5)
            
        # Calculate aggregate strength and confidence
        avg_strength = np.mean([node.strength for node in agent_primitives]) if agent_primitives else 0.0
        avg_confidence = np.mean([node.confidence for node in agent_primitives]) if agent_primitives else 0.0
        
        cognitive_link = AtomSpaceNode(
            atom_type=AtomType.COGNITIVE_STATE_LINK,
            name=name,
            tensor_embedding=state_embedding,
            strength=avg_strength,
            confidence=avg_confidence,
            children=agent_primitives,
            metadata={
                "num_primitives": len(agent_primitives),
                "aggregation_method": "mean"
            }
        )
        
        self.links.append(cognitive_link)
        return cognitive_link
    
    def create_inheritance_link(self, child_node: AtomSpaceNode, 
                               parent_node: AtomSpaceNode) -> AtomSpaceNode:
        """Create an InheritanceLink between two nodes"""
        link_name = f"inheritance_{child_node.name}_{parent_node.name}"
        
        # Combine embeddings for link representation
        combined_embedding = torch.cat([child_node.tensor_embedding, parent_node.tensor_embedding])
        
        inheritance_link = AtomSpaceNode(
            atom_type=AtomType.INHERITANCE_LINK,
            name=link_name,
            tensor_embedding=combined_embedding,
            strength=min(child_node.strength, parent_node.strength),
            confidence=min(child_node.confidence, parent_node.confidence),
            children=[child_node, parent_node],
            metadata={
                "link_type": "inheritance",
                "child": child_node.name,
                "parent": parent_node.name
            }
        )
        
        self.links.append(inheritance_link)
        return inheritance_link
    
    def create_similarity_link(self, node1: AtomSpaceNode, 
                              node2: AtomSpaceNode) -> AtomSpaceNode:
        """Create a SimilarityLink between two nodes"""
        link_name = f"similarity_{node1.name}_{node2.name}"
        
        # Calculate similarity based on tensor distance
        similarity_score = self._calculate_tensor_similarity(
            node1.tensor_embedding, node2.tensor_embedding)
        
        # Combine embeddings
        combined_embedding = torch.cat([node1.tensor_embedding, node2.tensor_embedding])
        
        similarity_link = AtomSpaceNode(
            atom_type=AtomType.SIMILARITY_LINK,
            name=link_name,
            tensor_embedding=combined_embedding,
            strength=similarity_score,
            confidence=(node1.confidence + node2.confidence) / 2,
            children=[node1, node2],
            metadata={
                "link_type": "similarity",
                "similarity_score": similarity_score
            }
        )
        
        self.links.append(similarity_link)
        return similarity_link
    
    def _calculate_tensor_similarity(self, tensor1: torch.Tensor, 
                                   tensor2: torch.Tensor) -> float:
        """Calculate similarity between two tensors using cosine similarity"""
        # Ensure tensors are the same size
        min_size = min(tensor1.shape[0], tensor2.shape[0])
        t1_trimmed = tensor1[:min_size]
        t2_trimmed = tensor2[:min_size]
        
        # Calculate cosine similarity
        dot_product = torch.dot(t1_trimmed, t2_trimmed)
        norm1 = torch.norm(t1_trimmed)
        norm2 = torch.norm(t2_trimmed)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def decode_to_primitives(self, hypergraph: Dict[str, Any]) -> List[torch.Tensor]:
        """
        Decode hypergraph representation back to tensor primitives
        
        Args:
            hypergraph: Dictionary representation of the hypergraph
            
        Returns:
            List of tensor primitives
        """
        primitives = []
        
        # Extract TensorNodes
        for node_data in hypergraph.get("nodes", []):
            if node_data.get("atom_type") == AtomType.TENSOR_NODE.value:
                tensor_data = node_data.get("tensor_data", [])
                if len(tensor_data) == 5:  # Validate primitive shape
                    primitive_tensor = torch.tensor(tensor_data)
                    primitives.append(primitive_tensor)
        
        return primitives
    
    def get_hypergraph_fragment(self) -> Dict[str, Any]:
        """
        Export current hypergraph as a structured fragment
        """
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "links": [link.to_dict() for link in self.links],
            "metadata": {
                "total_nodes": len(self.nodes),
                "total_links": len(self.links),
                "encoding_version": "1.0"
            }
        }
    
    def load_hypergraph_fragment(self, fragment: Dict[str, Any]):
        """
        Load hypergraph fragment from structured representation
        """
        self.nodes.clear()
        self.links.clear()
        
        # Load nodes
        for node_data in fragment.get("nodes", []):
            node = AtomSpaceNode.from_dict(node_data)
            self.nodes[node.name] = node
            
        # Load links
        for link_data in fragment.get("links", []):
            link = AtomSpaceNode.from_dict(link_data)
            self.links.append(link)
    
    def visualize_fragment_structure(self) -> Dict[str, Any]:
        """
        Generate structure for hypergraph fragment visualization
        """
        visualization_data = {
            "nodes": [],
            "edges": [],
            "clusters": {}
        }
        
        # Add nodes for visualization
        for name, node in self.nodes.items():
            visualization_data["nodes"].append({
                "id": name,
                "label": name,
                "type": node.atom_type.value,
                "strength": node.strength,
                "confidence": node.confidence,
                "modality": node.metadata.get("modality", 0)
            })
        
        # Add links as edges
        for link in self.links:
            if len(link.children) >= 2:
                for i in range(len(link.children) - 1):
                    for j in range(i + 1, len(link.children)):
                        visualization_data["edges"].append({
                            "source": link.children[i].name,
                            "target": link.children[j].name,
                            "type": link.atom_type.value,
                            "strength": link.strength
                        })
        
        # Group by modality for clustering
        for node_name, node in self.nodes.items():
            modality = node.metadata.get("modality", 0)
            if modality not in visualization_data["clusters"]:
                visualization_data["clusters"][modality] = []
            visualization_data["clusters"][modality].append(node_name)
        
        return visualization_data