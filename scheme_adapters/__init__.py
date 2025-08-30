"""
Scheme Adapters Module

Modular Scheme adapters for agentic grammar AtomSpace integration.
Provides microservices for bidirectional translation between Scheme
S-expressions and hypergraph representations.
"""

import re
import json
import torch
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_primitives.hypergraph_encoding import AtomSpaceNode, AtomType, HypergraphEncoder

class SchemeTokenType(Enum):
    """Token types in Scheme expressions"""
    OPEN_PAREN = "("
    CLOSE_PAREN = ")"
    SYMBOL = "symbol"
    STRING = "string"
    NUMBER = "number"
    ATOM_TYPE = "atom_type"

@dataclass
class SchemeToken:
    """Represents a token in Scheme expression"""
    token_type: SchemeTokenType
    value: str
    position: int

class SchemeLexer:
    """Lexical analyzer for Scheme expressions"""
    
    def __init__(self):
        self.token_patterns = [
            (SchemeTokenType.OPEN_PAREN, r'\('),
            (SchemeTokenType.CLOSE_PAREN, r'\)'),
            (SchemeTokenType.STRING, r'"[^"]*"'),
            (SchemeTokenType.NUMBER, r'-?\d+(\.\d+)?'),
            (SchemeTokenType.ATOM_TYPE, r'[A-Z][a-zA-Z]*'),
            (SchemeTokenType.SYMBOL, r'[a-zA-Z_][a-zA-Z0-9_\-]*'),
        ]
        
    def tokenize(self, text: str) -> List[SchemeToken]:
        """Tokenize Scheme expression into tokens"""
        tokens = []
        position = 0
        
        while position < len(text):
            # Skip whitespace
            if text[position].isspace():
                position += 1
                continue
                
            # Try to match token patterns
            matched = False
            for token_type, pattern in self.token_patterns:
                regex = re.compile(pattern)
                match = regex.match(text, position)
                
                if match:
                    value = match.group(0)
                    tokens.append(SchemeToken(token_type, value, position))
                    position = match.end()
                    matched = True
                    break
            
            if not matched:
                raise ValueError(f"Unexpected character at position {position}: {text[position]}")
        
        return tokens

@dataclass
class SchemeExpression:
    """Represents a parsed Scheme S-expression"""
    atom_type: Optional[str] = None
    name: Optional[str] = None
    children: List['SchemeExpression'] = None
    raw_value: Optional[str] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def is_atom(self) -> bool:
        """Check if this is an atomic expression"""
        return not self.children
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "atom_type": self.atom_type,
            "name": self.name,
            "raw_value": self.raw_value,
            "children": [child.to_dict() for child in self.children]
        }

class SchemeParser:
    """Parser for Scheme S-expressions into structured form"""
    
    def __init__(self):
        self.lexer = SchemeLexer()
        
    def parse(self, scheme_text: str) -> SchemeExpression:
        """Parse Scheme text into expression tree"""
        tokens = self.lexer.tokenize(scheme_text.strip())
        
        if not tokens:
            raise ValueError("Empty expression")
        
        expression, _ = self._parse_expression(tokens, 0)
        return expression
    
    def _parse_expression(self, tokens: List[SchemeToken], 
                         start_pos: int) -> Tuple[SchemeExpression, int]:
        """Parse expression starting at given position"""
        if start_pos >= len(tokens):
            raise ValueError("Unexpected end of expression")
        
        token = tokens[start_pos]
        
        if token.token_type == SchemeTokenType.OPEN_PAREN:
            # Parse list expression
            return self._parse_list(tokens, start_pos)
        else:
            # Parse atomic expression
            expression = SchemeExpression()
            
            if token.token_type == SchemeTokenType.ATOM_TYPE:
                expression.atom_type = token.value
            elif token.token_type == SchemeTokenType.SYMBOL:
                expression.name = token.value
            elif token.token_type in [SchemeTokenType.STRING, SchemeTokenType.NUMBER]:
                expression.raw_value = token.value
            
            return expression, start_pos + 1
    
    def _parse_list(self, tokens: List[SchemeToken], 
                   start_pos: int) -> Tuple[SchemeExpression, int]:
        """Parse list expression starting with open paren"""
        if tokens[start_pos].token_type != SchemeTokenType.OPEN_PAREN:
            raise ValueError("Expected open parenthesis")
        
        expression = SchemeExpression()
        pos = start_pos + 1
        
        # First element might be atom type
        if pos < len(tokens) and tokens[pos].token_type == SchemeTokenType.ATOM_TYPE:
            expression.atom_type = tokens[pos].value
            pos += 1
        
        # Parse children until close paren
        while pos < len(tokens) and tokens[pos].token_type != SchemeTokenType.CLOSE_PAREN:
            child_expr, pos = self._parse_expression(tokens, pos)
            expression.children.append(child_expr)
        
        if pos >= len(tokens):
            raise ValueError("Missing closing parenthesis")
        
        return expression, pos + 1

class AtomSpaceSchemeAdapter:
    """
    Bidirectional adapter between AtomSpace hypergraphs and Scheme S-expressions
    """
    
    def __init__(self):
        self.parser = SchemeParser()
        self.hypergraph_encoder = HypergraphEncoder()
        
    def scheme_to_atomspace(self, scheme_text: str) -> Dict[str, Any]:
        """
        Convert Scheme S-expression to AtomSpace hypergraph
        
        Args:
            scheme_text: Scheme expression as string
            
        Returns:
            Hypergraph representation
        """
        # Parse Scheme expression
        expression = self.parser.parse(scheme_text)
        
        # Convert to AtomSpace representation
        atomspace_node = self._expression_to_atomspace(expression)
        
        # Build hypergraph
        if atomspace_node:
            self.hypergraph_encoder.nodes[atomspace_node.name] = atomspace_node
            self._add_recursive_nodes(atomspace_node)
        
        return self.hypergraph_encoder.get_hypergraph_fragment()
    
    def _expression_to_atomspace(self, expression: SchemeExpression) -> Optional[AtomSpaceNode]:
        """Convert Scheme expression to AtomSpace node"""
        if expression.raw_value is not None:
            # Atomic value - create ConceptNode
            return AtomSpaceNode(
                atom_type=AtomType.CONCEPT_NODE,
                name=f"value_{expression.raw_value}",
                tensor_embedding=self._create_value_embedding(expression.raw_value),
                metadata={"raw_value": expression.raw_value}
            )
        
        elif expression.atom_type:
            # Typed expression
            atom_type = self._scheme_type_to_atomspace(expression.atom_type)
            name = expression.name or f"{expression.atom_type.lower()}_{len(self.hypergraph_encoder.nodes)}"
            
            # Create tensor embedding based on expression structure
            tensor_embedding = self._create_expression_embedding(expression)
            
            node = AtomSpaceNode(
                atom_type=atom_type,
                name=name,
                tensor_embedding=tensor_embedding,
                metadata={
                    "scheme_type": expression.atom_type,
                    "original_name": expression.name
                }
            )
            
            # Process children
            for child_expr in expression.children:
                child_node = self._expression_to_atomspace(child_expr)
                if child_node:
                    node.children.append(child_node)
            
            return node
        
        elif expression.name:
            # Named expression
            return AtomSpaceNode(
                atom_type=AtomType.CONCEPT_NODE,
                name=expression.name,
                tensor_embedding=self._create_name_embedding(expression.name),
                metadata={"scheme_name": expression.name}
            )
        
        return None
    
    def _scheme_type_to_atomspace(self, scheme_type: str) -> AtomType:
        """Map Scheme atom type to AtomSpace type"""
        type_mapping = {
            "ConceptNode": AtomType.CONCEPT_NODE,
            "PredicateNode": AtomType.PREDICATE_NODE,
            "EvaluationLink": AtomType.EVALUATION_LINK,
            "InheritanceLink": AtomType.INHERITANCE_LINK,
            "SimilarityLink": AtomType.SIMILARITY_LINK,
            "ImplicationLink": AtomType.IMPLICATION_LINK,
            "TensorNode": AtomType.TENSOR_NODE,
            "CognitiveStateLink": AtomType.COGNITIVE_STATE_LINK
        }
        
        return type_mapping.get(scheme_type, AtomType.CONCEPT_NODE)
    
    def _create_value_embedding(self, value: str) -> torch.Tensor:
        """Create tensor embedding for atomic value"""
        
        # Simple hash-based embedding
        hash_val = hash(value) % 10000
        embedding = torch.zeros(5)
        embedding[0] = 0  # Linguistic modality
        embedding[1] = 0  # Depth 0 for atomic values
        embedding[2] = len(str(value))  # Context from value length
        embedding[3] = (hash_val % 100) / 100.0  # Salience from hash
        embedding[4] = 0.1  # Low autonomy for atomic values
        
        return embedding
    
    def _create_name_embedding(self, name: str) -> torch.Tensor:
        """Create tensor embedding for named concept"""
        
        hash_val = hash(name) % 10000
        embedding = torch.zeros(5)
        embedding[0] = 0  # Linguistic modality
        embedding[1] = 1  # Depth 1 for named concepts
        embedding[2] = len(name)  # Context from name length
        embedding[3] = min((hash_val % 100) / 100.0 + 0.5, 1.0)  # Higher salience
        embedding[4] = 0.5  # Medium autonomy
        
        return embedding
    
    def _create_expression_embedding(self, expression: SchemeExpression) -> torch.Tensor:
        """Create tensor embedding for complex expression"""
        
        embedding = torch.zeros(5)
        embedding[0] = 0  # Linguistic modality
        embedding[1] = min(len(expression.children) + 1, 10)  # Depth from structure
        embedding[2] = self._calculate_expression_complexity(expression)  # Context complexity
        embedding[3] = min(0.7 + len(expression.children) * 0.1, 1.0)  # Salience from complexity
        embedding[4] = min(0.3 + len(expression.children) * 0.15, 1.0)  # Autonomy from structure
        
        return embedding
    
    def _calculate_expression_complexity(self, expression: SchemeExpression) -> int:
        """Calculate complexity score for expression"""
        complexity = 1  # Base complexity
        
        for child in expression.children:
            complexity += self._calculate_expression_complexity(child)
        
        return min(complexity, 100)  # Cap complexity
    
    def _add_recursive_nodes(self, node: AtomSpaceNode):
        """Recursively add all child nodes to hypergraph encoder"""
        for child in node.children:
            if child.name not in self.hypergraph_encoder.nodes:
                self.hypergraph_encoder.nodes[child.name] = child
                self._add_recursive_nodes(child)
            
            # Create links between parent and children
            if node.atom_type in [AtomType.EVALUATION_LINK, AtomType.INHERITANCE_LINK, 
                                 AtomType.SIMILARITY_LINK, AtomType.IMPLICATION_LINK]:
                link = AtomSpaceNode(
                    atom_type=node.atom_type,
                    name=f"link_{node.name}_{child.name}",
                    tensor_embedding=torch.cat([node.tensor_embedding, child.tensor_embedding]),
                    children=[node, child],
                    metadata={"link_type": node.atom_type.value}
                )
                self.hypergraph_encoder.links.append(link)
    
    def atomspace_to_scheme(self, hypergraph: Dict[str, Any]) -> str:
        """
        Convert AtomSpace hypergraph back to Scheme S-expression
        
        Args:
            hypergraph: Hypergraph representation
            
        Returns:
            Scheme S-expression as string
        """
        # Find root nodes (nodes not referenced as children)
        all_child_names = set()
        for link_data in hypergraph.get("links", []):
            for child_data in link_data.get("children", []):
                all_child_names.add(child_data.get("name", ""))
        
        root_nodes = []
        for node_data in hypergraph.get("nodes", []):
            if node_data.get("name") not in all_child_names:
                root_nodes.append(node_data)
        
        if not root_nodes:
            # Use first node as root if no clear root
            root_nodes = hypergraph.get("nodes", [])[:1]
        
        # Convert each root to Scheme expression
        scheme_expressions = []
        for root_data in root_nodes:
            scheme_expr = self._atomspace_to_scheme_expression(root_data, hypergraph)
            scheme_expressions.append(scheme_expr)
        
        if len(scheme_expressions) == 1:
            return scheme_expressions[0]
        else:
            # Wrap multiple expressions in a list
            return f"({' '.join(scheme_expressions)})"
    
    def _atomspace_to_scheme_expression(self, node_data: Dict[str, Any], 
                                      hypergraph: Dict[str, Any]) -> str:
        """Convert single AtomSpace node to Scheme expression"""
        atom_type = node_data.get("atom_type", "ConceptNode")
        name = node_data.get("name", "unknown")
        metadata = node_data.get("metadata", {})
        
        # Handle atomic values
        if "raw_value" in metadata:
            return metadata["raw_value"]
        
        # Handle named concepts
        if atom_type == "ConceptNode" and "scheme_name" in metadata:
            return metadata["scheme_name"]
        
        # Handle complex expressions
        children_data = node_data.get("children", [])
        if not children_data:
            # Leaf node
            return metadata.get("scheme_name", name)
        
        # Build expression with children
        child_expressions = []
        for child_data in children_data:
            child_expr = self._atomspace_to_scheme_expression(child_data, hypergraph)
            child_expressions.append(child_expr)
        
        if atom_type in ["ConceptNode", "PredicateNode"] and not child_expressions:
            return name
        
        # Format as S-expression
        all_parts = [atom_type]
        if metadata.get("original_name"):
            all_parts.append(metadata["original_name"])
        all_parts.extend(child_expressions)
        
        return f"({' '.join(all_parts)})"
    
    def round_trip_test(self, scheme_text: str) -> Dict[str, Any]:
        """
        Perform round-trip test: Scheme -> AtomSpace -> Scheme
        
        Args:
            scheme_text: Original Scheme expression
            
        Returns:
            Test results with fidelity metrics
        """
        try:
            # Forward translation
            hypergraph = self.scheme_to_atomspace(scheme_text)
            
            # Backward translation
            reconstructed_scheme = self.atomspace_to_scheme(hypergraph)
            
            # Compare structure
            original_tokens = self.parser.lexer.tokenize(scheme_text)
            reconstructed_tokens = self.parser.lexer.tokenize(reconstructed_scheme)
            
            # Calculate metrics
            token_similarity = self._calculate_token_similarity(original_tokens, reconstructed_tokens)
            structure_preservation = self._analyze_structure_preservation(scheme_text, reconstructed_scheme)
            
            return {
                "original_scheme": scheme_text,
                "hypergraph_nodes": len(hypergraph.get("nodes", [])),
                "hypergraph_links": len(hypergraph.get("links", [])),
                "reconstructed_scheme": reconstructed_scheme,
                "success": True,
                "metrics": {
                    "token_similarity": token_similarity,
                    "structure_preservation": structure_preservation
                }
            }
            
        except Exception as e:
            return {
                "original_scheme": scheme_text,
                "success": False,
                "error": str(e),
                "metrics": {
                    "token_similarity": 0.0,
                    "structure_preservation": 0.0
                }
            }
    
    def _calculate_token_similarity(self, tokens1: List[SchemeToken], 
                                  tokens2: List[SchemeToken]) -> float:
        """Calculate similarity between token sequences"""
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        
        # Compare token values
        values1 = [token.value for token in tokens1]
        values2 = [token.value for token in tokens2]
        
        common_tokens = len(set(values1) & set(values2))
        total_unique = len(set(values1) | set(values2))
        
        return common_tokens / total_unique if total_unique > 0 else 0.0
    
    def _analyze_structure_preservation(self, original: str, reconstructed: str) -> float:
        """Analyze how well structure is preserved"""
        try:
            orig_expr = self.parser.parse(original)
            recon_expr = self.parser.parse(reconstructed)
            
            return self._compare_expression_structure(orig_expr, recon_expr)
        except:
            return 0.0
    
    def _compare_expression_structure(self, expr1: SchemeExpression, 
                                    expr2: SchemeExpression) -> float:
        """Compare structure of two expressions"""
        score = 0.0
        
        # Compare atom types
        if expr1.atom_type == expr2.atom_type:
            score += 0.3
        
        # Compare names
        if expr1.name == expr2.name:
            score += 0.3
        
        # Compare children count
        if len(expr1.children) == len(expr2.children):
            score += 0.2
            
            # Recursively compare children
            if expr1.children:
                child_scores = []
                for child1, child2 in zip(expr1.children, expr2.children):
                    child_scores.append(self._compare_expression_structure(child1, child2))
                score += 0.2 * (sum(child_scores) / len(child_scores))
        
        return min(score, 1.0)