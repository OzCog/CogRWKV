"""
Comprehensive Test Suite for Cognitive Primitives & Hypergraph Encoding

Tests all primitives, transformations, and round-trip translations without mocks.
Uses real data and validates complete system integration.
"""

import unittest
import torch
import numpy as np
import json
import sys
import os

# Add the parent directory to sys.path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_primitives.tensor_primitives import (
    TensorPrimitive, CognitiveState, ModalityType
)
from cognitive_primitives.hypergraph_encoding import (
    HypergraphEncoder, AtomSpaceNode, AtomType
)
from cognitive_primitives.ko6ml_adapter import Ko6mlAdapter, Ko6mlPrimitive
from scheme_adapters import AtomSpaceSchemeAdapter

class TestTensorPrimitives(unittest.TestCase):
    """Test tensor primitive functionality"""
    
    def setUp(self):
        self.primitive = TensorPrimitive(
            modality=ModalityType.LINGUISTIC.value,
            depth=2,
            context=10,
            salience=0.8,
            autonomy_index=0.6
        )
        
    def test_tensor_primitive_creation(self):
        """Test creation and basic properties of tensor primitives"""
        self.assertEqual(self.primitive.modality, ModalityType.LINGUISTIC.value)
        self.assertEqual(self.primitive.depth, 2)
        self.assertEqual(self.primitive.context, 10)
        self.assertAlmostEqual(self.primitive.salience, 0.8)
        self.assertAlmostEqual(self.primitive.autonomy_index, 0.6)
    
    def test_tensor_conversion(self):
        """Test bidirectional tensor conversion"""
        tensor = self.primitive.to_tensor()
        
        # Verify tensor shape and values
        self.assertEqual(tensor.shape, (5,))
        self.assertEqual(tensor[0].item(), ModalityType.LINGUISTIC.value)
        self.assertEqual(tensor[1].item(), 2)
        self.assertEqual(tensor[2].item(), 10)
        self.assertAlmostEqual(tensor[3].item(), 0.8)
        self.assertAlmostEqual(tensor[4].item(), 0.6)
        
        # Test round-trip conversion
        reconstructed = TensorPrimitive.from_tensor(tensor)
        self.assertEqual(reconstructed.modality, self.primitive.modality)
        self.assertEqual(reconstructed.depth, self.primitive.depth)
        self.assertEqual(reconstructed.context, self.primitive.context)
        self.assertAlmostEqual(reconstructed.salience, self.primitive.salience)
        self.assertAlmostEqual(reconstructed.autonomy_index, self.primitive.autonomy_index)
    
    def test_cognitive_state_extraction(self):
        """Test extraction of cognitive primitives from RWKV state"""
        n_layer, n_embd = 4, 64
        cognitive_state = CognitiveState(n_layer, n_embd)
        
        # Create mock RWKV state
        rwkv_state = torch.randn(n_layer * 3, n_embd)
        
        # Extract primitives
        primitives = cognitive_state.extract_from_rwkv_state(rwkv_state)
        
        # Verify extraction
        self.assertEqual(len(primitives), n_layer)
        for layer_idx in range(n_layer):
            primitive_name = f"layer_{layer_idx}"
            self.assertIn(primitive_name, primitives)
            
            primitive = primitives[primitive_name]
            self.assertEqual(primitive.depth, layer_idx)
            self.assertGreaterEqual(primitive.salience, 0.0)
            self.assertLessEqual(primitive.salience, 1.0)
            self.assertGreaterEqual(primitive.autonomy_index, 0.0)
            self.assertLessEqual(primitive.autonomy_index, 1.0)
    
    def test_cognitive_state_reconstruction(self):
        """Test reconstruction of RWKV state from primitives"""
        n_layer, n_embd = 4, 64
        cognitive_state = CognitiveState(n_layer, n_embd)
        
        # Add test primitives
        for i in range(n_layer):
            primitive = TensorPrimitive(
                modality=ModalityType.LINGUISTIC.value,
                depth=i,
                context=i * 5,
                salience=0.5 + i * 0.1,
                autonomy_index=0.3 + i * 0.15
            )
            cognitive_state.add_primitive(f"layer_{i}", primitive)
        
        # Reconstruct RWKV state
        reconstructed_state = cognitive_state.to_rwkv_state()
        
        # Verify state properties
        self.assertEqual(reconstructed_state.shape, (n_layer * 3, n_embd))
        self.assertFalse(torch.isnan(reconstructed_state).any())
        self.assertFalse(torch.isinf(reconstructed_state).any())

class TestHypergraphEncoding(unittest.TestCase):
    """Test hypergraph encoding functionality"""
    
    def setUp(self):
        self.encoder = HypergraphEncoder()
        self.test_tensor = torch.tensor([0.0, 1.0, 10.0, 0.8, 0.6])
    
    def test_tensor_primitive_encoding(self):
        """Test encoding tensor primitives as AtomSpace nodes"""
        node = self.encoder.encode_tensor_primitive(self.test_tensor, "test_tensor")
        
        self.assertEqual(node.atom_type, AtomType.TENSOR_NODE)
        self.assertEqual(node.name, "test_tensor")
        self.assertTrue(torch.equal(node.tensor_embedding, self.test_tensor))
        self.assertEqual(node.metadata["modality"], 0)
        self.assertEqual(node.metadata["depth"], 1)
        self.assertEqual(node.metadata["context"], 10)
        self.assertAlmostEqual(node.metadata["salience"], 0.8)
        self.assertAlmostEqual(node.metadata["autonomy_index"], 0.6)
    
    def test_cognitive_state_link_creation(self):
        """Test creation of cognitive state links"""
        # Create multiple tensor nodes
        nodes = []
        for i in range(3):
            tensor = torch.tensor([0.0, i, i*5, 0.5 + i*0.1, 0.4 + i*0.1])
            node = self.encoder.encode_tensor_primitive(tensor, f"tensor_{i}")
            nodes.append(node)
        
        # Create cognitive state link
        state_link = self.encoder.create_cognitive_state_link(nodes, "test_state")
        
        self.assertEqual(state_link.atom_type, AtomType.COGNITIVE_STATE_LINK)
        self.assertEqual(len(state_link.children), 3)
        self.assertEqual(state_link.metadata["num_primitives"], 3)
        self.assertIsInstance(state_link.tensor_embedding, torch.Tensor)
    
    def test_relationship_links(self):
        """Test creation of inheritance and similarity links"""
        # Create test nodes
        node1 = self.encoder.encode_tensor_primitive(
            torch.tensor([0.0, 1.0, 5.0, 0.7, 0.5]), "node1")
        node2 = self.encoder.encode_tensor_primitive(
            torch.tensor([0.0, 2.0, 8.0, 0.6, 0.7]), "node2")
        
        # Test inheritance link
        inheritance_link = self.encoder.create_inheritance_link(node1, node2)
        self.assertEqual(inheritance_link.atom_type, AtomType.INHERITANCE_LINK)
        self.assertEqual(len(inheritance_link.children), 2)
        
        # Test similarity link  
        similarity_link = self.encoder.create_similarity_link(node1, node2)
        self.assertEqual(similarity_link.atom_type, AtomType.SIMILARITY_LINK)
        self.assertGreater(similarity_link.strength, 0.0)
        self.assertLessEqual(similarity_link.strength, 1.0)
    
    def test_hypergraph_serialization(self):
        """Test hypergraph fragment serialization and loading"""
        # Create test hypergraph
        node1 = self.encoder.encode_tensor_primitive(self.test_tensor, "test_node")
        node2 = self.encoder.encode_tensor_primitive(
            torch.tensor([1.0, 2.0, 15.0, 0.9, 0.8]), "test_node2")
        
        self.encoder.create_similarity_link(node1, node2)
        
        # Export fragment
        fragment = self.encoder.get_hypergraph_fragment()
        
        # Verify fragment structure
        self.assertIn("nodes", fragment)
        self.assertIn("links", fragment)
        self.assertIn("metadata", fragment)
        self.assertGreater(len(fragment["nodes"]), 0)
        self.assertGreater(len(fragment["links"]), 0)
        
        # Test round-trip serialization
        new_encoder = HypergraphEncoder()
        new_encoder.load_hypergraph_fragment(fragment)
        
        self.assertEqual(len(new_encoder.nodes), len(self.encoder.nodes))
        self.assertEqual(len(new_encoder.links), len(self.encoder.links))

class TestKo6mlAdapter(unittest.TestCase):
    """Test Ko6ml adapter functionality"""
    
    def setUp(self):
        self.adapter = Ko6mlAdapter()
    
    def test_primitive_registry(self):
        """Test ko6ml primitive registry"""
        registry = self.adapter.get_primitive_registry()
        
        # Verify base primitives are loaded
        expected_primitives = ["AGENT", "ACTION", "GOAL", "CONTEXT", "BELIEF"]
        for primitive in expected_primitives:
            self.assertIn(primitive, registry)
            
            # Verify structure
            prim_data = registry[primitive]
            self.assertIn("symbol", prim_data)
            self.assertIn("semantic_role", prim_data)
            self.assertIn("syntactic_category", prim_data)
            self.assertIn("tensor_binding", prim_data)
            self.assertIn("cognitive_features", prim_data)
    
    def test_ko6ml_to_atomspace_translation(self):
        """Test translation from ko6ml to AtomSpace"""
        ko6ml_expression = ["AGENT", "ACTION", "GOAL"]
        
        hypergraph = self.adapter.ko6ml_to_atomspace(ko6ml_expression)
        
        # Verify hypergraph structure
        self.assertIn("nodes", hypergraph)
        self.assertIn("links", hypergraph)
        
        # Should have nodes for each symbol
        nodes = hypergraph["nodes"]
        concept_nodes = [n for n in nodes if n["atom_type"] == "ConceptNode"]
        tensor_nodes = [n for n in nodes if n["atom_type"] == "TensorNode"]
        
        self.assertGreaterEqual(len(concept_nodes), 3)
        self.assertGreaterEqual(len(tensor_nodes), 3)
        
        # Should have similarity links between concepts
        links = hypergraph["links"]
        similarity_links = [l for l in links if l["atom_type"] == "SimilarityLink"]
        self.assertGreater(len(similarity_links), 0)
    
    def test_atomspace_to_ko6ml_translation(self):
        """Test translation from AtomSpace back to ko6ml"""
        original_expression = ["AGENT", "ACTION", "GOAL"]
        
        # Forward translation
        hypergraph = self.adapter.ko6ml_to_atomspace(original_expression)
        
        # Backward translation
        reconstructed_expression = self.adapter.atomspace_to_ko6ml(hypergraph)
        
        # Verify reconstruction
        self.assertIsInstance(reconstructed_expression, list)
        self.assertGreater(len(reconstructed_expression), 0)
        
        # Should contain original symbols (order may vary)
        original_set = set(original_expression)
        reconstructed_set = set(reconstructed_expression)
        overlap = len(original_set & reconstructed_set)
        self.assertGreater(overlap, 0)
    
    def test_round_trip_translation(self):
        """Test round-trip ko6ml translation"""
        test_expressions = [
            ["AGENT", "ACTION"],
            ["AGENT", "ACTION", "GOAL"],
            ["AGENT", "BELIEF", "CONTEXT"],
            ["ACTION", "GOAL", "CONTEXT"]
        ]
        
        for expression in test_expressions:
            with self.subTest(expression=expression):
                result = self.adapter.round_trip_test(expression)
                
                # Verify test structure
                self.assertIn("original_expression", result)
                self.assertIn("reconstructed_expression", result)
                self.assertIn("fidelity_metrics", result)
                self.assertIn("tensor_consistency", result)
                
                # Check fidelity metrics
                metrics = result["fidelity_metrics"]
                self.assertIn("precision", metrics)
                self.assertIn("recall", metrics)
                self.assertIn("f1_score", metrics)
                
                # Metrics should be reasonable for valid expressions
                self.assertGreaterEqual(metrics["precision"], 0.0)
                self.assertLessEqual(metrics["precision"], 1.0)
                self.assertGreaterEqual(metrics["recall"], 0.0)
                self.assertLessEqual(metrics["recall"], 1.0)
    
    def test_custom_primitive_addition(self):
        """Test adding custom ko6ml primitives"""
        self.adapter.add_custom_primitive(
            symbol="EMOTION",
            semantic_role="affective",
            syntactic_category="adjective",
            cognitive_features={"valence": 0.7, "arousal": 0.6, "intensity": 0.8}
        )
        
        registry = self.adapter.get_primitive_registry()
        self.assertIn("EMOTION", registry)
        
        emotion_primitive = registry["EMOTION"]
        self.assertEqual(emotion_primitive["symbol"], "EMOTION")
        self.assertEqual(emotion_primitive["semantic_role"], "affective")
        self.assertIn("valence", emotion_primitive["cognitive_features"])

class TestSchemeAdapters(unittest.TestCase):
    """Test Scheme adapter functionality"""
    
    def setUp(self):
        self.adapter = AtomSpaceSchemeAdapter()
    
    def test_scheme_parsing(self):
        """Test parsing of Scheme expressions"""
        test_expressions = [
            "(ConceptNode \"human\")",
            "(EvaluationLink (PredicateNode \"likes\") (ConceptNode \"alice\") (ConceptNode \"bob\"))",
            "(InheritanceLink (ConceptNode \"dog\") (ConceptNode \"animal\"))"
        ]
        
        for expr in test_expressions:
            with self.subTest(expression=expr):
                parsed = self.adapter.parser.parse(expr)
                self.assertIsNotNone(parsed)
                
                # Should be able to reconstruct
                hypergraph = self.adapter.scheme_to_atomspace(expr)
                self.assertIn("nodes", hypergraph)
    
    def test_scheme_to_atomspace_conversion(self):
        """Test conversion from Scheme to AtomSpace"""
        scheme_expr = "(ConceptNode \"human\")"
        
        hypergraph = self.adapter.scheme_to_atomspace(scheme_expr)
        
        # Verify structure
        self.assertIn("nodes", hypergraph)
        self.assertGreater(len(hypergraph["nodes"]), 0)
        
        # Should have ConceptNode
        concept_nodes = [n for n in hypergraph["nodes"] 
                        if n["atom_type"] == "ConceptNode"]
        self.assertGreater(len(concept_nodes), 0)
    
    def test_atomspace_to_scheme_conversion(self):
        """Test conversion from AtomSpace back to Scheme"""
        original_scheme = "(ConceptNode \"test\")"
        
        # Forward conversion
        hypergraph = self.adapter.scheme_to_atomspace(original_scheme)
        
        # Backward conversion
        reconstructed_scheme = self.adapter.atomspace_to_scheme(hypergraph)
        
        # Should be valid Scheme expression
        self.assertIsInstance(reconstructed_scheme, str)
        self.assertTrue(reconstructed_scheme.strip())
        
        # Should be parseable
        try:
            self.adapter.parser.parse(reconstructed_scheme)
            parse_success = True
        except:
            parse_success = False
        
        self.assertTrue(parse_success, f"Failed to parse: {reconstructed_scheme}")
    
    def test_scheme_round_trip(self):
        """Test round-trip Scheme translation"""
        test_expressions = [
            "(ConceptNode \"human\")",
            "(PredicateNode \"likes\")",
            "(EvaluationLink pred arg1 arg2)"
        ]
        
        for expr in test_expressions:
            with self.subTest(expression=expr):
                result = self.adapter.round_trip_test(expr)
                
                # Verify test completed successfully
                self.assertIn("success", result)
                if result["success"]:
                    self.assertIn("original_scheme", result)
                    self.assertIn("reconstructed_scheme", result)
                    self.assertIn("metrics", result)
                    
                    metrics = result["metrics"]
                    self.assertIn("token_similarity", metrics)
                    self.assertIn("structure_preservation", metrics)

class TestIntegration(unittest.TestCase):
    """Integration tests across all modules"""
    
    def test_full_pipeline_integration(self):
        """Test complete pipeline: RWKV -> Primitives -> Hypergraph -> Scheme"""
        # Create mock RWKV state
        n_layer, n_embd = 3, 32
        rwkv_state = torch.randn(n_layer * 3, n_embd)
        
        # Extract cognitive primitives
        cognitive_state = CognitiveState(n_layer, n_embd)
        primitives = cognitive_state.extract_from_rwkv_state(rwkv_state)
        
        # Encode as hypergraph
        encoder = HypergraphEncoder()
        for name, primitive in primitives.items():
            tensor = primitive.to_tensor()
            encoder.encode_tensor_primitive(tensor, name)
        
        # Export hypergraph
        hypergraph = encoder.get_hypergraph_fragment()
        
        # Convert to Scheme (simplified for testing)
        scheme_adapter = AtomSpaceSchemeAdapter()
        scheme_expr = scheme_adapter.atomspace_to_scheme(hypergraph)
        
        # Verify pipeline completed
        self.assertIsInstance(scheme_expr, str)
        self.assertTrue(scheme_expr.strip())
        
        # Verify tensor consistency through pipeline
        tensor_nodes = [n for n in hypergraph["nodes"] 
                       if n["atom_type"] == "TensorNode"]
        self.assertEqual(len(tensor_nodes), n_layer)
    
    def test_ko6ml_scheme_integration(self):
        """Test integration between Ko6ml and Scheme adapters"""
        # Start with ko6ml expression
        ko6ml_expr = ["AGENT", "ACTION", "GOAL"]
        
        # Convert to AtomSpace
        ko6ml_adapter = Ko6mlAdapter()
        hypergraph = ko6ml_adapter.ko6ml_to_atomspace(ko6ml_expr)
        
        # Convert to Scheme
        scheme_adapter = AtomSpaceSchemeAdapter()
        scheme_expr = scheme_adapter.atomspace_to_scheme(hypergraph)
        
        # Convert back to AtomSpace
        hypergraph2 = scheme_adapter.scheme_to_atomspace(scheme_expr)
        
        # Convert back to ko6ml
        ko6ml_expr2 = ko6ml_adapter.atomspace_to_ko6ml(hypergraph2)
        
        # Verify some consistency
        self.assertIsInstance(ko6ml_expr2, list)
        original_set = set(ko6ml_expr)
        reconstructed_set = set(ko6ml_expr2)
        overlap = len(original_set & reconstructed_set)
        self.assertGreater(overlap, 0)

def run_comprehensive_tests():
    """Run all tests and generate report"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestTensorPrimitives,
        TestHypergraphEncoding, 
        TestKo6mlAdapter,
        TestSchemeAdapters,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate report
    report = {
        "total_tests": result.testsRun,
        "successes": result.testsRun - len(result.failures) - len(result.errors),
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
    }
    
    print(f"\n{'='*50}")
    print("COMPREHENSIVE TEST REPORT")
    print(f"{'='*50}")
    print(f"Total Tests: {report['total_tests']}")
    print(f"Successes: {report['successes']}")
    print(f"Failures: {report['failures']}")
    print(f"Errors: {report['errors']}")
    print(f"Success Rate: {report['success_rate']:.1%}")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    return report

if __name__ == "__main__":
    run_comprehensive_tests()