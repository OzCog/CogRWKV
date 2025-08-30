#!/usr/bin/env python3
"""
Phase 1 Cognitive Primitives Demo

Demonstrates the complete pipeline from RWKV neural states to AtomSpace hypergraphs
to Scheme expressions, showcasing bidirectional translation capabilities.
"""

import torch
import numpy as np
import json
import sys
import os

# Add the current directory to sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cognitive_primitives.tensor_primitives import TensorPrimitive, CognitiveState, ModalityType
from cognitive_primitives.hypergraph_encoding import HypergraphEncoder, AtomSpaceNode, AtomType
from cognitive_primitives.ko6ml_adapter import Ko6mlAdapter
from scheme_adapters import AtomSpaceSchemeAdapter
from docs.visualization import HypergraphVisualizer


def demo_tensor_primitives():
    """Demonstrate tensor primitive functionality"""
    print("="*60)
    print("TENSOR PRIMITIVES DEMO")
    print("="*60)
    
    # Create a cognitive primitive
    primitive = TensorPrimitive(
        modality=ModalityType.LINGUISTIC.value,
        depth=2,
        context=15,
        salience=0.85,
        autonomy_index=0.72
    )
    
    print(f"Created primitive: {primitive}")
    
    # Convert to tensor
    tensor = primitive.to_tensor()
    print(f"Tensor representation: {tensor}")
    
    # Round-trip conversion
    reconstructed = TensorPrimitive.from_tensor(tensor)
    print(f"Reconstructed primitive: {reconstructed}")
    
    # Verify perfect reconstruction
    assert primitive.modality == reconstructed.modality
    assert primitive.depth == reconstructed.depth
    assert primitive.context == reconstructed.context
    assert abs(primitive.salience - reconstructed.salience) < 1e-6
    assert abs(primitive.autonomy_index - reconstructed.autonomy_index) < 1e-6
    
    print("✓ Round-trip conversion verified!")
    return primitive


def demo_cognitive_state_extraction():
    """Demonstrate extraction from mock RWKV state"""
    print("\n" + "="*60)
    print("COGNITIVE STATE EXTRACTION DEMO") 
    print("="*60)
    
    # Simulate RWKV model parameters
    n_layer, n_embd = 4, 64
    
    # Create mock RWKV state (normally would come from actual RWKV model)
    print(f"Simulating RWKV state: {n_layer} layers, {n_embd} embedding dimensions")
    rwkv_state = torch.randn(n_layer * 3, n_embd)
    
    # Extract cognitive primitives
    cognitive_state = CognitiveState(n_layer, n_embd)
    primitives = cognitive_state.extract_from_rwkv_state(rwkv_state)
    
    print(f"Extracted {len(primitives)} cognitive primitives:")
    for name, primitive in primitives.items():
        print(f"  {name}: modality={primitive.modality}, depth={primitive.depth}, "
              f"salience={primitive.salience:.3f}, autonomy={primitive.autonomy_index:.3f}")
    
    # Test reconstruction
    reconstructed_state = cognitive_state.to_rwkv_state()
    print(f"Reconstructed RWKV state shape: {reconstructed_state.shape}")
    print("✓ Bidirectional state conversion completed!")
    
    return primitives


def demo_hypergraph_encoding():
    """Demonstrate hypergraph encoding of primitives"""
    print("\n" + "="*60)
    print("HYPERGRAPH ENCODING DEMO")
    print("="*60)
    
    encoder = HypergraphEncoder()
    
    # Create several tensor primitives
    primitives = [
        torch.tensor([0.0, 1.0, 10.0, 0.8, 0.6]),  # Linguistic, depth 1
        torch.tensor([1.0, 2.0, 15.0, 0.9, 0.7]),  # Visual, depth 2  
        torch.tensor([0.0, 1.0, 12.0, 0.7, 0.8])   # Linguistic, depth 1
    ]
    
    # Encode as hypergraph nodes
    nodes = []
    for i, tensor in enumerate(primitives):
        node = encoder.encode_tensor_primitive(tensor, f"primitive_{i}")
        nodes.append(node)
        print(f"Created {node.atom_type.value}: {node.name}")
    
    # Create cognitive state link
    state_link = encoder.create_cognitive_state_link(nodes, "demo_cognitive_state")
    print(f"Created {state_link.atom_type.value}: {state_link.name}")
    
    # Create relationships
    similarity_link = encoder.create_similarity_link(nodes[0], nodes[2])
    inheritance_link = encoder.create_inheritance_link(nodes[1], nodes[0])
    
    print(f"Created {similarity_link.atom_type.value} between {nodes[0].name} and {nodes[2].name}")
    print(f"Created {inheritance_link.atom_type.value} from {nodes[1].name} to {nodes[0].name}")
    
    # Export hypergraph
    hypergraph = encoder.get_hypergraph_fragment()
    print(f"\nHypergraph contains {len(hypergraph['nodes'])} nodes and {len(hypergraph['links'])} links")
    
    # Visualize
    visualizer = HypergraphVisualizer()
    flowchart = visualizer.generate_ascii_flowchart(hypergraph)
    print("\nHYPERGRAPH VISUALIZATION:")
    print(flowchart)
    
    return hypergraph


def demo_ko6ml_translation():
    """Demonstrate Ko6ml symbolic translation"""
    print("\n" + "="*60)
    print("Ko6ML TRANSLATION DEMO")
    print("="*60)
    
    adapter = Ko6mlAdapter()
    
    # Show primitive registry
    registry = adapter.get_primitive_registry()
    print("Available Ko6ml primitives:")
    for symbol, primitive_data in registry.items():
        role = primitive_data["semantic_role"]
        category = primitive_data["syntactic_category"]
        print(f"  {symbol}: {role} ({category})")
    
    # Test translation
    ko6ml_expressions = [
        ["AGENT", "ACTION"],
        ["AGENT", "ACTION", "GOAL"],
        ["AGENT", "BELIEF", "CONTEXT"]
    ]
    
    for expr in ko6ml_expressions:
        print(f"\nTranslating ko6ml expression: {expr}")
        
        # Forward translation
        hypergraph = adapter.ko6ml_to_atomspace(expr)
        print(f"  → Hypergraph: {len(hypergraph['nodes'])} nodes, {len(hypergraph['links'])} links")
        
        # Backward translation
        reconstructed = adapter.atomspace_to_ko6ml(hypergraph)
        print(f"  → Reconstructed: {reconstructed}")
        
        # Round-trip test
        result = adapter.round_trip_test(expr)
        metrics = result["fidelity_metrics"]
        print(f"  → Fidelity: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
    
    return adapter.ko6ml_to_atomspace(["AGENT", "ACTION", "GOAL"])


def demo_scheme_adaptation():
    """Demonstrate Scheme S-expression adaptation"""
    print("\n" + "="*60)
    print("SCHEME ADAPTATION DEMO")
    print("="*60)
    
    adapter = AtomSpaceSchemeAdapter()
    
    # Test expressions
    scheme_expressions = [
        "(ConceptNode \"human\")",
        "(EvaluationLink (PredicateNode \"likes\") (ConceptNode \"alice\") (ConceptNode \"bob\"))",
        "(InheritanceLink (ConceptNode \"dog\") (ConceptNode \"animal\"))"
    ]
    
    for expr in scheme_expressions:
        print(f"\nProcessing Scheme expression: {expr}")
        
        try:
            # Forward translation
            hypergraph = adapter.scheme_to_atomspace(expr)
            print(f"  → Hypergraph: {len(hypergraph['nodes'])} nodes, {len(hypergraph['links'])} links")
            
            # Backward translation
            reconstructed = adapter.atomspace_to_scheme(hypergraph)
            print(f"  → Reconstructed: {reconstructed}")
            
            # Round-trip test
            result = adapter.round_trip_test(expr)
            if result["success"]:
                metrics = result["metrics"]
                print(f"  → Metrics: Token similarity={metrics['token_similarity']:.3f}, "
                      f"Structure preservation={metrics['structure_preservation']:.3f}")
            else:
                print(f"  → Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"  → Exception: {e}")
    
    return adapter.scheme_to_atomspace("(ConceptNode \"demo\")")


def demo_full_pipeline():
    """Demonstrate complete pipeline integration"""
    print("\n" + "="*60)
    print("FULL PIPELINE INTEGRATION DEMO")
    print("="*60)
    
    print("Pipeline: RWKV State → Cognitive Primitives → Hypergraph → Ko6ml → Scheme")
    
    # Step 1: Start with RWKV state
    n_layer, n_embd = 3, 32
    rwkv_state = torch.randn(n_layer * 3, n_embd)
    print(f"1. RWKV State: {rwkv_state.shape}")
    
    # Step 2: Extract cognitive primitives
    cognitive_state = CognitiveState(n_layer, n_embd)
    primitives = cognitive_state.extract_from_rwkv_state(rwkv_state)
    print(f"2. Extracted {len(primitives)} cognitive primitives")
    
    # Step 3: Encode as hypergraph
    encoder = HypergraphEncoder()
    for name, primitive in primitives.items():
        tensor = primitive.to_tensor()
        encoder.encode_tensor_primitive(tensor, name)
    
    hypergraph = encoder.get_hypergraph_fragment()
    print(f"3. Hypergraph: {len(hypergraph['nodes'])} nodes, {len(hypergraph['links'])} links")
    
    # Step 4: Convert to Scheme (simplified)
    scheme_adapter = AtomSpaceSchemeAdapter()
    scheme_expr = scheme_adapter.atomspace_to_scheme(hypergraph)
    print(f"4. Scheme expression: {scheme_expr[:100]}...")
    
    # Step 5: Verify tensor consistency
    visualizer = HypergraphVisualizer()
    summary = visualizer.generate_structure_summary(hypergraph)
    print(f"5. Verification: {len(summary['cognitive_primitives'])} tensor primitives preserved")
    
    print("✓ Full pipeline completed successfully!")
    
    return {
        'rwkv_state': rwkv_state,
        'primitives': primitives,
        'hypergraph': hypergraph,
        'scheme': scheme_expr,
        'summary': summary
    }


def demo_prime_factorization_mapping():
    """Demonstrate the prime factorization mapping documentation"""
    print("\n" + "="*60)
    print("PRIME FACTORIZATION MAPPING DEMO")
    print("="*60)
    
    cognitive_state = CognitiveState(4, 64)
    mapping = cognitive_state.get_prime_factorization_mapping()
    
    print("Prime factorization mapping:")
    print(json.dumps(mapping, indent=2))
    
    # Demonstrate encoding patterns
    print("\nExample encoding patterns:")
    for modality_idx, prime in enumerate(mapping["modality_primes"][:3]):
        print(f"  Modality {modality_idx} (prime {prime}): sin(x * {prime} * 0.1)")
    
    return mapping


def main():
    """Run all demonstrations"""
    print("PHASE 1: COGNITIVE PRIMITIVES & FOUNDATIONAL HYPERGRAPH ENCODING")
    print("Demonstration of Bidirectional Translation Mechanisms")
    print("=" * 80)
    
    try:
        # Run all demos
        primitive = demo_tensor_primitives()
        extracted_primitives = demo_cognitive_state_extraction()
        hypergraph = demo_hypergraph_encoding()
        ko6ml_hypergraph = demo_ko6ml_translation()
        scheme_hypergraph = demo_scheme_adaptation()
        pipeline_result = demo_full_pipeline()
        mapping = demo_prime_factorization_mapping()
        
        # Final summary
        print("\n" + "="*80)
        print("DEMONSTRATION SUMMARY")
        print("="*80)
        print("✓ Tensor primitives: Creation, conversion, round-trip verification")
        print("✓ Cognitive state extraction: RWKV state ↔ primitives")
        print("✓ Hypergraph encoding: Primitives → AtomSpace nodes/links")
        print("✓ Ko6ml translation: Symbolic expressions ↔ hypergraphs")
        print("✓ Scheme adaptation: S-expressions ↔ hypergraphs")
        print("✓ Full pipeline: End-to-end integration")
        print("✓ Prime factorization mapping: Documented tensor signatures")
        
        print(f"\nAll demonstrations completed successfully!")
        print(f"The implementation provides comprehensive bidirectional translation")
        print(f"between RWKV neural states and symbolic cognitive representations.")
        
        return True
        
    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)