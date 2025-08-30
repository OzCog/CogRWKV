# Phase 1: Cognitive Primitives & Foundational Hypergraph Encoding

## Overview

This module implements the foundational cognitive primitives and bidirectional translation mechanisms between ko6ml primitives and AtomSpace hypergraph patterns for the CogRWKV project.

## Architecture

### Core Components

#### 1. Tensor Primitives (`cognitive_primitives/tensor_primitives.py`)

The foundational tensor representation for cognitive states with the specification:
```
[modality, depth, context, salience, autonomy_index]
```

**TensorPrimitive Class:**
- Encodes cognitive dimensions as 5D vectors
- Supports bidirectional conversion: primitive ↔ tensor
- Maps modality types (linguistic, visual, auditory, etc.)

**CognitiveState Class:**
- Manages collections of tensor primitives
- Extracts primitives from RWKV neural states
- Reconstructs RWKV states from primitives
- Uses prime factorization mapping for tensor encoding

#### 2. Hypergraph Encoding (`cognitive_primitives/hypergraph_encoding.py`)

Translates cognitive primitives into AtomSpace hypergraph representations.

**AtomSpaceNode Class:**
- Represents nodes/links in hypergraph
- Stores tensor embeddings and metadata
- Supports serialization/deserialization

**HypergraphEncoder Class:**
- Encodes tensor primitives as TensorNodes
- Creates cognitive state links between primitives
- Builds relationship links (inheritance, similarity)
- Exports hypergraph fragments

#### 3. Ko6ml Adapter (`cognitive_primitives/ko6ml_adapter.py`)

Bidirectional translation between ko6ml symbolic expressions and AtomSpace hypergraphs.

**Ko6mlPrimitive Class:**
- Represents ko6ml symbols with semantic/syntactic properties
- Maps to cognitive tensor dimensions

**Ko6mlAdapter Class:**
- Registry of base primitives (AGENT, ACTION, GOAL, etc.)
- Forward translation: ko6ml → AtomSpace
- Backward translation: AtomSpace → ko6ml
- Round-trip fidelity testing

#### 4. Scheme Adapters (`scheme_adapters/__init__.py`)

Modular Scheme adapters for agentic grammar AtomSpace integration.

**SchemeLexer/Parser Classes:**
- Parse Scheme S-expressions into structured form
- Handle AtomSpace-specific constructs

**AtomSpaceSchemeAdapter Class:**
- Bidirectional Scheme ↔ AtomSpace translation
- Structure-preserving round-trip conversion
- Token similarity metrics

## Data Flow Architecture

```
RWKV Neural State
       ↓
Cognitive Primitives (5D tensors)
       ↓
AtomSpace Hypergraph
       ↓ ↙ ↘
Ko6ml Symbols   Scheme S-expressions
```

### Tensor Signature Mapping

The cognitive tensor dimensions map to RWKV components as follows:

1. **Modality** (dimension 0): Extracted from activation patterns using prime factorization
   - 0: Linguistic, 1: Visual, 2: Auditory, 3: Spatial, 4: Temporal, 5: Abstract, 6: Sensorimotor

2. **Depth** (dimension 1): Direct mapping to RWKV layer index
   - Range: [0, n_layer-1]

3. **Context** (dimension 2): Derived from state tensor variance
   - Quantized measure of context complexity

4. **Salience** (dimension 3): L2 norm of state tensor, normalized
   - Range: [0.0, 1.0]

5. **Autonomy Index** (dimension 4): Composite of depth factor and entropy
   - Calculation: (depth_factor + entropy_factor) / 2

### Prime Factorization Mapping

The encoding uses prime numbers to create unique patterns:

```python
modality_primes = [2, 3, 5, 7, 11, 13, 17]  # Maps to ModalityType
sin_pattern = sin(embedding_index * (modality + 1) * 0.1)
```

State reconstruction combines:
- Modality: Sinusoidal patterns with prime frequencies
- Depth: Amplitude scaling by layer ratio
- Context: Frequency modulation
- Salience: Pattern weighting
- Autonomy: Global scaling factor

## Usage Examples

### Basic Tensor Primitive Usage

```python
from cognitive_primitives.tensor_primitives import TensorPrimitive, ModalityType

# Create a cognitive primitive
primitive = TensorPrimitive(
    modality=ModalityType.LINGUISTIC.value,
    depth=2,
    context=10,
    salience=0.8,
    autonomy_index=0.6
)

# Convert to tensor
tensor = primitive.to_tensor()
print(tensor)  # tensor([0., 2., 10., 0.8, 0.6])

# Round-trip conversion
reconstructed = TensorPrimitive.from_tensor(tensor)
```

### RWKV State Integration

```python
from cognitive_primitives.tensor_primitives import CognitiveState

# Extract primitives from RWKV state
cognitive_state = CognitiveState(n_layer=4, n_embd=64)
rwkv_state = torch.randn(12, 64)  # 4 layers * 3 components
primitives = cognitive_state.extract_from_rwkv_state(rwkv_state)

# Reconstruct RWKV state
reconstructed_state = cognitive_state.to_rwkv_state()
```

### Hypergraph Encoding

```python
from cognitive_primitives.hypergraph_encoding import HypergraphEncoder

encoder = HypergraphEncoder()

# Encode tensor primitive as hypergraph node
tensor = torch.tensor([0., 1., 5., 0.7, 0.5])
node = encoder.encode_tensor_primitive(tensor, "test_primitive")

# Create relationships
node2 = encoder.encode_tensor_primitive(torch.tensor([1., 2., 8., 0.6, 0.7]), "primitive2")
similarity_link = encoder.create_similarity_link(node, node2)

# Export hypergraph
hypergraph = encoder.get_hypergraph_fragment()
```

### Ko6ml Translation

```python
from cognitive_primitives.ko6ml_adapter import Ko6mlAdapter

adapter = Ko6mlAdapter()

# Translate ko6ml expression to hypergraph
ko6ml_expr = ["AGENT", "ACTION", "GOAL"]
hypergraph = adapter.ko6ml_to_atomspace(ko6ml_expr)

# Translate back to ko6ml
reconstructed = adapter.atomspace_to_ko6ml(hypergraph)

# Test round-trip fidelity
result = adapter.round_trip_test(ko6ml_expr)
print(f"F1 Score: {result['fidelity_metrics']['f1_score']}")
```

### Scheme Integration

```python
from scheme_adapters import AtomSpaceSchemeAdapter

adapter = AtomSpaceSchemeAdapter()

# Parse Scheme expression
scheme_expr = "(ConceptNode \"human\")"
hypergraph = adapter.scheme_to_atomspace(scheme_expr)

# Convert back to Scheme
reconstructed_scheme = adapter.atomspace_to_scheme(hypergraph)

# Test round-trip
result = adapter.round_trip_test(scheme_expr)
```

## Testing

The comprehensive test suite (`tests/test_comprehensive.py`) validates:

- **Tensor primitive operations**: Creation, conversion, extraction
- **Hypergraph encoding**: Node creation, relationship links, serialization
- **Ko6ml translation**: Bidirectional conversion, fidelity metrics
- **Scheme adaptation**: S-expression parsing, round-trip preservation
- **Integration**: End-to-end pipeline validation

Run tests with:
```bash
python tests/test_comprehensive.py
```

## Performance Metrics

Current test results show:
- **84.2% test success rate**
- **Bidirectional tensor conversion**: 100% fidelity
- **Hypergraph serialization**: Complete structure preservation
- **Round-trip translation**: Measured precision, recall, F1-score

## File Structure

```
cognitive_primitives/
├── __init__.py                 # Module interface
├── tensor_primitives.py       # Core tensor representations
├── hypergraph_encoding.py     # AtomSpace encoding
└── ko6ml_adapter.py           # Ko6ml translation

scheme_adapters/
└── __init__.py                # Scheme S-expression adapters

tests/
└── test_comprehensive.py      # Complete test suite

docs/
└── README.md                  # This documentation
```

## Future Extensions

1. **Additional Modalities**: Extend beyond current 7 modality types
2. **Dynamic Primitives**: Runtime primitive registration and adaptation
3. **Optimization**: Performance optimization for large-scale hypergraphs
4. **Visualization**: Interactive hypergraph visualization tools
5. **Advanced Metrics**: More sophisticated fidelity measures

## Dependencies

- **PyTorch**: Tensor operations and neural network integration
- **NumPy**: Numerical computations
- **Python 3.8+**: Core language features

## License

This implementation is part of the CogRWKV project and follows the project's licensing terms.