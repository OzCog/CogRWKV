"""
Hypergraph Visualization Module

Generates ASCII flowcharts and structured representations of hypergraph fragments
for documentation and debugging purposes.
"""

import json
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict


class HypergraphVisualizer:
    """Creates ASCII visualizations of hypergraph fragments"""
    
    def __init__(self):
        self.symbols = {
            'node': '●',
            'concept': '○',
            'tensor': '▲',
            'link': '→',
            'inheritance': '↗',
            'similarity': '↔',
            'evaluation': '⟶',
            'cognitive_state': '◉'
        }
    
    def generate_ascii_flowchart(self, hypergraph: Dict[str, Any]) -> str:
        """Generate ASCII flowchart representation of hypergraph"""
        nodes = hypergraph.get("nodes", [])
        links = hypergraph.get("links", [])
        
        if not nodes:
            return "Empty hypergraph"
        
        # Build structure
        structure = self._analyze_structure(nodes, links)
        
        # Generate flowchart
        flowchart = []
        flowchart.append("HYPERGRAPH FRAGMENT FLOWCHART")
        flowchart.append("=" * 40)
        flowchart.append("")
        
        # Show nodes by type
        flowchart.extend(self._render_nodes_by_type(structure))
        flowchart.append("")
        
        # Show relationships
        flowchart.extend(self._render_relationships(structure))
        flowchart.append("")
        
        # Show statistics
        flowchart.extend(self._render_statistics(structure))
        
        return "\n".join(flowchart)
    
    def _analyze_structure(self, nodes: List[Dict], links: List[Dict]) -> Dict[str, Any]:
        """Analyze hypergraph structure"""
        structure = {
            'nodes_by_type': defaultdict(list),
            'relationships': [],
            'node_map': {},
            'statistics': defaultdict(int)
        }
        
        # Categorize nodes
        for node in nodes:
            atom_type = node.get("atom_type", "unknown")
            structure['nodes_by_type'][atom_type].append(node)
            structure['node_map'][node.get("name", "")] = node
            structure['statistics'][f"{atom_type}_count"] += 1
        
        # Process relationships
        for link in links:
            link_type = link.get("atom_type", "unknown")
            children = link.get("children", [])
            
            if len(children) >= 2:
                for i in range(len(children) - 1):
                    for j in range(i + 1, len(children)):
                        relationship = {
                            'type': link_type,
                            'source': children[i].get("name", ""),
                            'target': children[j].get("name", ""),
                            'strength': link.get("strength", 0.0)
                        }
                        structure['relationships'].append(relationship)
            
            structure['statistics'][f"{link_type}_count"] += 1
        
        return structure
    
    def _render_nodes_by_type(self, structure: Dict[str, Any]) -> List[str]:
        """Render nodes grouped by type"""
        lines = ["NODES BY TYPE:"]
        
        for atom_type, nodes in structure['nodes_by_type'].items():
            symbol = self._get_symbol_for_type(atom_type)
            lines.append(f"\n{atom_type} {symbol}")
            lines.append("-" * (len(atom_type) + 2))
            
            for node in nodes:
                name = node.get("name", "unnamed")
                strength = node.get("strength", 0.0)
                confidence = node.get("confidence", 0.0)
                
                # Show tensor info if available
                tensor_info = ""
                if "tensor_data" in node:
                    tensor_data = node["tensor_data"]
                    if len(tensor_data) == 5:
                        tensor_info = f" [mod:{tensor_data[0]:.0f} dep:{tensor_data[1]:.0f} ctx:{tensor_data[2]:.0f}]"
                
                lines.append(f"  {symbol} {name} (s:{strength:.2f} c:{confidence:.2f}){tensor_info}")
                
                # Show metadata if significant
                metadata = node.get("metadata", {})
                if "symbol" in metadata:
                    lines.append(f"    ↳ ko6ml: {metadata['symbol']}")
                if "semantic_role" in metadata:
                    lines.append(f"    ↳ role: {metadata['semantic_role']}")
        
        return lines
    
    def _render_relationships(self, structure: Dict[str, Any]) -> List[str]:
        """Render relationship connections"""
        lines = ["RELATIONSHIPS:"]
        
        if not structure['relationships']:
            lines.append("  (no relationships)")
            return lines
        
        # Group by relationship type
        by_type = defaultdict(list)
        for rel in structure['relationships']:
            by_type[rel['type']].append(rel)
        
        for rel_type, relationships in by_type.items():
            symbol = self._get_symbol_for_type(rel_type)
            lines.append(f"\n{rel_type} {symbol}")
            lines.append("-" * (len(rel_type) + 2))
            
            for rel in relationships:
                source = rel['source'][:15] + "..." if len(rel['source']) > 15 else rel['source']
                target = rel['target'][:15] + "..." if len(rel['target']) > 15 else rel['target']
                strength = rel['strength']
                
                lines.append(f"  {source} {symbol} {target} ({strength:.2f})")
        
        return lines
    
    def _render_statistics(self, structure: Dict[str, Any]) -> List[str]:
        """Render hypergraph statistics"""
        lines = ["STATISTICS:"]
        stats = structure['statistics']
        
        # Count totals
        total_nodes = sum(count for key, count in stats.items() if key.endswith('_count') and 'Link' not in key)
        total_links = sum(count for key, count in stats.items() if key.endswith('_count') and 'Link' in key)
        
        lines.append(f"  Total Nodes: {total_nodes}")
        lines.append(f"  Total Links: {total_links}")
        lines.append(f"  Total Relationships: {len(structure['relationships'])}")
        
        # Show breakdown
        lines.append("\n  Breakdown:")
        for stat_name, count in sorted(stats.items()):
            if count > 0:
                clean_name = stat_name.replace('_count', '').replace('Node', ' Node').replace('Link', ' Link')
                lines.append(f"    {clean_name}: {count}")
        
        return lines
    
    def _get_symbol_for_type(self, atom_type: str) -> str:
        """Get symbol for atom type"""
        type_mapping = {
            'ConceptNode': self.symbols['concept'],
            'TensorNode': self.symbols['tensor'], 
            'InheritanceLink': self.symbols['inheritance'],
            'SimilarityLink': self.symbols['similarity'],
            'EvaluationLink': self.symbols['evaluation'],
            'CognitiveStateLink': self.symbols['cognitive_state']
        }
        return type_mapping.get(atom_type, self.symbols['node'])
    
    def generate_structure_summary(self, hypergraph: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured summary for programmatic use"""
        nodes = hypergraph.get("nodes", [])
        links = hypergraph.get("links", [])
        
        summary = {
            'overview': {
                'total_nodes': len(nodes),
                'total_links': len(links),
                'node_types': {},
                'link_types': {}
            },
            'cognitive_primitives': [],
            'semantic_structure': {
                'modalities': set(),
                'depth_levels': set(),
                'ko6ml_symbols': set()
            },
            'connectivity': {
                'average_degree': 0,
                'max_strength': 0,
                'relationship_density': 0
            }
        }
        
        # Analyze nodes
        for node in nodes:
            atom_type = node.get("atom_type", "unknown")
            summary['overview']['node_types'][atom_type] = summary['overview']['node_types'].get(atom_type, 0) + 1
            
            # Extract cognitive primitives
            if atom_type == "TensorNode" and "tensor_data" in node:
                tensor_data = node["tensor_data"]
                if len(tensor_data) == 5:
                    primitive = {
                        'name': node.get("name", ""),
                        'modality': int(tensor_data[0]),
                        'depth': int(tensor_data[1]),
                        'context': int(tensor_data[2]),
                        'salience': float(tensor_data[3]),
                        'autonomy': float(tensor_data[4])
                    }
                    summary['cognitive_primitives'].append(primitive)
                    summary['semantic_structure']['modalities'].add(int(tensor_data[0]))
                    summary['semantic_structure']['depth_levels'].add(int(tensor_data[1]))
            
            # Extract ko6ml symbols
            metadata = node.get("metadata", {})
            if "symbol" in metadata:
                summary['semantic_structure']['ko6ml_symbols'].add(metadata["symbol"])
        
        # Analyze links
        strengths = []
        for link in links:
            atom_type = link.get("atom_type", "unknown")
            summary['overview']['link_types'][atom_type] = summary['overview']['link_types'].get(atom_type, 0) + 1
            
            strength = link.get("strength", 0)
            if strength > 0:
                strengths.append(strength)
        
        # Calculate connectivity metrics
        if strengths:
            summary['connectivity']['max_strength'] = max(strengths)
            summary['connectivity']['average_strength'] = sum(strengths) / len(strengths)
        
        if len(nodes) > 1:
            summary['connectivity']['relationship_density'] = len(links) / (len(nodes) * (len(nodes) - 1) / 2)
        
        # Convert sets to lists for JSON serialization
        summary['semantic_structure']['modalities'] = sorted(list(summary['semantic_structure']['modalities']))
        summary['semantic_structure']['depth_levels'] = sorted(list(summary['semantic_structure']['depth_levels']))
        summary['semantic_structure']['ko6ml_symbols'] = sorted(list(summary['semantic_structure']['ko6ml_symbols']))
        
        return summary
    
    def export_visualization(self, hypergraph: Dict[str, Any], filename: str):
        """Export visualization to file"""
        flowchart = self.generate_ascii_flowchart(hypergraph)
        summary = self.generate_structure_summary(hypergraph)
        
        with open(filename, 'w') as f:
            f.write(flowchart)
            f.write("\n\n")
            f.write("STRUCTURED SUMMARY:\n")
            f.write("=" * 20 + "\n")
            f.write(json.dumps(summary, indent=2))


def demonstrate_visualization():
    """Demonstrate visualization with sample hypergraph"""
    # Create sample hypergraph for demonstration
    sample_hypergraph = {
        "nodes": [
            {
                "atom_type": "ConceptNode",
                "name": "concept_agent",
                "strength": 0.8,
                "confidence": 0.9,
                "metadata": {
                    "symbol": "AGENT",
                    "semantic_role": "actor",
                    "syntactic_category": "noun_phrase"
                }
            },
            {
                "atom_type": "TensorNode", 
                "name": "tensor_agent",
                "tensor_data": [0.0, 1.0, 5.0, 0.8, 0.6],
                "strength": 0.8,
                "confidence": 0.6
            },
            {
                "atom_type": "ConceptNode",
                "name": "concept_action", 
                "strength": 0.7,
                "confidence": 0.8,
                "metadata": {
                    "symbol": "ACTION",
                    "semantic_role": "predicate",
                    "syntactic_category": "verb_phrase"
                }
            },
            {
                "atom_type": "TensorNode",
                "name": "tensor_action",
                "tensor_data": [0.0, 1.0, 6.0, 0.7, 0.5],
                "strength": 0.7,
                "confidence": 0.5
            }
        ],
        "links": [
            {
                "atom_type": "InheritanceLink",
                "name": "inheritance_tensor_agent_concept_agent",
                "strength": 0.8,
                "confidence": 0.75,
                "children": [
                    {"name": "tensor_agent"},
                    {"name": "concept_agent"}
                ]
            },
            {
                "atom_type": "SimilarityLink",
                "name": "similarity_concept_agent_concept_action",
                "strength": 0.6,
                "confidence": 0.65,
                "children": [
                    {"name": "concept_agent"},
                    {"name": "concept_action"}
                ]
            }
        ],
        "metadata": {
            "total_nodes": 4,
            "total_links": 2,
            "encoding_version": "1.0"
        }
    }
    
    visualizer = HypergraphVisualizer()
    flowchart = visualizer.generate_ascii_flowchart(sample_hypergraph)
    print(flowchart)
    
    print("\n" + "="*50)
    print("STRUCTURED SUMMARY:")
    print("="*50)
    summary = visualizer.generate_structure_summary(sample_hypergraph)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    demonstrate_visualization()