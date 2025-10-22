# Mermaid Diagram Files - RAG Types

This directory contains both the Mermaid source files (.mmd) and the generated PNG images (.png) for all 25 RAG types.

## File List and Descriptions

| File Number | Name | Description |
|-------------|------|-------------|
| 001 | `001_standard_rag` | Standard RAG - Basic retrieval-augmented generation workflow |
| 002 | `002_corrective_rag` | Corrective RAG - Error correction and feedback loop system |
| 003 | `003_speculative_rag` | Speculative RAG - Dual-model approach with specialist drafter and generalist verifier |
| 004 | `004_fusion_rag` | Fusion RAG - Multiple retrievers with reciprocal rank fusion |
| 005 | `005_agentic_rag` | Agentic RAG - Agent-based system with task planner |
| 006 | `006_self_rag` | Self RAG - Self-referential system with evidence checking |
| 007 | `007_adaptive_rag` | Adaptive RAG - Dynamic retrieval decision making |
| 008 | `008_refeed_retrieval_feedback` | REFEED - Retrieval feedback without fine-tuning |
| 009 | `009_realm` | REALM - Neural knowledge retriever with pre-training and fine-tuning |
| 010 | `010_raptor_tree_organized_retrieval` | RAPTOR - Tree-organized hierarchical retrieval |
| 011 | `011_reveal_visual_language_model` | REVEAL - Visual-language model with knowledge fusion |
| 012 | `012_react` | REACT - Reasoning and action with observation loops |
| 013 | `013_replug_retrieval_plugin` | REPLUG - Retrieval plugin for LLMs |
| 014 | `014_memo_rag` | MEMO RAG - Memory model with clue-based retrieval |
| 015 | `015_attention_based_rag_atlas` | ATLAS - Attention-based RAG with dual-encoder |
| 016 | `016_retro` | RETRO - Chunked cross-attention retrieval |
| 017 | `017_auto_rag` | AUTO RAG - Automated optimization pipeline |
| 018 | `018_corag_cost_constrained_rag` | CORAG - Cost-constrained with MCTS policy |
| 019 | `019_eaco_rag` | EACO-RAG - Edge computing optimized RAG |
| 020 | `020_rule_rag` | RULE RAG - Rule-based guidance system |
| 021 | `021_conversational_rag_coral` | CORAL - Conversational RAG benchmarking |
| 022 | `022_iterative_rag` | Iterative RAG - Multiple retrieval steps with feedback |
| 023 | `023_context_driven_tree_structured_retrieval_contregen` | ConTReGen - Context-driven tree-structured approach |
| 024 | `024_causality_enhanced_reflective_retrieval_augmented_translation_crat` | CRAT - Causality-enhanced translation system |
| 025 | `025_graph_rag` | Graph RAG - Knowledge graph construction and processing |

## Usage

### Source Files (.mmd)
- These are the original Mermaid markup files
- Can be edited with any text editor
- Can be used to regenerate PNG files with: `mmdc -i filename.mmd -o filename.png`

### PNG Files (.png)
- Ready-to-use image files
- Can be embedded in presentations, documents, or web pages
- Generated from the .mmd source files using Mermaid CLI

## Regenerating Images

To regenerate all PNG files from the Mermaid source files:

```powershell
Get-ChildItem *.mmd | ForEach-Object { 
    Write-Host "Converting $($_.Name)..."
    mmdc -i $_.Name -o ($_.BaseName + ".png") 
}
```

## File Sizes
The PNG files range from ~10KB to ~31KB, optimized for both quality and file size.