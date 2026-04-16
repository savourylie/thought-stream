# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ThoughtStream is a novel text compression system that models human thoughts as continuous trajectories through semantic space. The core concept treats words as discrete samples of continuous thought trajectories, enabling efficient text compression using signal processing techniques.

### Key Components (Planned Architecture)

- **Trajectory Network**: Neural ODEs modeling smooth semantic paths through high-dimensional space
- **Vocabulary Quantizer**: Maps continuous trajectory points to discrete words  
- **Density Estimator**: Learns optimal sampling density across semantic regions
- **Adaptive Sampling**: Samples more frequently in sparse/complex regions, less in common concept areas

### Current State

This is an early-stage research project with minimal implementation. The codebase currently contains:
- Basic Python project structure with pyproject.toml
- Placeholder main.py with "Hello World" functionality
- Comprehensive README.md outlining the theoretical approach

## Development Commands

### Basic Operations
```bash
# Run Phase 1 demonstrations (spline trajectories)
uv run python main.py

# Run Neural ODE complexity experiment  
uv run python complexity_experiment.py

# Run learnable splines with semantic awareness (Phase 2)
uv run python learnable_splines.py

# Create trajectory visualizations
uv run python visualize_trajectories.py

# Install dependencies (when added to pyproject.toml)
uv sync

# Add new dependencies
uv add <package-name>
```

### Experimental Results Files
- `complexity_experiment_results.json` - Neural ODE complexity study data
- `trajectory_comparison_detailed.png` - Visual comparison of spline vs Neural ODE
- `learnable_splines_comparison.png` - Comprehensive learnable vs naive splines comparison
- `word_flow_comparison.png` - Word-level trajectory flow visualizations
- Phase 1 demonstrations: 40-100% reconstruction accuracy with large vocabularies
- Phase 2 learnable splines: Consistent semantic advantages across all test categories

### Project Setup
This is a uv project using Python >=3.11. All Python commands should be prefixed with `uv run` to use the project's virtual environment. The project is configured as a standard Python package with pyproject.toml.

## Implementation Phases

Based on the README, development follows three phases:

1. **Phase 1 (MVP)**: ✅ COMPLETED
   - Basic trajectory fitting through word embeddings using cubic splines
   - Sentence reconstruction and compression demonstrations  
   - Interpolation between sentences
   - Large vocabulary reality check (revealed quantization limitations)

2. **Phase 2 (Prototype)**: 🔄 IN PROGRESS  
   - Neural ODE investigation: ✅ Completed systematic complexity experiment
   - **Key Finding**: Neural ODEs don't outperform splines for word-level reconstruction (50-67% vs 100% accuracy)
   - **Learnable Spline Coefficients**: ✅ Successfully implemented with semantic awareness
   - **Large Vocabulary Integration**: ✅ Tested with 5000-word realistic vocabularies
   - **Semantic Advantages Proven**: ✅ 50%+ better interpolation, 15%+ smoother trajectories
   - **🎯 NEXT**: Adaptive sampling - find optimal sampling points for best reconstruction (currently using uniform sampling)
   - Paragraph-level compression pending

3. **Phase 3 (Full System)**: 🔄 **NEXT PRIORITY**
   - Document-level compression using learned semantic trajectories
   - Multi-resolution adaptive sampling based on semantic density
   - Comprehensive benchmarking against traditional compression methods
   - Real-world performance evaluation on large text corpora

## Technical Approach

The system implements text compression by:
1. Inferring continuous semantic trajectories θ(t) from input text
2. Storing trajectory parameters instead of word sequences
3. Reconstructing text by resampling trajectories at desired resolution
4. Using adaptive sampling based on semantic density

Target metrics: 10:1 compression ratio, >0.8 fluency, >0.9 semantic similarity, 100x speed improvement over abstractive methods.

## Trajectory Modeling Approaches

### Current Status (Phase 1-2 Findings)

**Cubic Splines (Baseline)**:
- ✅ Perfect word-level reconstruction (100% accuracy)
- ✅ Smooth interpolation through embedding space
- ❌ No semantic awareness or learnable parameters
- ❌ Fixed mathematical form (cubic polynomials)

**Neural ODEs (Investigated)**:  
- ❌ Poor word-level reconstruction (50-67% accuracy)
- ✅ Very smooth trajectories (95% smoother than splines)  
- ❌ Complex training, takes shortcuts between endpoints
- **Conclusion**: Not suitable for precise word-level trajectory fitting

### Alternative Approaches to Explore

**1. Learnable Spline Coefficients** ✅ **IMPLEMENTED & PROVEN**
- ✅ Semantic-aware tension, curvature, and bias parameters  
- ✅ Perfect word-level reconstruction (0.000000 error like naive splines)
- ✅ 50-54% better interpolation quality between words
- ✅ 14-18% smoother semantic trajectories
- ✅ 2-4% better semantic coherence scores
- ✅ Maintains 50-100% accuracy with realistic 5000-word vocabularies
- ✅ Comprehensive evaluation on complex sentences (emotions, technical terms, narratives)
- **Key Innovation**: Learns semantic-aware spline parameters while guaranteeing word interpolation

**2. Gaussian Process Trajectories** 📊 **(POTENTIAL PHASE 3)**  
- Model trajectories as GP samples with semantic kernels
- Principled uncertainty quantification
- Smooth interpolation with learned semantic similarity
- More computationally expensive but theoretically sound
- Could build on learnable spline insights for semantic kernel design

**3. Variational Trajectory Autoencoders** 🧠 **(PHASE 3 CANDIDATE)**
- Encode sentences to low-dimensional trajectory representations
- Decode back to word embeddings at desired sampling rates
- Learned compression with generative capabilities
- Could leverage learnable spline principles for smoother latent trajectories

**4. Transformer-Based Trajectory Modeling** 🤖 **(ADVANCED PHASE 3)**
- Use attention mechanisms for word-to-word transitions
- Query-based interpolation at arbitrary time points  
- State-of-the-art sequence modeling capabilities
- Could integrate learned semantic awareness from spline approach

**5. Radial Basis Function Networks** 🎈 **(ALTERNATIVE TO CONSIDER)**
- Universal approximators with fewer parameters than MLPs
- Learnable centers, widths, and weights for interpolation
- Smooth interpolation with interpretable basis functions
- Potential middle ground between splines and neural approaches

## Current Implementation Status

### Phase 2 Achievement Summary
✅ **Learnable splines successfully prove semantic advantages over naive splines:**
- Perfect word reconstruction (matches naive splines: 0.000000 error)
- Superior interpolation between words (50-54% improvement)  
- Smoother semantic trajectories (14-18% improvement)
- Better semantic coherence (2-4% improvement)
- Realistic large vocabulary performance (50-100% accuracy with 5000 words)
- Comprehensive testing across semantic transitions, technical terms, emotions, abstract concepts, scientific progressions, and narrative flows

### Next Steps (Phase 3 Priorities)
🎯 **Document-Level Compression**: Apply learned semantic trajectory principles to paragraph and document-level text
🎯 **Adaptive Sampling**: Implement density-based sampling using semantic trajectory insights  
🎯 **Performance Benchmarking**: Compare against traditional compression methods
🎯 **Real-World Evaluation**: Test on large text corpora with practical applications