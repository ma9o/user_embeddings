# Training User-LLM (Efficient LLM Contextualization with User Embeddings) on Vana's DataDAOs Conversational Dataset

This project aims to leverage Vana’s DataDAOs private conversational datasets (Reddit, Telegram, ChatGPT) to outperform public alternatives (e.g., Reddit/Twitter PushShift). We produce two artifacts: a universal user-embedding model (for recall tasks) and an LLM personalization system (for precision tasks), showcasing the dataset’s value at different personalization stages.

## Project Goals
- **Unified Model Across Users**:
    - Compare datasets at aggregate level
    - Bootstrap personalization even with sparse user data
    - Cheap updates by encoding new data points
- **Efficient Long-Term History Compression**: Current personalization approaches rely on limited RAG methods to avoid attention bottlenecks, neglecting valuable long-term and implicit user signals.
- **Bias-Free Reasoning**: Prevent user personality biases from affecting the LLM's reasoning process (differently from LoRA-based methods).
- **Staged Training Approach**:
    - **Parameter Interpretability**: Simplify manual reasoning about parameters without complex Bayesian optimization
    - **Modular Artifacts**: Support producing both user embeddings and flexible personalization systems, not a single monolithic model
    - **Optimized Data Funnel**: Tiered data requirements make the project more manageable and cost-effective: millions of samples for early stages, scaling down to tens of thousands for final expensive reward evaluation stages. 

### Core Incremental Contributions
Building on User-LLM, our incremental contributions include:

- **Conversation Chunk Encoder**: Implement an encoder trained via distillation on subject-focused conversation summaries, using chat threads instead of traditional user-item interactions. 
- **Flexible Task Goaling**: Beyond next-item prediction, we focus on generating diverse user insights validated through agentic-RAG and context-stuffing.
- **Production-Grade Operationalization**:
    - Evaluate using state-of-the-art base encoder and decoder
    - Evaluate at exponentially increasing sequence embedding dimensions

## Methodology

### Staged Training Pipeline
The training pipeline comprises three independently trained components:

#### 1. Item Encoder
![Image 1](docs/images/1.png)

Fine-tune the text encoder via distillation from Qwen3-32B, capturing SUBJECT-centric contributions from conversation chunks. The summarization prompts are optimized via automated prompt refinement like DSPy.

#### 2. Sequence Encoder
![Image 2](docs/images/2.png)

The autoregressive encoder compresses noisy user histories, surfacing predictive signals like Knowledge, Opinions, Attributes, Intents (KOAIs).

**Note**: Non-predictive KOAIs explicitly stated by SUBJECTs may be ignored intentionally (which might actually be desirable: POSIWID). For potentially predictive KOAIs at the end of the sequence we should anyway rely on RAG at test time, as per original findings.

#### 3. Perceiver & Projector
![Image 3](docs/images/3.png)

Here we train the perceiver for cross-attention, optimizing for diverse user prediction tasks evaluated by a high-quality, context-stuffed judge model.

## Project Structure

```
.
├── data/ # Not tracked, contains artifacts
├── src/
│   └── user_embeddings/ # Code common to scripts and notebooks
├── notebooks/
└── scripts/
    └── data_preparation/
    └── evaluation/
```
## Limitations & Future Work

- Implement DSPy for prompt refinement (and program synthesis when available)
- Tune chunk embedder to KOAIs instead of summaries for interpretable semantic separation of user data (and potentially better prediction performance)
- Experiment with alternative goals for the autoencoder

## Getting Started
TODO

## Usage
TODO