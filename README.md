# 🤖 Transformer from Scratch

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A ground-up implementation of the **Transformer architecture** from the seminal paper [*"Attention Is All You Need"* (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762), built using **PyTorch** and **NumPy** without relying on any high-level Transformer libraries.

---

## 📌 Description

This repository walks through the complete Transformer architecture one component at a time from raw attention math to a fully functional encoder-decoder model. Each module is implemented from first principles, with accompanying notebooks that visualize and validate every step.

The project is structured as a **progressive learning series**: each file builds on the previous, making it ideal both as a portfolio project and as a personal reference for understanding how modern LLMs work internally.

---

## 💡 Motivation / Problem Statement

The Transformer architecture is the backbone of virtually every modern NLP and AI system GPT, BERT, T5, LLaMA, and beyond. Yet most practitioners treat it as a black box.

**The goals of this project:**
- Deeply understand *why* attention works, not just *how* to call `nn.MultiheadAttention`
- Build intuition for architectural decisions: why layer norm? why positional encoding? why multi-head?
- Demonstrate ML depth beyond calling pre-built APIs a critical skill for ML engineering roles

---

## ✨ Features

- ✅ **Self-Attention** implemented from scratch (scaled dot-product attention)
- ✅ **Multi-Head Attention** with proper head splitting and concatenation
- ✅ **Sinusoidal Positional Encoding** using NumPy
- ✅ **Layer Normalization** implemented manually
- ✅ **Transformer Encoder** full stack with residual connections and FFN
- ✅ **Transformer Decoder** with masked self-attention and cross-attention
- ✅ **Full Encoder-Decoder Transformer** assembled end-to-end
- ✅ **Training Notebook** with a working training loop

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Core Framework | PyTorch 2.x |
| Numerical Ops | NumPy |
| Experimentation | Jupyter Notebooks |
| Language | Python 3.8+ |

---

## 🏗️ Architecture Overview

The implementation mirrors the original Transformer paper architecture:

```
Input Tokens
     │
     ▼
[Token Embedding + Positional Encoding]
     │
     ▼
┌─────────────────────────────────┐
│        ENCODER (×N layers)      │
│  ┌────────────────────────────┐ │
│  │  Multi-Head Self-Attention │ │
│  └────────────┬───────────────┘ │
│               │ + Residual      │
│  ┌────────────▼───────────────┐ │
│  │      Layer Norm            │ │
│  └────────────┬───────────────┘ │
│               │                 │
│  ┌────────────▼───────────────┐ │
│  │  Feed-Forward Network      │ │
│  └────────────┬───────────────┘ │
│               │ + Residual      │
│  ┌────────────▼───────────────┐ │
│  │      Layer Norm            │ │
│  └────────────────────────────┘ │
└─────────────────────────────────┘
     │  Encoder Output (Memory)
     ▼
┌─────────────────────────────────┐
│        DECODER (×N layers)      │
│  Masked Self-Attention          │
│  Cross-Attention (w/ Encoder)   │
│  Feed-Forward Network           │
│  Layer Norm + Residual ×3       │
└─────────────────────────────────┘
     │
     ▼
[Linear + Softmax → Output Tokens]
```

**Key design decisions aligned with the original paper:**
- **Scaled dot-product attention**: divides by √d_k to prevent vanishing gradients in softmax
- **Multi-head attention**: allows the model to jointly attend to different representation subspaces
- **Sinusoidal positional encoding**: enables the model to generalize to sequence lengths not seen during training
- **Post-layer normalization**: applied after each sub-layer with residual connections

---

## 📁 Project Structure

```
Transformers/
│
├── 01_Self_Attention.ipynb           # Scaled dot-product attention from scratch
├── 02_Mutlihead_Attention.ipynb      # Multi-head attention with head splitting/concat
├── 03_Positional_Encoding.ipynb      # Sinusoidal positional encoding (NumPy)
├── 04_Layer_Normalization.ipynb      # Manual layer norm with visualization
│
├── 05_Transformer_Encoder.py         # Full encoder block (Attention + FFN + Residual)
├── 06_Transformer_Decoder.py         # Full decoder block (masked + cross-attention)
├── 07_Transformer.py                 # Complete encoder-decoder Transformer model
│
└── 08_Transformer_Trainer_Notebook.ipynb  # Training loop and end-to-end demo
```

**Convention:** Files `01–04` are exploratory notebooks for building intuition; files `05–07` are clean, reusable Python modules; file `08` ties everything together in a training pipeline.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/amey-21/Transformers.git
cd Transformers

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy jupyter matplotlib
```

> **Requirements:** Python 3.8+, PyTorch 2.x, NumPy 1.24+

---
## 🧠 Challenges & Learnings

**Key challenges encountered:**

- **Attention masking**: Correctly implementing the causal (look-ahead) mask in the decoder to prevent the model from attending to future tokens during training
- **Multi-head reshaping**: Getting the batch dimensions right when splitting and recombining attention heads across `(batch, heads, seq_len, d_k)`
- **Numerical stability**: Understanding *why* scaling by √d_k is necessary without it, dot products grow large and softmax gradients vanish
- **Residual connections**: Ensuring tensor shapes are compatible across the skip connections, particularly after projection layers

**Key learnings:**
- The Transformer's power comes from *parallel* attention computation vs. sequential RNN processing
- Positional encoding doesn't need to be learned fixed sinusoids generalize well
- Layer norm applied *per token* (not per batch) is crucial for NLP stability
- The decoder's two attention layers serve distinct purposes: one for target context, one for source context

---

## 📄 License

This project is licensed under the **MIT License** feel free to use, adapt, and build on it.

---

## 📚 References

- Vaswani et al. (2017) [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) Jay Alammar
- [Harvard NLP The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)

---

> Built from scratch to understand what every weight, head, and mask is *actually doing* — because real ML engineers don't just call APIs.
