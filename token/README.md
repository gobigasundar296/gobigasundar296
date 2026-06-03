# Tokenization Deep Dive: From Basics to Training a Small Transformer

A hands-on session covering tokenization end-to-end — from first principles to training a GPT-style language model.

## Session Structure

### Notebook 1: `01_tokenization_fundamentals.ipynb`
| Part | Topic | What You'll Build |
|------|-------|-------------------|
| 1 | Why Tokenization Matters | Conceptual overview |
| 2 | Character-Level Tokenization | `CharTokenizer` from scratch |
| 3 | Word-Level Tokenization | `WordTokenizer` from scratch |
| 4 | Subword Tokenization Concepts | BPE vs WordPiece vs Unigram theory |
| 5 | BPE from Scratch | `BPETokenizer` — full implementation |
| 6 | WordPiece from Scratch | `WordPieceTokenizer` — full implementation |

### Notebook 2: `02_transformers_and_training.ipynb`
| Part | Topic | What You'll Build |
|------|-------|-------------------|
| 7 | HuggingFace `tokenizers` Library | Train a BPE tokenizer with the Rust-backed library |
| 8 | Pre-trained Tokenizers (GPT-2 vs BERT) | Side-by-side comparison |
| 9 | Train a Custom BPE Tokenizer | On WikiText-2 dataset (vocab=8000) |
| 10 | Train a Small GPT-style LM | ~1M param Transformer from scratch |
| 11 | Key Takeaways | Summary + further reading |

## Setup

```bash
pip install -r requirements.txt
```

### Dependencies
- **PyTorch** >= 2.0
- **transformers** >= 4.35 (HuggingFace)
- **tokenizers** >= 0.15 (HuggingFace Rust tokenizers)
- **datasets** >= 2.14 (HuggingFace datasets)
- **matplotlib**, **numpy**, **tqdm**

## How to Run

1. Install dependencies: `pip install -r requirements.txt`
2. Open `01_tokenization_fundamentals.ipynb` and run all cells
3. Open `02_transformers_and_training.ipynb` and run all cells (Part 10 training takes ~5-15 min on CPU)

## Key Concepts Covered

- **Character tokenization** — simplest, no OOV, but very long sequences
- **Word tokenization** — short sequences, but huge vocab and OOV problem
- **BPE** (Byte Pair Encoding) — greedily merge most frequent character pairs (GPT-2/3/4, LLaMA)
- **WordPiece** — merge by mutual information score (BERT)
- **Byte-level BPE** — handle any UTF-8 input without UNK tokens
- **Causal language modeling** — predict next token, autoregressive generation
- **Transformer architecture** — embeddings, multi-head attention, feed-forward, layer norm
