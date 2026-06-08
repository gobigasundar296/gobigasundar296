import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

# Must match the embedding model used in build_embeddings.py
EMBED_MODEL = 'all-MiniLM-L6-v2'


def get_args():
    p = argparse.ArgumentParser(description='Retrieval-augmented generation over local chunks')
    p.add_argument('--query', required=True, help='Question / prompt to answer from your documents')
    p.add_argument('--model_dir', default='google/flan-t5-base',
                   help='Generator: local dir or HF id. Instruction-tuned seq2seq models '
                        '(e.g. google/flan-t5-base) give the best grounded answers. '
                        'Pass out_model to use your fine-tuned causal LM.')
    p.add_argument('--extractive', action='store_true',
                   help='Skip generation; return the top retrieved chunk verbatim (always faithful)')
    p.add_argument('--top_k', type=int, default=3, help='Number of chunks to retrieve')
    p.add_argument('--max_new_tokens', type=int, default=120)
    p.add_argument('--do_sample', action='store_true', help='Sample instead of greedy decoding')
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top_p', type=float, default=0.95)
    p.add_argument('--repetition_penalty', type=float, default=1.3,
                   help='>1.0 discourages repeating tokens (reduces looping)')
    p.add_argument('--no_repeat_ngram_size', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--show_context', action='store_true', help='Print the retrieved chunks')
    return p.parse_args()


def load_index(faiss_dir):
    emb_path = faiss_dir / 'embeddings.npy'
    meta_path = faiss_dir / 'meta.jsonl'
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f'Missing embeddings or metadata in {faiss_dir}.\n'
            'Run: python scripts/build_embeddings.py'
        )
    embeddings = np.load(str(emb_path)).astype('float32')
    meta = []
    with open(meta_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            meta.append(json.loads(line))
    return embeddings, meta


def retrieve(query, embedder, embeddings, meta, top_k):
    q_emb = embedder.encode([query], convert_to_numpy=True).astype('float32')[0]
    # L2-normalize the query so a dot product == cosine similarity
    q_emb = q_emb / max(float(np.linalg.norm(q_emb)), 1e-12)
    scores = embeddings @ q_emb  # (N,) cosine similarities
    top_k = min(top_k, len(scores))
    top_ids = np.argsort(-scores)[:top_k]
    hits = []
    for idx in top_ids.tolist():
        item = dict(meta[idx])
        item['score'] = float(scores[idx])
        hits.append(item)
    return hits


def build_prompt(query, hits):
    context = '\n\n'.join(f'[{i + 1}] {h["text"]}' for i, h in enumerate(hits))
    return (
        'Answer the question using only the context below. '
        'If the answer is not in the context, say you do not know.\n\n'
        f'Context:\n{context}\n\n'
        f'Question: {query}\n'
        'Answer:'
    )


def load_generator(model_dir):
    """Load a generator, auto-detecting seq2seq (e.g. flan-t5) vs causal (e.g. gpt2)."""
    config = AutoConfig.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    is_seq2seq = bool(getattr(config, 'is_encoder_decoder', False))
    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return tokenizer, model, is_seq2seq


def main():
    args = get_args()
    repo = Path(__file__).resolve().parents[1]
    faiss_dir = repo / 'data' / 'faiss'

    embeddings, meta = load_index(faiss_dir)

    print(f'Loading embedding model: {EMBED_MODEL}')
    embedder = SentenceTransformer(EMBED_MODEL)

    hits = retrieve(args.query, embedder, embeddings, meta, args.top_k)
    if not hits:
        print('No chunks retrieved. Did you build the index?')
        return

    print('\n=== Retrieved context ===')
    for i, h in enumerate(hits):
        src = h.get('source', '?')
        snippet = h['text'][:120].replace('\n', ' ')
        print(f'  [{i + 1}] score={h["score"]:.3f}  source={src}')
        if args.show_context:
            print(f'      {snippet}{"..." if len(h["text"]) > 120 else ""}')

    # Extractive mode: no generation, just return the most relevant chunk verbatim.
    if args.extractive:
        print('\n=== Question ===')
        print(args.query)
        print('\n=== Answer (top retrieved chunk, verbatim) ===')
        print(hits[0]['text'])
        return

    print(f'\nLoading generator: {args.model_dir}')
    tokenizer, model, is_seq2seq = load_generator(args.model_dir)
    print('Generator type:', 'seq2seq (instruction-tuned)' if is_seq2seq else 'causal LM')

    prompt = build_prompt(args.query, hits)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)

    gen_kwargs = {
        'max_new_tokens': args.max_new_tokens,
        'do_sample': args.do_sample,
        'repetition_penalty': args.repetition_penalty,
        'no_repeat_ngram_size': args.no_repeat_ngram_size,
        'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if args.do_sample:
        torch.manual_seed(args.seed)
        gen_kwargs.update({'temperature': args.temperature, 'top_p': args.top_p})

    with torch.no_grad():
        out_ids = model.generate(
            inputs['input_ids'], attention_mask=inputs['attention_mask'], **gen_kwargs
        )

    decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    if is_seq2seq:
        # Encoder-decoder models output only the answer, not the prompt.
        answer = decoded.strip()
    else:
        # Causal models echo the prompt; strip it to keep only the continuation.
        answer = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()

    print('\n=== Question ===')
    print(args.query)
    print('\n=== Answer (grounded in retrieved chunks) ===')
    print(answer)


if __name__ == '__main__':
    main()
