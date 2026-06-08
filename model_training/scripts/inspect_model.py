import os
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_texts(path, n=2):
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            try:
                obj = json.loads(line)
                texts.append(obj.get('text', ''))
            except Exception:
                continue
    return texts


def main():
    repo = Path(__file__).resolve().parents[1]
    train_jsonl = repo / 'data' / 'lm' / 'train.jsonl'
    model_dir = repo / 'out_model'
    if not model_dir.exists():
        print('out_model not found, falling back to distilgpt2 from HF')
        model_dir = 'distilgpt2'

    print('Loading tokenizer and model from', str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    # Use the eager attention implementation so output_attentions actually
    # returns the attention weights (SDPA/flash backends return None).
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), attn_implementation='eager')
    model.eval()

    texts = load_texts(str(train_jsonl), n=2)
    if not texts:
        texts = ["Hello world.", "The quick brown fox jumps over the lazy dog."]

    for i, txt in enumerate(texts):
        print('\n=== Sample', i, '===')
        print('Text:', txt[:200])
        enc = tokenizer(txt, return_tensors='pt')
        input_ids = enc['input_ids']
        print('Input IDs:', input_ids.tolist())
        toks = [tokenizer.convert_ids_to_tokens(x) for x in input_ids[0].tolist()]
        print('Tokens:', toks)
        print('Length:', input_ids.shape[1])

        with torch.no_grad():
            outputs = model(**enc, output_attentions=True)
            logits = outputs.logits
            attentions = outputs.attentions

        print('Logits shape:', logits.shape)
        # next-token distribution for last token
        last_logits = logits[0, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        topk = torch.topk(probs, k=10)
        top_tokens = [tokenizer.decode([int(i)]) for i in topk.indices.tolist()]
        print('Top-10 next-token predictions:')
        for tok, p in zip(top_tokens, topk.values.tolist()):
            print(f'  {tok!r}: {p:.4f}')

        if attentions:
            print('Number of attention tensors (layers):', len(attentions))
            for li, a in enumerate(attentions):
                # a shape: (batch, num_heads, seq_len, seq_len)
                print(f' Layer {li}:', tuple(a.shape))
            # show averaged attention for last layer, last head
            last_att = attentions[-1][0]  # (num_heads, seq_len, seq_len)
            avg_head = last_att.mean(0)  # (seq_len, seq_len)
            seq_len = avg_head.shape[0]
            print(' Averaged attention matrix shape:', tuple(avg_head.shape))
            if seq_len <= 50:
                print(' Averaged attention (rows=to, cols=from):')
                # pretty print small matrix
                for row in avg_head.tolist():
                    print('  ', ' '.join(f'{v:.3f}' for v in row))


if __name__ == '__main__':
    main()
