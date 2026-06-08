Short local LM fine-tune and usage

This workspace contains a minimal pipeline to fine-tune a small causal LM (distilgpt2/gpt2) on local text data and use the resulting model for generation. The scripts are tuned for CPU use by default.

Quick files
- data/docs/*.txt   -- your source documents (Pandyahistory.txt present)
- data/chunks.jsonl -- paragraph chunks (created by prepare_data.py)
- data/lm/train.jsonl, train.txt -- training data (created by prepare_lm_data.py)
- scripts/train_lm.py -- CPU-friendly HF Trainer fine-tune script
- scripts/inspect_model.py -- tokenization/forward pass inspector
- scripts/generate.py -- small script to generate from out_model
- out_model/ -- saved model produced by training

Commands
1) Prepare chunks from docs:
   python scripts/prepare_data.py

2) Prepare LM training data:
   python scripts/prepare_lm_data.py

3) Quick CPU demo training (small model, 1 epoch, short seq, frequent saves):
   python scripts/train_lm.py --model_name_or_path distilgpt2 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --num_train_epochs 1 --max_length 128 --save_steps 10 --logging_steps 5 --debug_run

4) Full training (adjust options as needed):
   python scripts/train_lm.py --model_name_or_path gpt2 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --num_train_epochs 3 --max_length 256 --save_steps 100

5) Inspect tokenizer / logits / attentions:
   python scripts/inspect_model.py

6) Generate from the trained model:
   python scripts/generate.py --model_dir out_model --prompt "Your prompt here" --max_new_tokens 100

RAG (retrieval-augmented generation) - answer from your documents
7) Build the embedding index over chunks.jsonl (run prepare_data.py first):
   python scripts/build_embeddings.py

8) Ask a question; relevant chunks are retrieved and fed to the model as context:
   python scripts/query_rag.py --query "Who were the Pandya rulers?" --top_k 3 --show_context

   The generator defaults to google/flan-t5-base (instruction-tuned, gives real answers from context). Use --model_dir out_model for your fine-tuned causal LM, or --extractive to skip generation and return the top retrieved chunk verbatim (always faithful).

   Useful flags: --model_dir <dir-or-hf-id>, --extractive, --do_sample, --temperature 0.8, --repetition_penalty 1.3, --no_repeat_ngram_size 3, --max_new_tokens 120

Notes and tips
- CPU is slow. Use small models (distilgpt2) and small max_length for experiments.
- If you have low RAM, keep batch size 1 and use gradient_accumulation_steps to simulate larger batches.
- To use PEFT/LoRA: install peft and pass --use_peft to train_lm.py; this can reduce memory and time by training adapters only.
- Required packages: torch, transformers, datasets, sentence-transformers (optional), faiss-cpu (optional), peft (optional).

If you want, I can further reduce defaults or produce a one-shot script that runs prepare_data -> prepare_lm_data -> train on a tiny subset for fast demo.