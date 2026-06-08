import os
import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name_or_path', default='gpt2', help='Pretrained model')
    p.add_argument('--train_file', default=None, help='Path to train.jsonl (overrides default)')
    p.add_argument('--output_dir', default='./out_model', help='Where to save checkpoints')
    p.add_argument('--per_device_train_batch_size', type=int, default=4)
    p.add_argument('--gradient_accumulation_steps', type=int, default=1,
                   help='Number of steps to accumulate gradients before an optimizer update')
    p.add_argument('--num_train_epochs', type=float, default=3.0)
    p.add_argument('--learning_rate', type=float, default=5e-5)
    p.add_argument('--max_length', type=int, default=512)
    p.add_argument('--save_steps', type=int, default=500)
    p.add_argument('--use_peft', action='store_true', help='Use PEFT/LoRA if available')
    p.add_argument('--logging_steps', type=int, default=50)
    p.add_argument('--debug_run', action='store_true',
                   help='Quick smoke test: use a tiny subset of the data and cap training steps')
    return p.parse_args()


def main():
    args = get_args()
    repo_root = Path(__file__).resolve().parents[1]
    default_train = repo_root / 'data' / 'lm' / 'train.jsonl'
    train_path = Path(args.train_file) if args.train_file else default_train

    if not train_path.exists():
        print(f'Training file not found: {train_path}')
        return

    print('Loading dataset...')
    ds = load_dataset('json', data_files={'train': str(train_path)})
    if 'text' not in ds['train'].column_names:
        # try to auto-detect single column
        print('Input jsonl must have a "text" field per line')
        return

    print('Loading tokenizer and model...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    # GPT2-like models don't have pad token by default
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token or '<pad>'})

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    # Resize token embeddings if tokenizer changed
    model.resize_token_embeddings(len(tokenizer))

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(examples['text'], return_attention_mask=False)

    tok_ds = ds.map(tokenize_fn, batched=True, remove_columns=ds['train'].column_names)

    # Group into blocks of max_length
    def group_texts(examples):
        concatenated = []
        for v in examples.values():
            concatenated.extend(v)
        joined = ''.join([tokenizer.decode(x) if isinstance(x, list) else str(x) for x in concatenated])
        # fallback: use tokenized flattening approach
        input_ids = []
        for arr in examples['input_ids']:
            input_ids.extend(arr)
        total_length = len(input_ids)
        if total_length >= args.max_length:
            total_length = (total_length // args.max_length) * args.max_length
        else:
            total_length = total_length
        result = {'input_ids': [], 'labels': []}
        for i in range(0, total_length, args.max_length):
            chunk = input_ids[i:i+args.max_length]
            result['input_ids'].append(chunk)
            result['labels'].append(list(chunk))
        return result

    # Better grouping using the standard approach
    def group_texts_tokens(examples):
        # Concatenate all input_ids
        all_input_ids = sum(examples['input_ids'], [])
        total_length = len(all_input_ids)
        if total_length >= args.max_length:
            total_length = (total_length // args.max_length) * args.max_length
        chunks = {
            'input_ids': [],
            'labels': [],
        }
        for i in range(0, total_length, args.max_length):
            chunk = all_input_ids[i:i+args.max_length]
            chunks['input_ids'].append(chunk)
            chunks['labels'].append(chunk.copy())
        return chunks

    grouped = tok_ds['train'].map(group_texts_tokens, batched=True, remove_columns=tok_ds['train'].column_names)

    # Debug runs only need a tiny slice of the data
    max_steps = -1
    if args.debug_run:
        n_keep = min(len(grouped), 8)
        grouped = grouped.select(range(n_keep))
        max_steps = 5
        print(f'Debug run: using {n_keep} blocks and capping training at {max_steps} steps')

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=max_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        optim='adamw_torch',
    )

    # Optional PEFT/LoRA setup (best-effort)
    if args.use_peft:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=['q_proj', 'v_proj'] if hasattr(model, 'transformer') else None,
                lora_dropout=0.05,
                bias='none',
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
            print('Enabled PEFT/LoRA')
        except Exception as e:
            print('PEFT requested but not available or failed to initialize:', e)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=grouped,
        data_collator=data_collator,
    )

    print('Starting training...')
    trainer.train()

    print('Saving final model...')
    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    main()
