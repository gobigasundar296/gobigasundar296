import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', default='out_model', help='Path to trained model directory')
    p.add_argument('--prompt', default='Hello', help='Generation prompt')
    p.add_argument('--max_new_tokens', type=int, default=100)
    p.add_argument('--do_sample', action='store_true', help='Use sampling instead of greedy')
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top_k', type=int, default=50)
    p.add_argument('--top_p', type=float, default=0.95)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = get_args()
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f'Model directory not found: {model_dir}')
        return

    # CPU device
    device = torch.device('cpu')

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    # Tokenize prompt
    inputs = tokenizer(args.prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)

    # Generation settings
    gen_kwargs = {
        'max_new_tokens': args.max_new_tokens,
        'do_sample': args.do_sample,
        'pad_token_id': tokenizer.eos_token_id,
    }
    # Sampling-only params should not be passed during greedy decoding
    if args.do_sample:
        gen_kwargs.update({
            'temperature': args.temperature,
            'top_k': args.top_k,
            'top_p': args.top_p,
        })

    # For reproducibility when sampling
    if args.do_sample:
        torch.manual_seed(args.seed)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(input_ids, **gen_kwargs)

    # decode and print only the newly generated portion
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # If you want only new tokens appended to prompt, strip the prompt prefix
    if generated.startswith(args.prompt):
        suffix = generated[len(args.prompt):]
    else:
        suffix = generated

    print('\n=== Prompt ===')
    print(args.prompt)
    print('\n=== Generated ===')
    print(suffix.strip())


if __name__ == '__main__':
    main()
