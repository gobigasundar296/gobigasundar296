import os
import json
import pathlib


def main():
    scripts_dir = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(scripts_dir, '..'))
    chunks_file = os.path.join(repo_root, 'data', 'chunks.jsonl')
    out_dir = os.path.join(repo_root, 'data', 'lm')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(chunks_file):
        print(f'Chunks file not found: {chunks_file}\nRun prepare_data.py first to create chunks.jsonl')
        return

    texts = []
    with open(chunks_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            obj = json.loads(line)
            text = obj.get('text', '').strip()
            if text:
                texts.append(text)

    # Write a simple concatenated training file (one paragraph separated by double newlines)
    train_txt = os.path.join(out_dir, 'train.txt')
    with open(train_txt, 'w', encoding='utf-8') as fout:
        fout.write('\n\n'.join(texts))

    # Also write a jsonl suitable for huggingface/datasets (one object per line with 'text')
    train_jsonl = os.path.join(out_dir, 'train.jsonl')
    with open(train_jsonl, 'w', encoding='utf-8') as fout:
        for t in texts:
            fout.write(json.dumps({'text': t}, ensure_ascii=False) + '\n')

    print(f'Wrote LM train files: {train_txt}, {train_jsonl}')


if __name__ == '__main__':
    main()
