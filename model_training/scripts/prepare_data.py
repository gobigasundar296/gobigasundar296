import os
import json
import pathlib
from hashlib import sha1

def list_text_files(folder):
    p = pathlib.Path(folder)
    return [str(x) for x in p.glob('**/*.txt')]

def chunk_text(text):
    # simple paragraph split
    parts = [p.strip() for p in text.split('\n\n') if p.strip()]
    return parts

def make_id(text, source):
    return sha1((source + text).encode('utf-8')).hexdigest()

def main():
    scripts_dir = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(scripts_dir, '..'))
    docs_folder = os.path.join(repo_root, 'data', 'docs')
    out_file = os.path.join(repo_root, 'data', 'chunks.jsonl')
    files = list_text_files(docs_folder)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as fout:
        for fp in files:
            with open(fp, 'r', encoding='utf-8') as f:
                text = f.read()
            chunks = chunk_text(text)
            for c in chunks:
                item = {
                    'id': make_id(c, fp),
                    'text': c,
                    'source': os.path.relpath(fp, start=repo_root)
                }
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'Wrote chunks to {out_file}')

if __name__ == '__main__':
    main()
