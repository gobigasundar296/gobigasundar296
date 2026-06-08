import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss  # optional accelerator; not required for this small demo
except ImportError:
    faiss = None


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    chunks_file = os.path.join(repo_root, 'data', 'chunks.jsonl')
    out_dir = os.path.join(repo_root, 'data', 'faiss')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(chunks_file):
        print(f'Chunks file not found: {chunks_file}\nRun prepare_data.py first to create chunks.jsonl')
        return

    items = []
    texts = []
    with open(chunks_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            obj = json.loads(line)
            items.append(obj)
            texts.append(obj.get('text', ''))

    if not texts:
        print('No texts to embed.')
        return

    model = SentenceTransformer('all-MiniLM-L6-v2')
    print('Computing embeddings...')
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Ensure float32 and L2-normalize so a dot product == cosine similarity
    embeddings = embeddings.astype('float32')
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)

    # Primary store: a plain NumPy matrix (no extra dependencies)
    emb_path = os.path.join(out_dir, 'embeddings.npy')
    np.save(emb_path, embeddings)

    meta_path = os.path.join(out_dir, 'meta.jsonl')
    with open(meta_path, 'w', encoding='utf-8') as fout:
        for it in items:
            fout.write(json.dumps(it, ensure_ascii=False) + '\n')

    print(f'Wrote embeddings to {emb_path} and metadata to {meta_path}')

    # Optional: also build a FAISS index if the library is available
    if faiss is not None:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        index_path = os.path.join(out_dir, 'index.faiss')
        faiss.write_index(index, index_path)
        print(f'FAISS available - also wrote index to {index_path}')
    else:
        print('FAISS not installed - using NumPy search (fine for this small dataset)')


if __name__ == '__main__':
    main()
