"""Index builder that imports retrieval by path, avoiding project package collisions."""
import argparse
from pathlib import Path
import importlib.util

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='Directory containing data/ and index/')
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    p = root / 'retrieval.py'
    spec = importlib.util.spec_from_file_location('rag_retrieval', p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(module.build_index(root=root))

if __name__ == '__main__':
    main()
