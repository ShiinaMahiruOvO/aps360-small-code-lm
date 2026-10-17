"""Prepare the frozen BST examples or tokenize a local Python corpus."""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from code_lm.tokenizer import Tokenizer

EOS = "<|endoftext|>"


def read_rows(path):
    path = Path(path)
    if path.is_dir():
        for name in ("test_known.jsonl", "test_combinations.jsonl"):
            yield from read_rows(path / name)
        return
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def data_digest(path):
    """Hash a JSONL file or the two ordered evaluation splits in a directory."""
    path = Path(path)
    paths = [path / name for name in ("test_known.jsonl", "test_combinations.jsonl")] if path.is_dir() else [path]
    value = hashlib.sha256()
    for item in paths:
        value.update(item.read_bytes())
    return value.hexdigest()


def verify_covered(source):
    """Validate frozen data, suite hashes and full SFT specification coverage."""
    manifest = json.loads((source / "manifest.json").read_text())
    for name, digest in manifest["artifact_sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen dataset hash mismatch: {name}")
    suites = json.loads((source / "suites.json").read_text())
    splits = {}
    for split, count in manifest["counts"].items():
        rows = list(read_rows(source / f"{split}.jsonl"))
        if len(rows) != count or len({r['id'] for r in rows}) != count:
            raise ValueError(f"Invalid count or duplicate IDs: {split}")
        for row in rows:
            suite = suites[row['suite_id']]
            if hashlib.sha256(suite['tests'].encode()).hexdigest() != row['suite_id']:
                raise ValueError(f"Invalid test suite: {row['id']}")
            tree = ast.parse(row['answer'])
            if not any(isinstance(n, ast.FunctionDef) and n.name == row['entry_point'] for n in tree.body):
                raise ValueError(f"Missing entry point: {row['id']}")
        splits[split] = rows
    taught = {r['spec_id'] for r in splits['train']}
    evaluated = {r['spec_id'] for r in splits['test_known'] + splits['test_combinations']}
    if taught != evaluated:
        raise ValueError("SFT must cover every evaluation specification")
    print(json.dumps({'counts': manifest['counts'], 'taught_specs': len(taught), 'hashes_verified': True}))
    return manifest


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_tokenizer(path):
    return Tokenizer.from_files(path / "vocab.json", path / "merges.txt", [EOS])


def prepare_tokens(source, tokenizer_dir, output):
    """Stream {text, split} JSONL into uint16 token files; reject cross-split copies.

    Split must be train or validation. Normalize line endings, remove exact
    duplicates within each split, and append one EOS per source document.
    The bundled tokenizer stays fixed; validation never trains the vocabulary.
    """
    tokenizer = load_tokenizer(tokenizer_dir)
    if max(tokenizer.vocab) > np.iinfo(np.uint16).max:
        raise ValueError("Tokenizer does not fit uint16")
    eos = tokenizer.token_to_id[EOS.encode()]
    output.mkdir(parents=True, exist_ok=True)
    seen, counts = {}, Counter()
    paths = {s: output / f"{name}.bin" for s, name in [("train", "train"), ("validation", "val")]}
    temporary = {s: p.with_suffix(".bin.partial") for s, p in paths.items()}
    try:
        from contextlib import ExitStack
        with ExitStack() as stack:
            streams = {s: stack.enter_context(p.open("wb")) for s, p in temporary.items()}
            if source.is_dir():
                rows = ({**row, "split": split}
                        for name, split in [("train.jsonl", "train"), ("val.jsonl", "validation")]
                        for row in read_rows(source / name))
            else:
                rows = read_rows(source)
            for row in rows:
                split = row["split"]
                if split not in streams:
                    raise ValueError(f"Unknown split: {split}")
                text = row["text"].replace("\r\n", "\n").replace("\r", "\n").strip()
                if not text:
                    continue
                if EOS in text:
                    raise ValueError("Source text contains reserved EOS")
                digest = hashlib.sha256(text.encode()).hexdigest()
                if digest in seen:
                    if seen[digest] != split:
                        raise ValueError("Document duplicated across train and validation")
                    continue
                seen[digest] = split
                ids = tokenizer.encode(text) + [eos]
                np.asarray(ids, dtype=np.uint16).tofile(streams[split])
                counts[split] += len(ids)
        if not all(counts[s] for s in paths):
            raise ValueError("Both splits must contain tokens")
        for split in paths:
            temporary[split].replace(paths[split])
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
    print(json.dumps(dict(counts)))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "python-edu":
        from code_lm.data.python_corpus import main as prepare_python
        return prepare_python(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("python-edu", help="Download, clean and split Python-Edu; use python-edu --help")
    token_fit = sub.add_parser("tokenizer", help="Train a fresh BPE tokenizer on a training text file only")
    token_fit.add_argument("--train-text", type=Path, required=True)
    token_fit.add_argument("--output", type=Path, default=Path("data/generated/tokenizer"))
    token_fit.add_argument("--vocab-size", type=int, default=8192)
    bst = sub.add_parser("bst", help="Verify the final frozen dataset, or regenerate it in a new directory")
    bst.add_argument("--dataset", type=Path, default=Path("data/bst_covered"))
    bst.add_argument("--regenerate", action="store_true")
    bst.add_argument("--seed", type=int, default=360921)
    bst.add_argument("--output", type=Path)
    tokens = sub.add_parser("tokens")
    tokens.add_argument("--source", type=Path, required=True)
    tokens.add_argument("--tokenizer-dir", type=Path, default=Path("data/tokenizer"))
    tokens.add_argument("--output", type=Path, default=Path("data/processed/python"))
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--output", type=Path, default=Path("data/processed/smoke"))
    args = parser.parse_args()
    if args.action == "tokenizer":
        from code_lm.bpe import train_bpe
        from code_lm.tokenizer import _gpt2_bytes_to_unicode
        if args.output.exists():
            raise FileExistsError("Use a new tokenizer output directory")
        vocab, merges = train_bpe(args.train_text, args.vocab_size, [EOS])
        mapping = _gpt2_bytes_to_unicode()
        encode = lambda value: ''.join(mapping[b] for b in value)
        args.output.mkdir(parents=True)
        (args.output / 'vocab.json').write_text(json.dumps({encode(v): k for k, v in vocab.items()}))
        (args.output / 'merges.txt').write_text(''.join(f'{encode(a)} {encode(b)}\n' for a, b in merges))
        (args.output / 'manifest.json').write_text(json.dumps({
            'fit_split': 'train', 'source': str(args.train_text),
            'source_sha256': hashlib.sha256(args.train_text.read_bytes()).hexdigest(),
            'vocab_size': len(vocab)}, indent=2) + '\n')
    elif args.action == "bst":
        if args.regenerate:
            if args.output is None:
                parser.error("--regenerate requires --output pointing to a new directory")
            from code_lm.data.bst_dataset import build
            build(args.output, args.seed)
            verify_covered(args.output)
        elif args.output is not None:
            parser.error("Use --regenerate with --output, or --dataset to verify existing data")
        else:
            verify_covered(args.dataset)
    elif args.action == "tokens":
        prepare_tokens(args.source, args.tokenizer_dir, args.output)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        for split in ("train", "val"):
            rng.integers(0, 32, size=4096, dtype=np.uint16).tofile(args.output / f"{split}.bin")


if __name__ == "__main__":
    main()
