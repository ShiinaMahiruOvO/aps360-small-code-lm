from __future__ import annotations

import heapq
import os
from collections import Counter, defaultdict
from dataclasses import dataclass

import regex as re


GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT2_RE = re.compile(GPT2_SPLIT_PATTERN)

type Token = bytes
type Pair = tuple[Token, Token]
type Word = tuple[Token, ...]


def iter_pretokenize(text: str):
    """Yield GPT-2-style pre-tokens as UTF-8 bytes."""
    for match in GPT2_RE.finditer(text):
        yield match.group(0).encode("utf-8")


def merge_pair(word: Word, pair: Pair, merged_token: Token) -> Word:
    """Replace non-overlapping occurrences of pair, scanning left to right."""
    left, right = pair
    output: list[Token] = []
    index = 0

    while index < len(word):
        if index + 1 < len(word) and word[index] == left and word[index + 1] == right:
            output.append(merged_token)
            index += 2
        else:
            output.append(word[index])
            index += 1

    return tuple(output)


def _adjacent_pair_counts(word: Word) -> Counter[Pair]:
    return Counter(zip(word, word[1:]))


def _ordinary_text_parts(text: str, special_tokens: list[str]):
    """Split on special tokens and yield only ordinary corpus text."""
    if not special_tokens:
        yield text
        return

    specials = set(special_tokens)
    alternatives = "|".join(re.escape(token) for token in sorted(specials, key=len, reverse=True))
    for part in re.split(f"({alternatives})", text):
        if part and part not in specials:
            yield part


@dataclass(frozen=True)
class _DescendingPair:
    """Reverse the pair comparison so heapq breaks ties by greatest byte pair."""

    value: Pair

    def __lt__(self, other: _DescendingPair) -> bool:
        return self.value > other.value


type HeapEntry = tuple[int, _DescendingPair, Pair]


def _push_pair(heap: list[HeapEntry], pair: Pair, count: int) -> None:
    if count > 0:
        heapq.heappush(heap, (-count, _DescendingPair(pair), pair))


def _pop_best_pair(heap: list[HeapEntry], pair_counts: Counter[Pair]) -> Pair | None:
    """Pop the best non-stale heap entry."""
    while heap:
        negative_count, _, pair = heapq.heappop(heap)
        count = -negative_count
        if count > 0 and pair_counts.get(pair, 0) == count:
            return pair
    return None


def _initialize_vocabulary(special_tokens: list[str]) -> tuple[dict[int, bytes], dict[bytes, int]]:
    vocab = {token_id: bytes([token_id]) for token_id in range(256)}
    token_to_id = {token: token_id for token_id, token in vocab.items()}

    for special_token in special_tokens:
        token = special_token.encode("utf-8")
        if token not in token_to_id:
            token_id = len(vocab)
            vocab[token_id] = token
            token_to_id[token] = token_id

    return vocab, token_to_id


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **_: object,
) -> tuple[dict[int, bytes], list[Pair]]:
    """Train a byte-level BPE vocabulary and return it with merge history."""
    vocab, token_to_id = _initialize_vocabulary(special_tokens)
    if vocab_size < len(vocab):
        raise ValueError(f"vocab_size must be at least {len(vocab)}, got {vocab_size}")

    with open(input_path, encoding="utf-8", newline="") as corpus_file:
        text = corpus_file.read()

    single_byte_tokens = tuple(bytes([value]) for value in range(256))
    word_counts: Counter[Word] = Counter()
    for part in _ordinary_text_parts(text, special_tokens):
        for pretoken in iter_pretokenize(part):
            word = tuple(single_byte_tokens[value] for value in pretoken)
            if len(word) >= 2:
                word_counts[word] += 1

    words = list(word_counts)
    frequencies = [word_counts[word] for word in words]
    pair_counts: Counter[Pair] = Counter()
    pair_occurrences: dict[Pair, set[int]] = defaultdict(set)

    for word_index, (word, frequency) in enumerate(zip(words, frequencies, strict=True)):
        for pair, occurrences in _adjacent_pair_counts(word).items():
            pair_counts[pair] += frequency * occurrences
            pair_occurrences[pair].add(word_index)

    heap: list[HeapEntry] = []
    for pair, count in pair_counts.items():
        _push_pair(heap, pair, count)

    merges: list[Pair] = []
    while len(vocab) < vocab_size:
        pair = _pop_best_pair(heap, pair_counts)
        if pair is None:
            break

        merged_token = pair[0] + pair[1]
        merges.append(pair)
        if merged_token not in token_to_id:
            token_id = len(vocab)
            vocab[token_id] = merged_token
            token_to_id[merged_token] = token_id

        affected_words = tuple(pair_occurrences.get(pair, ()))
        changed_pairs: set[Pair] = set()

        for word_index in affected_words:
            old_word = words[word_index]
            frequency = frequencies[word_index]
            old_pairs = _adjacent_pair_counts(old_word)

            for old_pair, occurrences in old_pairs.items():
                pair_counts[old_pair] -= frequency * occurrences
                pair_occurrences[old_pair].discard(word_index)
                changed_pairs.add(old_pair)

            new_word = merge_pair(old_word, pair, merged_token)
            words[word_index] = new_word

            for new_pair, occurrences in _adjacent_pair_counts(new_word).items():
                pair_counts[new_pair] += frequency * occurrences
                pair_occurrences[new_pair].add(word_index)
                changed_pairs.add(new_pair)

        for changed_pair in changed_pairs:
            count = pair_counts.get(changed_pair, 0)
            if count > 0:
                _push_pair(heap, changed_pair, count)
            else:
                pair_counts.pop(changed_pair, None)
                pair_occurrences.pop(changed_pair, None)

    return vocab, merges
