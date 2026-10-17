from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator, Sequence

import regex as re

from .bpe import Pair, iter_pretokenize, merge_pair


def _gpt2_bytes_to_unicode() -> dict[int, str]:
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    unicode_values = byte_values.copy()
    next_unicode_offset = 0

    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            unicode_values.append(256 + next_unicode_offset)
            next_unicode_offset += 1

    return dict(zip(byte_values, (chr(value) for value in unicode_values), strict=True))


class Tokenizer:
    """A byte-level BPE tokenizer built from a vocabulary and ordered merges."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[Pair],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])

        existing_tokens = set(self.vocab.values())
        for special_token in self.special_tokens:
            encoded = special_token.encode("utf-8")
            if encoded not in existing_tokens:
                self.vocab[len(self.vocab)] = encoded
                existing_tokens.add(encoded)

        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        if self.special_tokens:
            alternatives = "|".join(
                re.escape(token) for token in sorted(set(self.special_tokens), key=len, reverse=True)
            )
            self._special_pattern = re.compile(f"({alternatives})")
            self._special_token_set = set(self.special_tokens)
        else:
            self._special_pattern = None
            self._special_token_set = set()

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        bytes_to_unicode = _gpt2_bytes_to_unicode()
        unicode_to_bytes = {character: byte_value for byte_value, character in bytes_to_unicode.items()}

        with open(vocab_filepath, encoding="utf-8") as vocab_file:
            serialized_vocab = json.load(vocab_file)
        vocab = {
            int(token_id): bytes(unicode_to_bytes[character] for character in serialized_token)
            for serialized_token, token_id in serialized_vocab.items()
        }

        merges: list[Pair] = []
        with open(merges_filepath, encoding="utf-8") as merges_file:
            for line in merges_file:
                pieces = line.rstrip().split(" ")
                if len(pieces) != 2:
                    continue
                left, right = pieces
                merges.append(
                    (
                        bytes(unicode_to_bytes[character] for character in left),
                        bytes(unicode_to_bytes[character] for character in right),
                    )
                )

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _split_by_special_tokens(self, text: str) -> Iterator[str]:
        if self._special_pattern is None:
            yield text
            return

        for part in self._special_pattern.split(text):
            if part:
                yield part

    def _encode_pretoken(self, pretoken: bytes) -> Iterator[int]:
        word = tuple(bytes([byte_value]) for byte_value in pretoken)

        while len(word) >= 2:
            best_pair: Pair | None = None
            best_rank: int | None = None

            for pair in zip(word, word[1:]):
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break
            word = merge_pair(word, best_pair, best_pair[0] + best_pair[1])

        for token in word:
            yield self.token_to_id[token]

    def encode(self, text: str) -> list[int]:
        return list(self._encode(text))

    def _encode(self, text: str) -> Iterator[int]:
        for part in self._split_by_special_tokens(text):
            if part in self._special_token_set:
                yield self.token_to_id[part.encode("utf-8")]
                continue

            for pretoken in iter_pretokenize(part):
                yield from self._encode_pretoken(pretoken)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self._encode(chunk)

    def decode(self, ids: Sequence[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")
