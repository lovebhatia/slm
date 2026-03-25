# src/data/bpe_tokenizer.py
import json
import os
from collections import Counter


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer.
    Learns frequent subword patterns from training text
    and merges them into single tokens.
    Handles encode, decode, save and load.
    """

    SPECIAL_TOKENS = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size  = vocab_size
        self.token_to_id = dict(self.SPECIAL_TOKENS)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.merges: dict = {}

    # ── Private helpers ──────────────────────────────────────

    def _get_initial_vocab(self, text: str) -> list:
        """Split text into character-level tokens with end-of-word marker."""
        return [
            tuple(list(word) + ["</w>"])
            for word in text.strip().split()
        ]

    def _get_pair_counts(self, vocab: list) -> Counter:
        """Count frequency of every adjacent token pair."""
        pairs = Counter()
        for word in vocab:
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += 1
        return pairs

    def _merge_pair(self, pair: tuple, vocab: list) -> list:
        """Replace every occurrence of pair with merged token."""
        merged = ''.join(pair)
        new_vocab = []
        for word in vocab:
            new_word, i = [], 0
            while i < len(word):
                if i < len(word)-1 and (word[i], word[i+1]) == pair:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab.append(tuple(new_word))
        return new_vocab

    # ── Public API ───────────────────────────────────────────

    def train(self, text: str):
        """Run the BPE training loop on raw text."""
        print(f"Training BPE — target vocab size: {self.vocab_size}")
        vocab   = self._get_initial_vocab(text)
        next_id = len(self.token_to_id)

        # Seed with all unique characters
        for word in vocab:
            for ch in word:
                if ch not in self.token_to_id:
                    self.token_to_id[ch] = next_id
                    self.id_to_token[next_id] = ch
                    next_id += 1

        num_merges = self.vocab_size - len(self.token_to_id)
        for i in range(num_merges):
            pairs = self._get_pair_counts(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            if pairs[best] < 2:
                break
            merged = ''.join(best)
            self.merges[best]         = merged
            self.token_to_id[merged]  = next_id
            self.id_to_token[next_id] = merged
            next_id += 1
            vocab = self._merge_pair(best, vocab)
            if (i + 1) % 500 == 0:
                print(f"  Merge {i+1}/{num_merges} — '{merged}' (freq: {pairs[best]})")

        self.vocab_size = len(self.token_to_id)
        print(f"Training complete. Final vocab size: {self.vocab_size}")

    def encode(self, text: str) -> list:
        """Convert string → list of token IDs."""
        ids = [self.SPECIAL_TOKENS["<BOS>"]]
        for word in text.strip().split():
            chars = tuple(list(word) + ["</w>"])
            while len(chars) > 1:
                pairs     = [(chars[i], chars[i+1]) for i in range(len(chars)-1)]
                mergeable = [p for p in pairs if p in self.merges]
                if not mergeable:
                    break
                best = min(mergeable, key=lambda p: list(self.merges.keys()).index(p))
                merged, new = ''.join(best), []
                i = 0
                while i < len(chars):
                    if i < len(chars)-1 and (chars[i], chars[i+1]) == best:
                        new.append(merged); i += 2
                    else:
                        new.append(chars[i]); i += 1
                chars = tuple(new)
            for token in chars:
                ids.append(self.token_to_id.get(token, self.SPECIAL_TOKENS["<UNK>"]))
        ids.append(self.SPECIAL_TOKENS["<EOS>"])
        return ids

    def decode(self, ids: list) -> str:
        """Convert list of token IDs → string."""
        tokens = [
            self.id_to_token[i]
            for i in ids
            if i in self.id_to_token and self.id_to_token[i] not in self.SPECIAL_TOKENS
        ]
        return ''.join(tokens).replace("</w>", " ").strip()

    def save(self, path: str):
        """Persist tokenizer to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size" : self.vocab_size,
                "token_to_id": self.token_to_id,
                "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
                "merges"     : {f"{k[0]}|||{k[1]}": v for k, v in self.merges.items()},
            }, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved → {path}")

    def load(self, path: str):
        """Load tokenizer from JSON."""
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        self.vocab_size  = d["vocab_size"]
        self.token_to_id = d["token_to_id"]
        self.id_to_token = {int(k): v for k, v in d["id_to_token"].items()}
        self.merges      = {tuple(k.split("|||")): v for k, v in d["merges"].items()}
        print(f"Tokenizer loaded. Vocab size: {self.vocab_size}")