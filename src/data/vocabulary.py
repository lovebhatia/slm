# src/data/vocabulary.py


class Vocabulary:
    """
    Character-level vocabulary.
    Scans raw text and assigns every unique character a unique integer ID.
    Foundation for the BPE tokenizer.
    """

    def __init__(self):
        self.char_to_id: dict = {}
        self.id_to_char: dict = {}
        self.vocab_size: int  = 0

    def build(self, text: str):
        """Build vocabulary from raw text."""
        unique_chars = sorted(set(text))
        for idx, char in enumerate(unique_chars):
            self.char_to_id[char] = idx
            self.id_to_char[idx]  = char
        self.vocab_size = len(unique_chars)
        print(f"Vocabulary built: {self.vocab_size} unique characters")

    def encode(self, text: str) -> list:
        """Convert string → list of integers."""
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids: list) -> str:
        """Convert list of integers → string."""
        return ''.join([self.id_to_char[i] for i in ids])