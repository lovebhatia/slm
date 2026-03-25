# src/data/dataset_builder.py
import os
import re


class DatasetBuilder:
    """
    Collects and cleans raw text from multiple sources.
    Saves everything to a single train.txt for training.
    """

    def __init__(self, data_dir: str = "data/"):
        self.data_dir  = data_dir
        self.raw_texts = []
        self.total_words = 0
        os.makedirs(data_dir, exist_ok=True)

    def add_text(self, text: str, source: str = "manual"):
        cleaned = self._clean(text)
        if len(cleaned) > 20:
            self.raw_texts.append({"source": source, "text": cleaned})
            self.total_words += len(cleaned.split())

    def _clean(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\?\!\:\;\-\(\)\'\"/\%]', '', text)
        return text.strip()

    def get_stats(self):
        print(f"\n{'='*40}")
        print(f"Total samples : {len(self.raw_texts)}")
        print(f"Total words   : {self.total_words:,}")
        print(f"Est. tokens   : {int(self.total_words * 1.3):,}")
        sources = {}
        for item in self.raw_texts:
            sources[item['source']] = sources.get(item['source'], 0) + 1
        for src, count in sources.items():
            print(f"  {src:25s}: {count} samples")

    def save(self, filename: str = "train.txt") -> str:
        path = os.path.join(self.data_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(item["text"] for item in self.raw_texts))
        print(f"Dataset saved → {path}  ({os.path.getsize(path)/1024:.1f} KB)")
        return path