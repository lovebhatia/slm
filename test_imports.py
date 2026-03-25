# test_imports.py
from src.config import cfg, device
from src.data   import Vocabulary, BPETokenizer, DatasetBuilder, TextDataset
from src.model  import Embedding, MultiHeadAttention, FeedForward, TransformerBlock, SLM
from src.utils  import CheckpointManager, TrainingLogger
import torch

print(f"Device     : {device}")
print(f"Config     : vocab={cfg.vocab_size}, layers={cfg.n_layers}, embd={cfg.n_embd}")

model = SLM().to(device)
x     = torch.randint(0, cfg.vocab_size, (2, 32)).to(device)
y     = torch.randint(0, cfg.vocab_size, (2, 32)).to(device)

logits, loss = model(x, targets=y)
print(f"Logits     : {logits.shape}")
print(f"Loss       : {loss.item():.4f}")
print(f"\nAll imports working. Production structure ready.")