import torch
import torch.nn as nn
from src.config import cfg, device

class Embedding(nn.Module):
    """
    Converts token IDs into vectors and adds position information.

    Two things happen here:
    1. Token embedding  : integer ID → vector of n_embd numbers
    2. Position encoding: adds a unique signal for each position 0..context_len

    Both are learned during training — the model figures out
    the best representations on its own.
    """

    def __init__(self):
        super().__init__()

        # Token embedding table — shape: (vocab_size, n_embd)
        # Think of it as a lookup table with vocab_size rows
        # Each row is a vector representing one token
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)

        # Position embedding table — shape: (context_len, n_embd)
        # One vector per position — learned during training
        self.pos_emb = nn.Embedding(cfg.context_len, cfg.n_embd)

        # Dropout for regularization — randomly zeros some values
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, token_ids):
        """
        token_ids shape: (batch_size, sequence_length)
        output shape  : (batch_size, sequence_length, n_embd)

        forward() is called automatically when you do: embedding(token_ids)
        """
        batch_size, seq_len = token_ids.shape

        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=device)

        # Look up token vectors + position vectors
        token_vectors   = self.token_emb(token_ids)   # (B, T, n_embd)
        position_vectors = self.pos_emb(positions)    # (T, n_embd)

        # Add them together — broadcasting handles the batch dimension
        x = token_vectors + position_vectors          # (B, T, n_embd)

        return self.dropout(x)
    
if __name__ == "__main__":
    from config import cfg, device

    print(f"Device: {device}")
    print(f"Vocab size: {cfg.vocab_size}")
    print(f"Embedding dim: {cfg.n_embd}")

    # Create the embedding layer
    emb = Embedding().to(device)

    # Fake a batch of 2 sequences, each 10 tokens long
    fake_tokens = torch.randint(0, cfg.vocab_size, (2, 10)).to(device)
    print(f"\nInput shape  : {fake_tokens.shape}  (batch=2, seq_len=10)")

    # Run through embedding
    output = emb(fake_tokens)
    print(f"Output shape : {output.shape}  (batch=2, seq_len=10, embd=256)")

    # Count parameters
    params = sum(p.numel() for p in emb.parameters())
    print(f"\nEmbedding parameters: {params:,}")
    print(f"  token_emb : {cfg.vocab_size} × {cfg.n_embd} = {cfg.vocab_size * cfg.n_embd:,}")
    print(f"  pos_emb   : {cfg.context_len} × {cfg.n_embd} = {cfg.context_len * cfg.n_embd:,}")