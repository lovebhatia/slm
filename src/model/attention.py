import torch
import torch.nn as nn
from src.config import cfg, device
from src.model.embedding import Embedding

class AttentionHead(nn.Module):
    """
    One single attention head.
    Learns one type of relationship between tokens.

    head_size: how many dimensions this head works in
               = n_embd / n_heads = 256 / 8 = 32
    """

    def __init__(self, head_size: int):
        super().__init__()

        # Three linear projections — no bias needed
        # Each takes n_embd (256) → head_size (32)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.key   = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)

        # Causal mask — stored as a constant, not a parameter
        # Lower triangular matrix of ones — prevents looking at future tokens
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.context_len, cfg.context_len))
        )

        self.dropout = nn.Dropout(cfg.dropout)
        self.head_size = head_size

    def forward(self, x):
        """
        x shape: (batch, seq_len, n_embd)
        output : (batch, seq_len, head_size)
        """
        B, T, C = x.shape   # batch, time(seq), channels(n_embd)

        # Project input into Q, K, V spaces
        Q = self.query(x)   # (B, T, head_size)
        K = self.key(x)     # (B, T, head_size)
        V = self.value(x)   # (B, T, head_size)

        # Compute attention scores — Q times K transposed
        # Scale by sqrt(head_size) to keep gradients stable
        scale = self.head_size ** -0.5
        scores = Q @ K.transpose(-2, -1) * scale   # (B, T, T)

        # Apply causal mask — set future positions to -infinity
        # softmax of -inf = 0, so future tokens get zero attention
        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float('-inf')
        )

        # Softmax — convert scores to probabilities that sum to 1
        weights = torch.softmax(scores, dim=-1)   # (B, T, T)
        weights = self.dropout(weights)

        # Weighted sum of values
        out = weights @ V   # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """
    Runs n_heads attention heads in parallel.
    Concatenates their outputs and projects back to n_embd.

    This is what lets the model learn multiple types of
    relationships simultaneously — grammar, meaning,
    facts, numerical patterns — all at once.
    """

    def __init__(self):
        super().__init__()

        # Each head works on n_embd / n_heads dimensions
        head_size = cfg.n_embd // cfg.n_heads   # 256 // 8 = 32

        # Create n_heads attention heads — stored as a list
        self.heads = nn.ModuleList([
            AttentionHead(head_size)
            for _ in range(cfg.n_heads)
        ])

        # After concatenating all heads: n_heads * head_size = n_embd
        # Project back to n_embd so the shape is preserved
        self.projection = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout    = nn.Dropout(cfg.dropout)

    def forward(self, x):
        """
        x shape: (batch, seq_len, n_embd)
        Each head outputs (batch, seq_len, head_size=32)
        Concatenated: (batch, seq_len, n_heads * head_size = 256)
        Projected back: (batch, seq_len, n_embd=256)
        """
        # Run all heads in parallel — concat along last dimension
        head_outputs = [head(x) for head in self.heads]
        combined = torch.cat(head_outputs, dim=-1)  # (B, T, n_embd)

        # Final linear projection + dropout
        out = self.dropout(self.projection(combined))
        return out
    
if __name__ == "__main__":
    from config import cfg, device

    # Test Embedding
    emb = Embedding().to(device)
    fake_tokens = torch.randint(0, cfg.vocab_size, (2, 10)).to(device)
    x = emb(fake_tokens)
    print(f"After Embedding     : {x.shape}")   # (2, 10, 256)

    # Test MultiHeadAttention
    attn = MultiHeadAttention().to(device)
    x_attn = attn(x)
    print(f"After Attention     : {x_attn.shape}")  # (2, 10, 256)

    # Count attention parameters
    attn_params = sum(p.numel() for p in attn.parameters())
    print(f"Attention parameters: {attn_params:,}")