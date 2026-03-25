import torch
import torch.nn as nn
from src.config import cfg, device

class FeedForward(nn.Module):
    """
    Two linear layers with ReLU non-linearity between them.
    Applied independently to each token position.

    This is where the model "thinks" — after attention
    has gathered context from other tokens, FeedForward
    processes that information token by token.

    Architecture:
        n_embd (256) → 4×n_embd (1024) → ReLU → n_embd (256)
    """
    def __init__(self):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),  # expand: 256 → 1024
            nn.ReLU(),                              # non-linearity
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),  # contract: 1024 → 256
            nn.Dropout(cfg.dropout),
        )
        
    def forward(self, x):
        """
        x shape in : (batch, seq_len, n_embd)
        x shape out: (batch, seq_len, n_embd)  — same shape, richer content
        """
        return self.network(x)