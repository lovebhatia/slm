import math

import torch
import torch.nn as nn
from src.config import cfg, device
from src.model.block import TransformerBlock
from src.model.embedding import Embedding

class SLM(nn.Module):
    """
    The complete Small Language Model.

    Architecture:
        1. Embedding (token + position)
        2. TransformerBlock × n_layers (6)
        3. Final LayerNorm
        4. Linear head: n_embd → vocab_size
        5. Softmax → next token probabilities

    Given a sequence of token IDs, predicts the next token.
    During training: computes loss against actual next tokens.
    During inference: samples from the probability distribution.
    """

    def __init__(self):
        super().__init__()

        self.embedding = Embedding()

        # Stack n_layers transformer blocks
        self.blocks = nn.Sequential(
            *[TransformerBlock() for _ in range(cfg.n_layers)]
        )

        # Final layer norm before the output head
        self.final_norm = nn.LayerNorm(cfg.n_embd)

        # Language model head — projects to vocab size
        # Each position outputs scores for all 8000 tokens
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Initialize weights properly
        self.apply(self._init_weights)

        print(f"SLM initialized — {self._count_params():,} parameters")

    def _init_weights(self, module):
        """
        Proper weight initialization.
        nn.Linear and nn.Embedding start with small random values.
        This prevents activations from being too large at step 0.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _count_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, token_ids, targets=None):
        """
        token_ids: (batch, seq_len) — input token IDs
        targets:   (batch, seq_len) — next token IDs (for training)

        Returns:
            logits: (batch, seq_len, vocab_size) — raw scores
            loss:   scalar if targets provided, else None
        """
        # 1. Token + position embeddings
        x = self.embedding(token_ids)          # (B, T, n_embd)

        # 2. Pass through all transformer blocks
        x = self.blocks(x)                     # (B, T, n_embd)

        # 3. Final normalization
        x = self.final_norm(x)                 # (B, T, n_embd)

        # 4. Project to vocabulary size
        logits = self.lm_head(x)               # (B, T, vocab_size)

        # 5. Compute loss if targets are provided (training mode)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            # Flatten for cross entropy: (B×T, vocab_size) vs (B×T,)
            loss = torch.nn.functional.cross_entropy(
                logits.view(B * T, V),
                targets.view(B * T)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens=50, temperature=1.0):
        """
        Generates new tokens given a starting sequence.
        Used at inference time — no gradient computation needed.

        temperature: 1.0 = normal, <1.0 = more confident, >1.0 = more creative
        """
        for _ in range(max_new_tokens):
            # Crop to context length if needed
            ids_crop = token_ids[:, -cfg.context_len:]

            # Forward pass — no loss needed
            logits, _ = self(ids_crop)

            # Take only the last position's predictions
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)

            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append to sequence
            token_ids = torch.cat([token_ids, next_token], dim=1)

        return token_ids
    
if __name__ == "__main__":
    from config import cfg, device

    # Build the full model
    model = SLM().to(device)

    # Test forward pass
    fake_input   = torch.randint(0, cfg.vocab_size, (2, 32)).to(device)
    fake_targets = torch.randint(0, cfg.vocab_size, (2, 32)).to(device)

    logits, loss = model(fake_input, targets=fake_targets)

    print(f"\nInput shape  : {fake_input.shape}")
    print(f"Logits shape : {logits.shape}")
    print(f"Loss         : {loss.item():.4f}")
    print(f"Expected loss: ~{math.log(cfg.vocab_size):.4f}  (random = log(vocab_size))")

    # Test generation
    seed = torch.zeros((1, 1), dtype=torch.long).to(device)
    generated = model.generate(seed, max_new_tokens=10)
    print(f"\nGenerated shape: {generated.shape}")
    print(f"Generated IDs  : {generated[0].tolist()}")
    print("\nModel is working end-to-end!")