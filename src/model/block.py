import torch
import torch.nn as nn
from src.config import cfg, device
from src.model.attention import MultiHeadAttention
from src.model.feedforward import FeedForward

class TransformerBlock(nn.Module):
    """
    One complete transformer layer.
    Combines multi-head attention + feedforward with
    layer normalization and residual connections.

    The residual connection (x = x + sublayer(x)) is critical —
    it creates a gradient highway through all 6 layers,
    preventing the vanishing gradient problem.

    Applied 6 times in our model (n_layers = 6).
    """
    def __init__(self):
        super().__init__()
        
        self.attention = MultiHeadAttention()
        self.feedForward = FeedForward()
        
        # LayerNorm — normalizes values before each sub-layer
        # Keeps activations in a stable range → stable training
        self.norm1 = nn.LayerNorm(cfg.n_embd)
        self.norm2 = nn.LayerNorm(cfg.n_embd)
        
    def forward(self, x):
        """
        x shape: (batch, seq_len, n_embd) — same in and out

        Pre-norm architecture (normalize BEFORE sub-layer):
            x = x + Attention(LayerNorm(x))
            x = x + FeedForward(LayerNorm(x))
        """
        # Residual + Attention
        x = x + self.attention(self.norm1(x))
        
        # Residual + FeedForward
        x = x + self.feedForward(self.norm2(x))
        return x 
        
    ### Residual connections — why they matter

'''
Without residuals, a 6-layer network has a serious problem. Gradients must flow backward 
through 6 layers during training. Each layer slightly shrinks them. By layer 6, 
gradients are nearly zero — the early layers stop learning. 
This is the **vanishing gradient problem**.

The residual connection `x = x + sublayer(x)` creates a shortcut:
```
Layer 6 gradient → straight back to Layer 1 via the + shortcut
                 → doesn't shrink through 6 multiplications
```

Mathematically — the gradient of `x + f(x)` with respect to `x` is `1 + ∂f/∂x`. 
That `1` guarantees gradients never vanish completely.

### LayerNorm — why it matters

Before each sub-layer, normalize the values to have mean≈0 and std≈1. This keeps activations in a healthy range. Without it, values can explode or shrink across layers, making training unstable.
```
LayerNorm(x) = (x - mean(x)) / (std(x) + ε)  × γ + β

'''