# src/config/config.py
import torch


class Config:
    # ── Model architecture ──────────────────────
    vocab_size   = 8000
    context_len  = 256
    n_embd       = 256
    n_heads      = 8
    n_layers     = 6
    dropout      = 0.1

    # ── Training ────────────────────────────────
    batch_size   = 32
    max_iters    = 5000
    eval_interval= 200
    learning_rate= 3e-4
    weight_decay = 0.1
    grad_clip    = 1.0

    # ── Paths ───────────────────────────────────
    data_dir      = "data/"
    models_dir    = "models/"
    checkpoint_dir= "checkpoints/"
    log_dir       = "logs/"

    # ── Device ──────────────────────────────────
    @staticmethod
    def get_device():
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


cfg    = Config()
device = cfg.get_device()